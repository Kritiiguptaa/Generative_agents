from dataclasses import dataclass
from datetime import datetime, time, timezone
import csv
import json
import os
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Action:
	type: str
	symbol: Optional[str]
	quantity: int
	reason: str



def _market_is_open(market_state) -> bool:
	if hasattr(market_state, "is_open"):
		return bool(market_state.is_open)
	if hasattr(market_state, "current_time"):
		market_time = market_state.current_time.time()
		return time(9, 30) <= market_time < time(16, 0)
	return True


def _get_agent_name(agent_state) -> str:
	if hasattr(agent_state, "scratch") and hasattr(agent_state.scratch, "name"):
		return agent_state.scratch.name
	return getattr(agent_state, "name", "unknown")


def _get_tradeable_symbols(agent_state, market_state) -> List[str]:
	watchlist = getattr(agent_state.scratch, "watchlist", []) if hasattr(agent_state, "scratch") else []
	if watchlist:
		return list(watchlist)
	return list(getattr(market_state, "SYMBOLS", []))


def _get_holdings(agent_state) -> Dict[str, int]:
	positions = getattr(agent_state.scratch, "positions", {}) if hasattr(agent_state, "scratch") else {}
	return {symbol: int(pos.get("qty", 0)) for symbol, pos in positions.items()}


def _get_max_trade_size(agent_state, market_state) -> float:
	"""Risk cap alone, ignoring cash. Callers that show a number to the model
	must use _get_spendable_budget() instead -- see the note there."""
	if not hasattr(agent_state, "scratch"):
		return 0.0
	portfolio_value = agent_state.portfolio_value(market_state.current_prices)
	return portfolio_value * float(getattr(agent_state.scratch, "risk_limit_per_trade", 0.0))


def _get_spendable_budget(agent_state, market_state) -> float:
	"""
	What the agent can actually spend on one purchase: the risk cap AND the
	cash, whichever binds first.

	get_legal_actions() has always applied min(cash, risk_cap) when building the
	BUY menu, but build_prompt() printed the raw risk cap as "Max single-trade
	size". Marcus Webb was therefore shown "Max single-trade size: $5,017.00"
	while holding $76.41, with a legal list that contained no BUY at all. He
	requested `buy NVDA x10` ($2,030) on 20 near-consecutive steps. A prompt
	that advertises a budget the agent does not have manufactures exactly the
	infeasible requests this module exists to count.
	"""
	if not hasattr(agent_state, "scratch"):
		return 0.0
	cash = float(getattr(agent_state.scratch, "cash_balance", 0.0))
	return max(0.0, min(cash, _get_max_trade_size(agent_state, market_state)))


def get_legal_actions(agent_state, market_state) -> List[Action]:
	legal: List[Action] = []

	# HOLD is always legal
	legal.append(Action("HOLD", None, 0, "always available"))

	# NOTE: ANALYZE is deliberately NOT offered.
	#
	# It was added here briefly so both arms shared one action vocabulary, and
	# the measured result was unambiguous: every agent in both arms chose
	# ANALYZE on 100% of 600 decisions and placed zero trades. The arm that had
	# previously placed 16 trades placed none once it had the option.
	#
	# ANALYZE is a free escape hatch. A model can always justify gathering more
	# information, so an agent offered it never has to commit -- and an agent
	# that never commits cannot produce an action hallucination, which leaves
	# the metric with nothing to measure. HOLD already expresses "do nothing
	# this step" without implying a deferred decision.
	#
	# Both arms are kept symmetric by removing it from BOTH prompts rather than
	# adding it to both. See free_form_decision() in trading_reverie.py.

	# Market closed -> only HOLD
	if not _market_is_open(market_state):
		return legal

	# BUY: check cash, price, and max_trade_size
	tradeable = _get_tradeable_symbols(agent_state, market_state)
	cash_balance = float(getattr(agent_state.scratch, "cash_balance", 0.0))
	max_trade_size = _get_max_trade_size(agent_state, market_state)
	prices = getattr(market_state, "current_prices", {})
	for symbol in tradeable:
		price = prices.get(symbol)
		if not price:
			continue
		max_spend = min(cash_balance, max_trade_size)
		max_qty = int(max_spend // price)

		if max_qty > 0:
			legal.append(
				Action(
					"BUY",
					symbol,
					max_qty,
					f"has ${cash_balance} cash, limit ${max_trade_size}",
				)
			)

	# SELL: only if holdings for that ticker > 0
	for symbol, qty in _get_holdings(agent_state).items():
		if qty > 0:
			legal.append(
				Action(
					"SELL",
					symbol,
					qty,
					f"holds {qty} shares of {symbol}",
				)
			)

	return legal


def _format_legal_actions(legal_actions: List[Action]) -> str:
	lines: List[str] = []
	for action in legal_actions:
		if action.type == "HOLD":
			lines.append("- HOLD (place no trade this step)")
		elif action.type == "BUY":
			lines.append(f"- BUY {action.symbol} (max {action.quantity} shares)")
		elif action.type == "SELL":
			lines.append(f"- SELL {action.symbol} (max {action.quantity} shares)")
	return "\n".join(lines)


def build_prompt(agent_state, market_state, memory_context: str) -> Tuple[str, List[Action]]:
	legal_actions = get_legal_actions(agent_state, market_state)
	action_str = _format_legal_actions(legal_actions)
	name = _get_agent_name(agent_state)
	persona = getattr(agent_state.scratch, "trader_type", "") if hasattr(agent_state, "scratch") else ""
	innate = getattr(agent_state.scratch, "innate", "") if hasattr(agent_state, "scratch") else ""
	learned = getattr(agent_state.scratch, "learned", "") if hasattr(agent_state, "scratch") else ""
	currently = getattr(agent_state.scratch, "currently", "") if hasattr(agent_state, "scratch") else ""
	risk_tolerance = getattr(agent_state.scratch, "risk_tolerance", "") if hasattr(agent_state, "scratch") else ""
	cash_balance = float(getattr(agent_state.scratch, "cash_balance", 0.0))
	positions = _get_holdings(agent_state)
	market_open = "OPEN" if _market_is_open(market_state) else "CLOSED"
	prices = getattr(market_state, "current_prices", {})
	risk_cap = _get_max_trade_size(agent_state, market_state)
	budget = _get_spendable_budget(agent_state, market_state)
	# Derive the wording from the menu that was actually built, rather than from
	# budget > 0. A budget of $76.41 is nonzero but buys no share of any
	# tradeable symbol (cheapest is $150), so get_legal_actions() offers no BUY
	# while a bare budget figure still reads as an invitation to purchase. The
	# two must agree or the prompt contradicts its own action list again.
	can_buy = any(a.type == "BUY" for a in legal_actions)
	if not can_buy:
		budget_line = (f"You cannot afford to buy any share right now "
		               f"(spendable budget ${budget:,.2f}). Only HOLD or SELL "
		               f"are available.")
	else:
		budget_line = (f"Spendable budget for one purchase: ${budget:,.2f} "
		               f"(risk cap ${risk_cap:,.2f}, cash ${cash_balance:,.2f}, "
		               f"whichever is lower).")

	prompt = f"""
You are {name}, a {persona} trader.
Traits: {innate}
Background: {learned}
Current situation: {currently}
Risk tolerance: {risk_tolerance}
{budget_line}

Current state:
- Cash: ${cash_balance:,.2f}
- Holdings: {positions}

Market:
- Status: {market_open}
- Prices: {prices}

Memory:
{memory_context}

Rules:
- You cannot borrow money or trade on margin. A purchase's total cost
  (price x quantity) must not exceed your spendable budget above.
- You cannot sell shares you do not hold.

You MUST choose from ONLY these actions:
{action_str}

Respond ONLY in this JSON format:
{{"action": "BUY/SELL/HOLD", "symbol": "TICKER or null", "quantity": number, "reasoning": "why"}}
"""
	return prompt, legal_actions


def _strip_json_wrapper(response_text: str) -> str:
	"""
	Local instruct models (e.g. phi3:mini via Ollama) routinely wrap JSON
	answers in a markdown code fence -- ```json\n{...}\n``` -- or prefix them
	with a line of chatty preamble. json.loads() rejects both verbatim, which
	was previously flagging every syntactically-fine response as "not valid
	JSON" and forcing a fallback HOLD. Strip the fence if present, then fall
	back to slicing out the first {...} block.
	"""
	text = response_text.strip()
	if text.startswith("```"):
		text = text[3:]
		if text.lower().startswith("json"):
			text = text[4:]
		fence_end = text.rfind("```")
		if fence_end != -1:
			text = text[:fence_end]
		text = text.strip()

	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		text = text[start:end + 1]

	return text


def _parse_json_response(response_text: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
	try:
		response = json.loads(_strip_json_wrapper(response_text))
	except json.JSONDecodeError:
		return None, "response was not valid JSON"
	if not isinstance(response, dict):
		return None, "response JSON was not an object"
	return response, None


def _split_combined_action(action_raw: object) -> Tuple[Optional[str], Optional[str]]:
	"""
	Local instruct models (e.g. phi3:mini) routinely fold the symbol into the
	action field -- {"action": "BUY NVDA", "symbol": "NVDA", ...} -- even
	though the rest of the JSON is well-formed. That's a small model's
	formatting quirk, not a hallucinated/illegal trade -- rejecting the whole
	response over it would just be parser pickiness inflating the fallback
	rate without reflecting anything about actual decision quality. Split the
	leading action word off and hand back whatever trailed it as a fallback
	symbol hint, in case the "symbol" key itself is missing/null.
	"""
	if not isinstance(action_raw, str):
		return None, None
	parts = action_raw.strip().upper().split()
	if not parts:
		return None, None
	return parts[0], (parts[1] if len(parts) > 1 else None)


# Validation failures that mean "the model asked for a trade it was not allowed
# to make" -- i.e. an action hallucination. Everything else validate_response
# can return is a small-model *formatting* failure (bad JSON, a float where an
# int was wanted) and must NOT be counted as a hallucination, or the headline
# metric just measures how badly phi3 formats JSON.
ILLEGAL_ACTION_ERRORS = frozenset({
	"action type is invalid",
	"HOLD was not in legal list",
	"action/symbol not in legal list",
	"quantity outside legal limit",
})


def is_illegal_action_error(error: Optional[str]) -> bool:
	"""True if `error` from validate_response() denotes an illegal trade request."""
	return error in ILLEGAL_ACTION_ERRORS


# Actions that move no shares. Anything else the model names -- including a verb
# that does not exist, like "short" or "hedge" -- is it asking to trade.
NON_TRADE_ACTIONS = frozenset({"hold", "analyze"})


def is_trade_request(requested: Optional[Dict[str, object]]) -> bool:
	"""
	True if `requested` (from describe_request) is the model asking to trade.

	This is the denominator for every hallucination rate, and it must not be
	`action in ("buy", "sell")`. describe_request() reports the verb the model
	actually emitted, and validate_response() rejects anything outside
	BUY/SELL/HOLD with "action type is invalid" -- which is_illegal_action_error
	counts as a hallucination. So an invalid verb incremented the numerator
	while a ("buy", "sell") test left the denominator untouched, and the rate
	could exceed 100%: run 05 reported Alex Chen at 48/33 = 145.5%, where all 48
	were "action type is invalid" and none of the 48 could ever be counted.

	Requesting an action that is not in the legal action space is the purest
	form of action hallucination, so it belongs in both terms, not neither.
	A null action (unparseable JSON) is a formatting failure, not a trade
	request, and stays out of both -- see ILLEGAL_ACTION_ERRORS.
	"""
	action = (requested or {}).get("action")
	if not isinstance(action, str):
		return False
	return action.strip().lower() not in NON_TRADE_ACTIONS


def describe_request(response_text: str) -> Dict[str, object]:
	"""
	Best-effort extraction of what the model *asked for*, independent of whether
	the request validated. This is the raw pre-filter request, which is the only
	place an action hallucination is observable in the middleware arm -- once
	run_action_filtering_step() has done its job the illegal action is gone.
	"""
	response, _error = _parse_json_response(response_text)
	if response is None:
		return {"action": None, "symbol": None, "quantity": None}
	action_type, symbol_hint = _split_combined_action(response.get("action"))
	symbol = response.get("symbol") or response.get("ticker") or symbol_hint
	if isinstance(symbol, str):
		symbol = symbol.strip().upper()
		if symbol in ("NULL", "NONE", ""):
			symbol = None
	return {
		"action": action_type.lower() if action_type else None,
		"symbol": symbol,
		"quantity": response.get("quantity"),
	}


def coerce_quantity(raw: object) -> Optional[int]:
	"""
	Normalise a model-supplied quantity to an int, or None if it isn't numeric.

	Small instruct models emit 10, "10" and 10.0 interchangeably. The baseline
	arm already coerced all three via int(); this arm used to demand a real int
	and reject the rest, which burned a retry and usually ended as a fallback
	HOLD. That made part of the filtered arm's do-nothing rate an artifact of
	parser strictness rather than a difference in decision quality -- the exact
	thing _split_combined_action's docstring argues against. Both arms now share
	this one definition.

	bool is rejected explicitly: isinstance(True, int) is True in Python, so
	{"quantity": true} would otherwise sail through as a 1-share order.
	"""
	if isinstance(raw, bool):
		return None
	if isinstance(raw, int):
		return raw
	if isinstance(raw, float):
		return int(raw)
	if isinstance(raw, str):
		try:
			return int(float(raw.strip().replace(",", "")))
		except (TypeError, ValueError):
			return None
	return None


def validate_response(
	response_text: str, legal_actions: List[Action]
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
	response, error = _parse_json_response(response_text)
	if error:
		return None, error

	action_raw = response.get("action")
	action_type, action_symbol_hint = _split_combined_action(action_raw)
	symbol = response.get("symbol")
	if symbol is None:
		symbol = response.get("ticker")
	if symbol is None:
		symbol = action_symbol_hint
	if isinstance(symbol, str):
		symbol = symbol.strip().upper()
		# phi3:mini frequently emits the *string* "null"/"none" rather than a
		# JSON null for "no symbol". Left as a string it stays truthy, and
		# execute_trading_action() then treats a plain HOLD as a trade in an
		# unknown ticker and flags it as a hallucination -- inflating the
		# headline hallucination rate with decisions that were entirely valid.
		if symbol in ("NULL", "NONE", ""):
			symbol = None
	quantity = response.get("quantity")

	# ANALYZE is accepted at the parser and normalised to HOLD rather than
	# rejected. It is not offered in the prompt (see get_legal_actions), but a
	# model that emits it anyway is expressing "no trade", not an illegal trade
	# -- rejecting it would burn a retry and inflate the fallback rate with a
	# decision that was semantically fine.
	if action_type == "ANALYZE":
		action_type = "HOLD"

	if action_type not in {"BUY", "SELL", "HOLD"}:
		return None, "action type is invalid"

	legal_map = {(a.type, a.symbol): a.quantity for a in legal_actions}

	if action_type == "HOLD":
		if ("HOLD", None) not in legal_map:
			return None, "HOLD was not in legal list"
		# A no-op action carrying a quantity is a formatting slip, not an
		# illegal trade -- neither HOLD nor ANALYZE moves any shares, so
		# normalise the quantity to 0 rather than rejecting the whole response
		# and burning a retry (which previously ended as a fallback HOLD and
		# inflated this arm's do-nothing rate).
		# Normalise exactly like the buy/sell path below does. Without this the
		# action stays whatever case the model produced ("HOLD"), and
		# execute_trading_action()'s `action in ("hold", "analyze")` check
		# misses it, sending a plain hold down the trade-execution path.
		response["action"] = action_type.lower()
		response["symbol"] = symbol
		response["quantity"] = 0
		if "reasoning" not in response:
			response["reasoning"] = ""
		return response, None

	if (action_type, symbol) not in legal_map:
		return None, "action/symbol not in legal list"

	quantity = coerce_quantity(quantity)
	if quantity is None:
		return None, "quantity must be an integer"

	max_qty = legal_map[(action_type, symbol)]
	if quantity < 1 or quantity > max_qty:
		return None, "quantity outside legal limit"

	response["action"] = action_type.lower()
	response["symbol"] = symbol
	response["quantity"] = quantity
	if "reasoning" not in response:
		response["reasoning"] = ""
	return response, None


LOG_COLUMNS = [
	"timestamp", "agent", "action", "symbol", "quantity", "status",
	"error_reason",
	# What the model actually asked for before validation. The first four
	# columns hold the FINAL action, so a request that was rejected and retried
	# into a HOLD logged as `hold,,0` -- all 44 rejections in run 03 were
	# indistinguishable from a genuine hold, and the CSV could not be audited
	# without cross-referencing trading_log.json.
	"requested_action", "requested_symbol", "requested_quantity",
]


def _ensure_log_header(log_path: str) -> None:
	if os.path.exists(log_path):
		return
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	with open(log_path, "w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(LOG_COLUMNS)


def log_action(
	log_path: str,
	agent_name: str,
	action: Dict[str, object],
	status: str,
	error_reason: str = "",
	requested: Optional[Dict[str, object]] = None,
) -> None:
	_ensure_log_header(log_path)
	# utcnow() returns a naive datetime and is deprecated in 3.12+; the log is
	# compared against market timestamps, so the offset has to be explicit.
	timestamp = datetime.now(timezone.utc).isoformat()
	action_type = action.get("action")
	symbol = action.get("symbol")
	quantity = action.get("quantity")
	req = requested or {}
	with open(log_path, "a", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow([
			timestamp, agent_name, action_type, symbol, quantity, status,
			error_reason,
			req.get("action"), req.get("symbol"), req.get("quantity"),
		])


def run_action_filtering_step(
	agent_state,
	market_state,
	memory_context: str,
	llm_call: Callable[[str], str],
	log_path: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
	"""
	Returns (decision, stats).

	`stats` exists because this function is the *only* place the middleware
	arm's action hallucinations are observable. By the time the decision reaches
	execute_trading_action() every illegal request has already been rejected and
	replaced with a legal one, so a detector sitting downstream of this function
	can only ever report zero. stats keys:

	  status          "success" | "retry" | "fallback"
	  illegal_request True if the model's FIRST attempt asked for an illegal
	                  trade (as opposed to merely emitting malformed JSON)
	  error_reason    the first attempt's validation error, if any
	  requested       what the first attempt actually asked for
	"""
	prompt, legal_actions = build_prompt(agent_state, market_state, memory_context)

	response_text = llm_call(prompt)
	response, error = validate_response(response_text, legal_actions)

	# Capture the pre-filter request before any correction happens.
	stats: Dict[str, object] = {
		"status": "success",
		"illegal_request": is_illegal_action_error(error),
		"error_reason": error or "",
		"requested": describe_request(response_text),
	}

	requested = stats["requested"]

	if response is not None:
		log_action(log_path, _get_agent_name(agent_state), response, "success",
		           requested=requested)
		return response, stats

	retry_prompt = prompt + f"\n\nValidation error: {error}. Try again."
	retry_text = llm_call(retry_prompt)
	retry_response, retry_error = validate_response(retry_text, legal_actions)

	# When the FIRST attempt was unparseable, describe_request() returned
	# {"action": None, ...}, but illegal_request below is still allowed to fire
	# on the retry. _generate_report() takes its numerator from illegal_request
	# and its denominator from stats["requested"], so an unparseable-then-illegal
	# step incremented the numerator and never the denominator. Run 04 reported
	# Alex Chen at 48 hallucinations against 33 attempts (145.5%) -- his
	# trade_attempts exactly equalled his executed buy+sell, i.e. not one of the
	# 48 was counted. Adopt the retry's request when the first yielded none.
	if not requested.get("action"):
		retry_requested = describe_request(retry_text)
		if retry_requested.get("action"):
			stats["requested"] = requested = retry_requested

	if retry_response is not None:
		log_action(log_path, _get_agent_name(agent_state), retry_response, "retry",
		           error, requested=requested)
		stats["status"] = "retry"
		return retry_response, stats

	fallback = {"action": "hold", "symbol": None, "quantity": 0, "reasoning": ""}
	log_action(
		log_path,
		_get_agent_name(agent_state),
		fallback,
		"fallback",
		retry_error or "validation failed twice",
		requested=requested,
	)
	stats["status"] = "fallback"
	# A second illegal attempt still counts, even if the first was just malformed.
	stats["illegal_request"] = bool(stats["illegal_request"]) or is_illegal_action_error(retry_error)
	return fallback, stats
