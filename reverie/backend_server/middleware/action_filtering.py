from dataclasses import dataclass
from datetime import datetime, time
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
	if not hasattr(agent_state, "scratch"):
		return 0.0
	portfolio_value = agent_state.portfolio_value(market_state.current_prices)
	return portfolio_value * float(getattr(agent_state.scratch, "risk_limit_per_trade", 0.0))


def get_legal_actions(agent_state, market_state) -> List[Action]:
	legal: List[Action] = []

	# HOLD is always legal
	legal.append(Action("HOLD", None, 0, "always available"))

	# ANALYZE is always legal -- a no-op "gather information, trade nothing"
	# action. The baseline arm's free-form prompt has always offered it, so
	# omitting it here gave the two arms different action vocabularies and made
	# their action distributions incomparable (baseline analyze=200 vs
	# middleware hold=200 was partly a prompt difference, not a behaviour one).
	# Legal even when the market is closed -- reading the tape needs no venue.
	legal.append(Action("ANALYZE", None, 0, "always available"))

	# Market closed -> only the no-op actions
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
			lines.append("- HOLD")
		elif action.type == "ANALYZE":
			lines.append("- ANALYZE (study the market, place no trade)")
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
	max_trade_size = _get_max_trade_size(agent_state, market_state)

	prompt = f"""
You are {name}, a {persona} trader.
Traits: {innate}
Background: {learned}
Current situation: {currently}
Risk tolerance: {risk_tolerance}
Max single-trade size: ${max_trade_size:,.2f}

Current state:
- Cash: ${cash_balance:,.2f}
- Holdings: {positions}

Market:
- Status: {market_open}
- Prices: {prices}

Memory:
{memory_context}

You MUST choose from ONLY these actions:
{action_str}

Respond ONLY in this JSON format:
{{"action": "BUY/SELL/HOLD/ANALYZE", "symbol": "TICKER or null", "quantity": number, "reasoning": "why"}}
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

	if action_type not in {"BUY", "SELL", "HOLD", "ANALYZE"}:
		return None, "action type is invalid"

	legal_map = {(a.type, a.symbol): a.quantity for a in legal_actions}

	if action_type in ("HOLD", "ANALYZE"):
		if (action_type, None) not in legal_map:
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


def _ensure_log_header(log_path: str) -> None:
	if os.path.exists(log_path):
		return
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	with open(log_path, "w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			["timestamp", "agent", "action", "symbol", "quantity", "status", "error_reason"]
		)


def log_action(
	log_path: str,
	agent_name: str,
	action: Dict[str, object],
	status: str,
	error_reason: str = "",
) -> None:
	_ensure_log_header(log_path)
	timestamp = datetime.utcnow().isoformat()
	action_type = action.get("action")
	symbol = action.get("symbol")
	quantity = action.get("quantity")
	with open(log_path, "a", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[timestamp, agent_name, action_type, symbol, quantity, status, error_reason]
		)


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

	if response is not None:
		log_action(log_path, _get_agent_name(agent_state), response, "success")
		return response, stats

	retry_prompt = prompt + f"\n\nValidation error: {error}. Try again."
	retry_text = llm_call(retry_prompt)
	retry_response, retry_error = validate_response(retry_text, legal_actions)
	if retry_response is not None:
		log_action(log_path, _get_agent_name(agent_state), retry_response, "retry", error)
		stats["status"] = "retry"
		return retry_response, stats

	fallback = {"action": "hold", "symbol": None, "quantity": 0, "reasoning": ""}
	log_action(
		log_path,
		_get_agent_name(agent_state),
		fallback,
		"fallback",
		retry_error or "validation failed twice",
	)
	stats["status"] = "fallback"
	# A second illegal attempt still counts, even if the first was just malformed.
	stats["illegal_request"] = bool(stats["illegal_request"]) or is_illegal_action_error(retry_error)
	return fallback, stats
