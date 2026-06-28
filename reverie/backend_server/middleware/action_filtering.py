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
			lines.append("- HOLD")
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
{{"action": "BUY/SELL/HOLD", "symbol": "TICKER or null", "quantity": number, "reasoning": "why"}}
"""
	return prompt, legal_actions


def _parse_json_response(response_text: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
	try:
		response = json.loads(response_text)
	except json.JSONDecodeError:
		return None, "response was not valid JSON"
	if not isinstance(response, dict):
		return None, "response JSON was not an object"
	return response, None


def validate_response(
	response_text: str, legal_actions: List[Action]
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
	response, error = _parse_json_response(response_text)
	if error:
		return None, error

	action_raw = response.get("action")
	action_type = action_raw.strip().upper() if isinstance(action_raw, str) else None
	symbol = response.get("symbol")
	if symbol is None:
		symbol = response.get("ticker")
	if isinstance(symbol, str):
		symbol = symbol.strip().upper()
	quantity = response.get("quantity")

	if action_type not in {"BUY", "SELL", "HOLD"}:
		return None, "action type is invalid"

	legal_map = {(a.type, a.symbol): a.quantity for a in legal_actions}

	if action_type == "HOLD":
		if ("HOLD", None) not in legal_map:
			return None, "HOLD was not in legal list"
		if quantity not in (0, None):
			return None, "HOLD quantity must be 0"
		return response, None

	if (action_type, symbol) not in legal_map:
		return None, "action/symbol not in legal list"

	if not isinstance(quantity, int):
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
) -> Dict[str, object]:
	prompt, legal_actions = build_prompt(agent_state, market_state, memory_context)

	response_text = llm_call(prompt)
	response, error = validate_response(response_text, legal_actions)
	if response is not None:
		log_action(log_path, _get_agent_name(agent_state), response, "success")
		return response

	retry_prompt = prompt + f"\n\nValidation error: {error}. Try again."
	retry_text = llm_call(retry_prompt)
	retry_response, retry_error = validate_response(retry_text, legal_actions)
	if retry_response is not None:
		log_action(log_path, _get_agent_name(agent_state), retry_response, "retry", error)
		return retry_response

	fallback = {"action": "hold", "symbol": None, "quantity": 0, "reasoning": ""}
	log_action(
		log_path,
		_get_agent_name(agent_state),
		fallback,
		"fallback",
		retry_error or "validation failed twice",
	)
	return fallback
