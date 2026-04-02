from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class MiddlewareConfig:
    # -- chaos injector --
    # --metrics --
    # --validator --

    # enable_chaos: bool = False
    # chaos_schedule: str = "every_N_steps"
    # chaos_injection_probability: float = 0.3
    # chaos_step_interval: int = 10
    # random_seed: int = 42
    
    # # ============ STABILIZER SETTINGS ============
    enable_stabilizer: bool = True
    entropy_discard_patterns: Optional[List[str]] = None
    persona_anchor_template: str = (...)
    # redundancy_window: int = 5
    # redundancy_threshold: float = 0.85
    
    # # ============ VALIDATOR SETTINGS ============
    # enable_validator: bool = True
    # max_validator_retries: int = 1
    # retry_with_llm_feedback: bool = True
    # fallback_action: str = "stay_in_place"
    
    # # ============ METRICS SETTINGS ============
    # enable_metrics: bool = True
    # baseline_steps: int = 10
    # metrics_csv_path: str = "metrics/metrics.csv"

    # enable_stabilizer: bool=True

    # persona_anchor_template: str = (
    #     "You are {name}. Key traits: {traits}. Skills: {skills}. "
    #     "Lifestyle: {lifestyle}. Act consistently with these. Never deviate."
    # )
    # def __post_init__(self):
    #     """
    #     Called automatically after dataclass creation.
    #     Initialize list defaults.
    #     """
    #     if self.entropy_discard_patterns is None:
    #         self.entropy_discard_patterns = ["idle", "greetings", "sleep", "waiting"]

def load_middleware_config(config_path: str) -> MiddlewareConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to JSON config file
        
    Returns:
        MiddlewareConfig: Configuration object
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return MiddlewareConfig(**config_dict)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        print("Using default configuration.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        print("Using default configuration.")
        return get_default_config()


def get_default_config() -> MiddlewareConfig:
    """
    Get default configuration with all standard values.
    
    Returns:
        MiddlewareConfig: Fresh config with defaults
    """
    return MiddlewareConfig()

