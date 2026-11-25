# middleware/__init__.py

# A real import. This used to sit in a try/except ImportError that fell back to
# MiddlewareWrapper = None, because middleware_wrapper.py was an empty file.
# The effect was that `from middleware import MiddlewareWrapper` handed the
# caller None instead of raising, so a wiring mistake would surface as
# "NoneType is not callable" somewhere far away, or not at all. If the wrapper
# cannot be imported now, that is a real failure and should say so here.
from .middleware_wrapper import (
    MiddlewareWrapper,
    WrapperConfig,
    ALL_COMPONENTS,
)
from .config.middleware_config import load_middleware_config

__all__ = [
    'MiddlewareWrapper',
    'WrapperConfig',
    'ALL_COMPONENTS',
    'load_middleware_config',
]
