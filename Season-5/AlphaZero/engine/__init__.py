"""AlphaZero Go engine — C++ with Python bindings."""

try:
    from .go_engine import *  # noqa: F401,F403
except ImportError:
    pass  # C++ extension not built yet
