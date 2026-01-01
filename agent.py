"""Compatibility shim.

`poolenv.py` imports `Agent`, `BasicAgent`, and `NewAgent` from a top-level module
named `agent`. The actual implementations live under the `agents/` package.

This file preserves the original import path without modifying `poolenv.py`.
"""

from agents.agent import Agent
from agents.basic_agent import BasicAgent
from agents.new_agent import NewAgent
