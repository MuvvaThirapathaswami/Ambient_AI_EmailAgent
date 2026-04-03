"""
graph.py - Assembles all nodes into a LangGraph StateGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    load_memory,
    triage,
    react_agent,
    tool_executor,
    finalize,
    notify_human,
)
from tools import DANGEROUS_TOOL_NAMES


# ──────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────

def route_after_triage(state: AgentState) -> str:
    """Route based on the triage decision."""
    decision = state.get("triage_result", "notify_human")
    if decision == "ignore":
        return "end"
    elif decision == "notify_human":
        return "notify_human"
    else:  # respond / act
        return "react_agent"


def route_after_react(state: AgentState) -> str:
    """
    After the ReAct node runs, decide what happens next:
      - If the last AI message contains tool calls → execute tools.
      - If no tool calls → the agent has finished; go to finalize.
    """
    if not state.get("messages"):
        return "finalize"
    
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])

    if not tool_calls:
        return "finalize"

    # Execute tools directly (HITL handled by UI)
    return "tool_executor"


def route_after_tools(state: AgentState) -> str:
    """After tools execute, continue the agent loop or finalize."""
    # Check if we need to continue the agent loop
    return "react_agent"


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────

def build_graph(use_checkpointer: bool = True) -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────
    builder.add_node("load_memory", load_memory)
    builder.add_node("triage", triage)
    builder.add_node("react_agent", react_agent)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("finalize", finalize)
    builder.add_node("notify_human", notify_human)

    # ── Entry point ────────────────────────────
    builder.set_entry_point("load_memory")

    # ── Edges ──────────────────────────────────
    builder.add_edge("load_memory", "triage")

    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "end": END,
            "notify_human": "notify_human",
            "react_agent": "react_agent",
        },
    )

    builder.add_conditional_edges(
        "react_agent",
        route_after_react,
        {
            "finalize": "finalize",
            "tool_executor": "tool_executor",
        },
    )

    # After tool execution, loop back to agent
    builder.add_edge("tool_executor", "react_agent")

    # Terminal nodes
    builder.add_edge("notify_human", END)
    builder.add_edge("finalize", END)

    # ── Compile ────────────────────────────────
    checkpointer = MemorySaver() if use_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


# Singleton graph instance
graph = build_graph()