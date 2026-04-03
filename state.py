"""
state.py - Defines the AgentState for the LangGraph email assistant.
All nodes in the graph read from and write to this shared state object.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Email(TypedDict):
    """Represents an incoming email."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class AgentState(TypedDict):
    """
    The central state object passed between every node in the LangGraph graph.

    Fields:
        email_input      : The raw incoming email being processed.
        messages         : Full conversation/tool-call history (append-only via add_messages).
        triage_result    : Classification decision: 'ignore' | 'notify_human' | 'respond'.
        draft_response   : The agent's latest drafted email reply.
        hitl_decision    : Human review outcome: 'approve' | 'deny' | 'edit'.
        human_edit       : The corrected text provided by the human during an 'edit' decision.
        memory_context   : Serialized user preferences loaded from the persistent memory store.
        final_response   : The approved final email that will be sent.
    """
    email_input: Email
    messages: Annotated[list[BaseMessage], add_messages]
    triage_result: Optional[Literal["ignore", "notify_human", "respond"]]
    draft_response: Optional[str]
    hitl_decision: Optional[Literal["approve", "deny", "edit"]]
    human_edit: Optional[str]
    memory_context: Optional[str]
    final_response: Optional[str]
