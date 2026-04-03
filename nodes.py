"""
nodes.py - Every node function that runs inside the LangGraph graph.
"""

import json
import os
import time
from functools import wraps
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from state import AgentState
from memory import (
    load_all_preferences,
    extract_and_save_preferences_from_edit,
    log_interaction,
)
from tools import ALL_TOOLS, DANGEROUS_TOOL_NAMES


# ──────────────────────────────────────────────────────────
# RETRY DECORATOR FOR QUOTA ISSUES
# ──────────────────────────────────────────────────────────

def retry_on_quota(max_retries=3, initial_delay=45):
    """Decorator to automatically retry when hitting rate limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Quota exceeded. Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                            time.sleep(delay)
                            delay *= 2
                        else:
                            print(f"❌ Max retries reached. Error: {error_str}")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────
# LLM INITIALIZATION
# ──────────────────────────────────────────────────────────

def _build_llm(*, tools: bool = False):
    """Return Gemini LLM with API key"""
    
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file")

    # Try models in order of preference
    model_names = [
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ]
    
    llm = None
    
    for model_name in model_names:
        try:
            print(f"🔄 Attempting to use model: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key,
                max_retries=2,
                request_timeout=60
            )
            print(f"✅ Successfully connected using: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ Failed with {model_name}: {str(e)[:100]}")
            continue
    
    if llm is None:
        raise ValueError("Could not initialize any Gemini model")

    if tools:
        return llm.bind_tools(ALL_TOOLS)

    return llm


# ══════════════════════════════════════════════════════════
# NODE 1 — load_memory
# ══════════════════════════════════════════════════════════

def load_memory(state: AgentState) -> dict:
    memory_text = load_all_preferences()
    return {"memory_context": memory_text}


# ══════════════════════════════════════════════════════════
# NODE 2 — triage
# ══════════════════════════════════════════════════════════

TRIAGE_SYSTEM_PROMPT = """
You are an intelligent email triage assistant.

Classify email into ONE:
- ignore
- notify_human
- respond

Reply ONLY JSON:
{"decision": "...", "reason": "..."}
"""

@retry_on_quota(max_retries=3, initial_delay=45)
def triage(state: AgentState) -> dict:
    email = state["email_input"]
    memory = state.get("memory_context", "")

    user_prompt = f"""
{memory}

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
"""

    llm = _build_llm()

    response = llm.invoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        decision = parsed.get("decision", "notify_human")
        reason = parsed.get("reason", "")
    except Exception as e:
        print(f"Parse error: {e}")
        decision = "notify_human"
        reason = "Parsing failed"

    print(f"[TRIAGE] {decision} - {reason}")

    return {
        "triage_result": decision,
        "messages": [AIMessage(content=f"{decision}: {reason}")]
    }


# ══════════════════════════════════════════════════════════
# NODE 3 — react_agent
# ══════════════════════════════════════════════════════════

@retry_on_quota(max_retries=3, initial_delay=45)
def react_agent(state: AgentState) -> dict:
    email = state["email_input"]
    memory = state.get("memory_context", "")

    system_prompt = f"""
You are an email assistant.

{memory}

When responding to emails:
- Be professional and helpful
- Address the sender's questions directly
- Keep responses concise but complete
"""

    messages = state.get("messages", [])

    if not any(isinstance(m, HumanMessage) for m in messages):
        messages = [
            HumanMessage(content=f"""
From: {email['sender']}
Subject: {email['subject']}
{email['body']}
""")
        ]

    llm = _build_llm(tools=True)

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)

    draft = None
    if not response.tool_calls and isinstance(response.content, str):
        draft = response.content

    return {"messages": [response], "draft_response": draft}


# ══════════════════════════════════════════════════════════
# NODE 4 — tool_executor
# ══════════════════════════════════════════════════════════

_TOOL_MAP: dict[str, Any] = {t.name: t for t in ALL_TOOLS}

def tool_executor(state: AgentState) -> dict:
    if not state.get("messages"):
        return {}
        
    last_message = state["messages"][-1]

    tool_messages = []
    draft = state.get("draft_response")

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]

            print(f"[TOOL] {name} → {args}")

            if name not in _TOOL_MAP:
                result = "Tool not found"
            else:
                try:
                    result = _TOOL_MAP[name].invoke(args)
                    if name == "send_email":
                        draft = args.get("body", draft)
                except Exception as e:
                    result = str(e)

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages, "draft_response": draft}


# ══════════════════════════════════════════════════════════
# NODE 5 — finalize
# ══════════════════════════════════════════════════════════

def finalize(state: AgentState) -> dict:
    email = state["email_input"]

    log_interaction(
        email_id=email.get("id", "unknown"),
        triage=state.get("triage_result"),
        draft=state.get("draft_response"),
        human_action=state.get("hitl_decision"),
        human_edit=state.get("human_edit"),
    )

    return {"final_response": state.get("draft_response")}


# ══════════════════════════════════════════════════════════
# NODE 6 — notify_human
# ══════════════════════════════════════════════════════════

def notify_human(state: AgentState) -> dict:
    email = state["email_input"]
    print(f"⚠️ Needs human review: {email['subject']}")
    return {"final_response": "⚠️ This email requires human review"}