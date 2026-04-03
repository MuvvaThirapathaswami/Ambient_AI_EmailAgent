"""
evaluation.py - Evaluation framework for the email agent (Milestone 2).

Provides:
  1. A golden test dataset (25 seed emails + labels).
  2. Triage accuracy measurement.
  3. An LLM-as-a-judge scorer for response quality.
  4. A runner that uses LangSmith for tracing.
"""

import json
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph import graph
from state import AgentState, Email


# ══════════════════════════════════════════════════════════
# 1. Golden Dataset
# ══════════════════════════════════════════════════════════

GOLDEN_DATASET: list[dict] = [
    # Format: {email, expected_triage, expected_keywords_in_reply}
    {
        "email": {
            "id": "test_001",
            "sender": "boss@company.com",
            "subject": "Project deadline update",
            "body": "Hi, can you confirm the project will be delivered by Friday?",
            "timestamp": "2024-01-15T09:00:00",
        },
        "expected_triage": "respond",
        "reply_keywords": ["Friday", "confirm", "delivery"],
    },
    {
        "email": {
            "id": "test_002",
            "sender": "newsletter@deals.com",
            "subject": "50% OFF — Today Only!!!",
            "body": "Click here for amazing deals you can't miss!",
            "timestamp": "2024-01-15T09:05:00",
        },
        "expected_triage": "ignore",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_003",
            "sender": "hr@company.com",
            "subject": "Confidential: Performance Review",
            "body": "Please review the attached performance documentation.",
            "timestamp": "2024-01-15T09:10:00",
        },
        "expected_triage": "notify_human",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_004",
            "sender": "client@bigcorp.com",
            "subject": "Meeting request for next week",
            "body": "Could we schedule a 30-minute call on Tuesday or Wednesday?",
            "timestamp": "2024-01-15T09:15:00",
        },
        "expected_triage": "respond",
        "reply_keywords": ["Tuesday", "Wednesday", "schedule"],
    },
    {
        "email": {
            "id": "test_005",
            "sender": "noreply@github.com",
            "subject": "Your pull request was merged",
            "body": "Pull request #42 'Fix login bug' was successfully merged.",
            "timestamp": "2024-01-15T09:20:00",
        },
        "expected_triage": "ignore",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_006",
            "sender": "legal@law.com",
            "subject": "Urgent: Contract dispute — action required",
            "body": "Our client is pursuing legal action. Please contact us immediately.",
            "timestamp": "2024-01-15T09:25:00",
        },
        "expected_triage": "notify_human",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_007",
            "sender": "alice@partner.com",
            "subject": "Quick question about the API",
            "body": "Hi, what is the base URL for your REST API? Thanks!",
            "timestamp": "2024-01-15T09:30:00",
        },
        "expected_triage": "respond",
        "reply_keywords": ["API", "URL"],
    },
    {
        "email": {
            "id": "test_008",
            "sender": "security@bank.com",
            "subject": "Your account has been suspended",
            "body": "Click here to verify your identity or your account will be closed.",
            "timestamp": "2024-01-15T09:35:00",
        },
        "expected_triage": "ignore",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_009",
            "sender": "ceo@company.com",
            "subject": "All-hands meeting tomorrow",
            "body": "Please join the all-hands meeting at 10am tomorrow. Very important.",
            "timestamp": "2024-01-15T09:40:00",
        },
        "expected_triage": "notify_human",
        "reply_keywords": [],
    },
    {
        "email": {
            "id": "test_010",
            "sender": "vendor@supplier.com",
            "subject": "Invoice #1234 — Payment Due",
            "body": "Please find attached invoice #1234 for $5,000 due in 30 days.",
            "timestamp": "2024-01-15T09:45:00",
        },
        "expected_triage": "notify_human",
        "reply_keywords": [],
    },
]

# Add 15 more synthetic entries to reach 25+
for i in range(11, 26):
    GOLDEN_DATASET.append({
        "email": {
            "id": f"test_{i:03d}",
            "sender": f"user{i}@example.com",
            "subject": f"Test email {i}",
            "body": f"This is synthetic test email number {i}.",
            "timestamp": "2024-01-15T10:00:00",
        },
        "expected_triage": "ignore" if i % 3 == 0 else "respond",
        "reply_keywords": [] if i % 3 == 0 else ["test"],
    })


# ══════════════════════════════════════════════════════════
# 2. Triage Accuracy Evaluator
# ══════════════════════════════════════════════════════════

def run_triage_evaluation(dataset: list[dict] = GOLDEN_DATASET) -> dict:
    """
    Run each email through the triage node and measure accuracy.

    Returns:
        {total, correct, accuracy, failures}
    """
    correct = 0
    failures = []

    for item in dataset:
        initial_state: AgentState = {
            "email_input": item["email"],
            "messages": [],
            "triage_result": None,
            "draft_response": None,
            "hitl_decision": None,
            "human_edit": None,
            "memory_context": None,
            "final_response": None,
        }

        config = {"configurable": {"thread_id": item["email"]["id"]}}

        # Run only up to (and including) the triage node
        from nodes import load_memory, triage as triage_node
        state = load_memory(initial_state)
        initial_state.update(state)
        state = triage_node(initial_state)
        initial_state.update(state)

        predicted = initial_state.get("triage_result")
        expected = item["expected_triage"]

        if predicted == expected:
            correct += 1
        else:
            failures.append({
                "id": item["email"]["id"],
                "subject": item["email"]["subject"],
                "expected": expected,
                "got": predicted,
            })

    total = len(dataset)
    accuracy = correct / total if total else 0

    print(f"\n{'═'*50}")
    print(f"TRIAGE EVALUATION RESULTS")
    print(f"{'═'*50}")
    print(f"Total emails  : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.1%}")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  [{f['id']}] '{f['subject']}' → expected {f['expected']}, got {f['got']}")
    else:
        print("\n✓ No failures!")
    print(f"{'═'*50}\n")

    return {"total": total, "correct": correct, "accuracy": accuracy, "failures": failures}


# ══════════════════════════════════════════════════════════
# 3. LLM-as-a-Judge Quality Scorer
# ══════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator of AI-generated email responses.

Score the response on each criterion from 1 (poor) to 5 (excellent):
  - helpfulness  : Does it address the sender's request?
  - tone         : Is it professional and appropriately formal?
  - conciseness  : Is it concise without omitting key info?
  - accuracy     : Are all facts / dates / names correct?

Reply ONLY with JSON, no other text:
{
  "helpfulness": <1-5>,
  "tone": <1-5>,
  "conciseness": <1-5>,
  "accuracy": <1-5>,
  "overall": <average, 1 decimal place>,
  "comments": "<one sentence>"
}
"""

def llm_judge(email: dict, response: str) -> dict:
    """
    Use Gemini to score an agent-generated email reply.

    Returns a dict with per-criterion scores + an overall average.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    user_prompt = f"""
Original Email:
From:    {email['sender']}
Subject: {email['subject']}
Body:    {email['body']}

---
Agent's Reply:
{response}
"""

    resp = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    try:
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {"error": str(e), "raw": resp.content}


def run_quality_evaluation(test_cases: list[dict]) -> list[dict]:
    """
    Run the LLM-as-a-judge over a list of {email, response} dicts.

    Args:
        test_cases: [{"email": Email, "response": str}, ...]

    Returns:
        List of score dicts, one per test case.
    """
    results = []
    for i, tc in enumerate(test_cases, 1):
        score = llm_judge(tc["email"], tc["response"])
        score["email_id"] = tc["email"]["id"]
        results.append(score)
        print(f"[{i}/{len(test_cases)}] {tc['email']['id']} → overall: {score.get('overall', 'ERR')}")

    # Summary
    valids = [r for r in results if "overall" in r]
    if valids:
        avg = sum(r["overall"] for r in valids) / len(valids)
        print(f"\nMean overall quality score: {avg:.2f} / 5.0")

    return results


# ══════════════════════════════════════════════════════════
# 4. CLI entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running triage accuracy evaluation …\n")
    run_triage_evaluation()
