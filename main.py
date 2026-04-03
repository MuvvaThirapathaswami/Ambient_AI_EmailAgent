"""
main.py - CLI entry point for the Ambient Email Agent.

Run:
    python main.py                  # process a hard-coded demo email
    python main.py --evaluate       # run the triage evaluation suite
    python main.py --live           # fetch real emails from Gmail (Milestone 4)
"""

import argparse
import uuid
from dotenv import load_dotenv

load_dotenv()

from graph import graph
from state import AgentState


# ──────────────────────────────────────────────
# Demo emails for quick testing
# ──────────────────────────────────────────────

DEMO_EMAILS = [
    {
        "id": str(uuid.uuid4()),
        "sender": "client@bigcorp.com",
        "subject": "Meeting request for next week",
        "body": (
            "Hi,\n\n"
            "I hope you are doing well! Could we schedule a 30-minute call "
            "on Tuesday or Wednesday afternoon to discuss the new proposal?\n\n"
            "Best regards,\nJohn"
        ),
        "timestamp": "2024-01-15T09:00:00",
    },
    {
        "id": str(uuid.uuid4()),
        "sender": "newsletter@spam.com",
        "subject": "🎉 You've been selected! Claim your prize NOW",
        "body": "Click here to claim your $1,000 gift card. Limited time only!",
        "timestamp": "2024-01-15T09:05:00",
    },
]


def run_single_email(email: dict) -> None:
    """Process a single email through the full agent graph."""
    print(f"\n{'━'*60}")
    print(f"Processing email: {email['subject']}")
    print(f"From: {email['sender']}")
    print(f"{'━'*60}")

    initial_state: AgentState = {
        "email_input": email,
        "messages": [],
        "triage_result": None,
        "draft_response": None,
        "hitl_decision": None,
        "human_edit": None,
        "memory_context": None,
        "final_response": None,
    }

    config = {"configurable": {"thread_id": email["id"]}}

    final_state = graph.invoke(initial_state, config=config)

    print(f"\n{'━'*60}")
    print("RESULT")
    print(f"{'━'*60}")
    print(f"Triage      : {final_state.get('triage_result')}")
    print(f"HITL        : {final_state.get('hitl_decision')}")
    print(f"Final reply : {final_state.get('final_response', 'N/A')}")
    print(f"{'━'*60}\n")


def run_live_gmail() -> None:
    """Fetch unread emails from Gmail and process each one (Milestone 4)."""
    try:
        from gmail_service import fetch_unread_emails, mark_as_read
    except ImportError:
        print("Gmail service not configured. Please complete Milestone 4 setup.")
        return

    print("Fetching unread emails from Gmail …")
    emails = fetch_unread_emails(max_results=5)

    if not emails:
        print("Inbox is empty!")
        return

    for email in emails:
        run_single_email(email)
        choice = input("\nMark as read? [y/N]: ").strip().lower()
        if choice == "y":
            mark_as_read(email["id"])
            print(f"Email {email['id']} marked as read.")


def main():
    parser = argparse.ArgumentParser(description="Ambient Email Agent")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--live", action="store_true", help="Fetch live Gmail emails")
    parser.add_argument("--demo", type=int, default=0, help="Demo email index (0 or 1)")
    args = parser.parse_args()

    if args.evaluate:
        from evaluation import run_triage_evaluation
        run_triage_evaluation()

    elif args.live:
        run_live_gmail()

    else:
        email = DEMO_EMAILS[args.demo % len(DEMO_EMAILS)]
        run_single_email(email)


if __name__ == "__main__":
    main()
