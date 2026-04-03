"""
tools.py - All tools available to the ReAct agent.

Tools are split into two categories:
  SAFE_TOOLS     — run automatically, no human approval needed.
  DANGEROUS_TOOLS — trigger the HITL checkpoint before execution.

In Milestone 4 the mock implementations are replaced by real Gmail /
Google Calendar API calls (see gmail_service.py).
"""

from datetime import datetime, timedelta
from langchain_core.tools import tool


# ══════════════════════════════════════════════════════════
# SAFE TOOLS  (read-only, no side effects)
# ══════════════════════════════════════════════════════════

@tool
def read_calendar(date: str = "") -> str:
    """
    Return calendar availability for a given date (YYYY-MM-DD).
    Defaults to today if no date is supplied.

    This is a SAFE tool — it runs without human approval.
    """
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    # ── Mock implementation ──────────────────────────────
    # Replace with real Google Calendar API call in Milestone 4.
    mock_slots = [
        f"{date} 09:00–10:00  Free",
        f"{date} 11:00–12:00  Busy (Team Standup)",
        f"{date} 14:00–15:00  Free",
        f"{date} 16:00–17:00  Free",
    ]
    return "Calendar for " + date + ":\n" + "\n".join(mock_slots)


@tool
def get_email_thread(thread_id: str) -> str:
    """
    Retrieve the previous messages in an email thread for context.

    This is a SAFE tool — it runs without human approval.
    """
    # ── Mock implementation ──────────────────────────────
    return (
        f"[Thread {thread_id}]\n"
        "Previous message (3 days ago): 'Hi, just checking in on the proposal.'\n"
        "Previous message (1 week ago): 'Hi, I sent over the proposal last week. "
        "Please let me know your thoughts.'"
    )


@tool
def search_contacts(name: str) -> str:
    """
    Look up a contact's email address by name.

    This is a SAFE tool — it runs without human approval.
    """
    # ── Mock contact book ────────────────────────────────
    contacts = {
        "alice": "alice@example.com",
        "bob": "bob@example.com",
        "robert": "robert@example.com",
        "carol": "carol@example.com",
    }
    email = contacts.get(name.lower(), f"No contact found for '{name}'")
    return f"Contact lookup → {name}: {email}"


# ══════════════════════════════════════════════════════════
# DANGEROUS TOOLS  (write / send — require human approval)
# ══════════════════════════════════════════════════════════

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to the specified recipient.

    ⚠️  DANGEROUS — this tool PAUSES execution and waits for
    human approval before running.

    In Milestone 4 replace the mock below with a real Gmail API send call.
    """
    # ── Mock implementation ──────────────────────────────
    print(f"\n[MOCK send_email] To: {to} | Subject: {subject}")
    print(f"Body:\n{body}\n")
    return f"Email successfully sent to {to} with subject '{subject}'."


@tool
def create_calendar_invite(
    title: str,
    date: str,
    start_time: str,
    end_time: str,
    attendees: str,
) -> str:
    """
    Create a Google Calendar event and invite attendees.

    ⚠️  DANGEROUS — this tool PAUSES execution and waits for
    human approval before running.

    Args:
        title      : Event title / subject.
        date       : Date in YYYY-MM-DD format.
        start_time : Start time in HH:MM (24-h) format.
        end_time   : End time in HH:MM (24-h) format.
        attendees  : Comma-separated list of attendee email addresses.
    """
    # ── Mock implementation ──────────────────────────────
    print(
        f"\n[MOCK create_calendar_invite] '{title}' on {date} "
        f"{start_time}–{end_time} with {attendees}"
    )
    return (
        f"Calendar invite '{title}' created for {date} "
        f"{start_time}–{end_time}. Attendees notified: {attendees}."
    )


@tool
def archive_email(email_id: str) -> str:
    """
    Archive (label as read + move out of inbox) the specified email.

    ⚠️  DANGEROUS — requires human approval.
    """
    print(f"\n[MOCK archive_email] Archiving email id={email_id}")
    return f"Email {email_id} archived successfully."


# ══════════════════════════════════════════════════════════
# Grouped lists — used by the agent graph for routing logic
# ══════════════════════════════════════════════════════════

SAFE_TOOLS = [read_calendar, get_email_thread, search_contacts]
DANGEROUS_TOOLS = [send_email, create_calendar_invite, archive_email]
ALL_TOOLS = SAFE_TOOLS + DANGEROUS_TOOLS

# Name sets for quick membership checks
SAFE_TOOL_NAMES: set[str] = {t.name for t in SAFE_TOOLS}
DANGEROUS_TOOL_NAMES: set[str] = {t.name for t in DANGEROUS_TOOLS}
