"""
gmail_service.py - Real Gmail API integration (Milestone 4).

Replaces mock tool implementations with live Gmail & Google Calendar calls.
Requires:
  - credentials.json  (OAuth 2.0 client secrets downloaded from Google Cloud Console)
  - Gmail API & Google Calendar API enabled in the project

Usage:
    from gmail_service import get_gmail_service, fetch_unread_emails, send_gmail

Scopes needed (add to OAuth consent screen):
    https://www.googleapis.com/auth/gmail.modify
    https://www.googleapis.com/auth/calendar
"""

import os
import base64
import pickle
from pathlib import Path
from email.mime.text import MIMEText
from datetime import datetime, timedelta

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]

CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "token.json")


# ──────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────

def _get_credentials() -> Credentials:
    """Load or refresh OAuth 2.0 credentials, running the browser flow if needed."""
    creds = None

    if Path(TOKEN_PATH).exists():
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    return creds


def get_gmail_service():
    """Return an authenticated Gmail API service object."""
    return build("gmail", "v1", credentials=_get_credentials())


def get_calendar_service():
    """Return an authenticated Google Calendar API service object."""
    return build("calendar", "v3", credentials=_get_credentials())


# ──────────────────────────────────────────────
# Gmail helpers
# ──────────────────────────────────────────────

def fetch_unread_emails(max_results: int = 10) -> list[dict]:
    """
    Fetch unread emails from the inbox.

    Returns a list of dicts with keys: id, sender, subject, body, timestamp.
    """
    service = get_gmail_service()
    result = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results)
        .execute()
    )

    emails = []
    for msg_ref in result.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_ref["id"], format="full")
            .execute()
        )

        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        body = _extract_body(msg["payload"])

        emails.append(
            {
                "id": msg_ref["id"],
                "sender": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "body": body,
                "timestamp": headers.get("Date", ""),
            }
        )

    return emails


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            text = _extract_body(part)
            if text:
                return text
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    return ""


def send_gmail(to: str, subject: str, body: str) -> str:
    """Send an email via the Gmail API."""
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
    )
    return f"Email sent. Message ID: {sent['id']}"


def mark_as_read(email_id: str) -> None:
    """Remove the UNREAD label from an email."""
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me",
        id=email_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()


# ──────────────────────────────────────────────
# Google Calendar helpers
# ──────────────────────────────────────────────

def get_calendar_availability(date: str) -> str:
    """
    Return free/busy information for the given date (YYYY-MM-DD).
    Checks the primary calendar.
    """
    service = get_calendar_service()
    start = datetime.fromisoformat(date)
    end = start + timedelta(days=1)

    body = {
        "timeMin": start.isoformat() + "Z",
        "timeMax": end.isoformat() + "Z",
        "items": [{"id": "primary"}],
    }
    result = service.freebusy().query(body=body).execute()
    busy_slots = result["calendars"]["primary"]["busy"]

    if not busy_slots:
        return f"No events on {date}. You are completely free!"

    lines = [f"Busy slots on {date}:"]
    for slot in busy_slots:
        lines.append(f"  {slot['start']} → {slot['end']}")
    return "\n".join(lines)


def create_calendar_event(
    title: str,
    date: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
) -> str:
    """Create a Google Calendar event."""
    service = get_calendar_service()
    event = {
        "summary": title,
        "start": {"dateTime": f"{date}T{start_time}:00", "timeZone": "UTC"},
        "end": {"dateTime": f"{date}T{end_time}:00", "timeZone": "UTC"},
        "attendees": [{"email": a} for a in attendees],
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Event created: {created.get('htmlLink')}"
