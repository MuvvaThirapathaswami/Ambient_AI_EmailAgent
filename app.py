"""
app.py - Streamlit web interface for the Ambient Email Agent.
"""

import uuid
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph import graph
from state import AgentState
from memory import load_all_preferences, get_interaction_history, save_preference


st.set_page_config(
    page_title="Ambient Email Agent",
    page_icon="📧",
    layout="wide",
)


# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# Sidebar
with st.sidebar:
    st.title("🧠 Agent Memory")
    prefs = load_all_preferences()
    st.text_area("Stored Preferences", value=prefs, height=200, disabled=True)

    st.divider()
    
    st.title("📋 Interaction Log")
    log = get_interaction_history(limit=10)
    if log:
        for entry in log:
            with st.expander(f"[{entry['triage']}] {entry['email_id'][:8]}…"):
                st.json(entry)
    else:
        st.caption("No interactions yet.")

    st.divider()
    
    st.title("⚙️ Add Preference")
    pref_key = st.text_input("Key (e.g. preferred_name)")
    pref_val = st.text_input("Value (e.g. Robert, not Bob)")
    if st.button("Save Preference") and pref_key and pref_val:
        save_preference(pref_key, pref_val)
        st.success(f"Saved: {pref_key} = {pref_val}")
        st.rerun()


# Main panel
st.title("📧 Ambient Email Agent")
st.caption("An autonomous email assistant powered by LangGraph + Google Gemini")

# Warning about rate limits
st.info("💡 **Free Tier Notice:** Using gemini-2.0-flash-lite for better quota. Wait 10-15 seconds between requests.")

# Input tabs
tab_compose, tab_results = st.tabs(["📝 Compose Email", "📊 Results"])

# Compose tab
with tab_compose:
    col1, col2 = st.columns(2)
    
    with col1:
        sender = st.text_input("From Email", placeholder="sender@example.com", key="sender")
        subject = st.text_input("Subject", placeholder="Email subject...", key="subject")
        body = st.text_area("Body", height=250, placeholder="Paste email body here...", key="body")
    
    with col2:
        st.markdown("### Quick Test Emails")
        
        if st.button("📅 Meeting Request", use_container_width=True):
            st.session_state.sender = "sarah@company.com"
            st.session_state.subject = "Meeting request for next week"
            st.session_state.body = "Hi,\n\nCould we schedule a 30-minute call on Tuesday or Wednesday afternoon to discuss the proposal?\n\nBest,\nSarah"
            st.rerun()
        
        if st.button("🗑️ Spam Email", use_container_width=True):
            st.session_state.sender = "winner@prize.com"
            st.session_state.subject = "YOU WON $1000!!!"
            st.session_state.body = "Click here to claim your prize! http://fake-link.com"
            st.rerun()
        
        if st.button("🔒 HR Confidential", use_container_width=True):
            st.session_state.sender = "hr@company.com"
            st.session_state.subject = "Confidential: Performance Review"
            st.session_state.body = "Please review the attached performance documentation."
            st.rerun()
        
        if st.button("❓ Client Question", use_container_width=True):
            st.session_state.sender = "client@bigcorp.com"
            st.session_state.subject = "Question about pricing"
            st.session_state.body = "Hi,\n\nWhat are your pricing tiers? Do you offer discounts?\n\nThanks"
            st.rerun()
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        process_btn = st.button("🚀 Process Email", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.agent_state = None
            st.rerun()
    
    if process_btn and sender and subject and body:
        st.session_state.processing = True
        
        email = {
            "id": st.session_state.thread_id,
            "sender": sender,
            "subject": subject,
            "body": body,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
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
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner("🤔 Agent is analyzing the email... (may take 5-10 seconds)"):
            try:
                final_state = graph.invoke(initial_state, config=config)
                st.session_state.agent_state = final_state
                st.success("✅ Email processed successfully!")
                st.rerun()
            except Exception as e:
                error_msg = str(e)
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    st.error("❌ API quota exceeded! Please wait 1-2 minutes and try again.")
                    st.info("💡 Tip: The free tier has rate limits. Waiting helps reset the quota.")
                else:
                    st.error(f"❌ Error: {error_msg}")
                st.session_state.processing = False


# Results tab
with tab_results:
    if st.session_state.agent_state is None:
        st.info("👈 Compose an email in the 'Compose Email' tab and click 'Process Email' to see results here.")
    else:
        state = st.session_state.agent_state
        
        # Triage result
        triage = state.get("triage_result", "unknown")
        if triage == "respond":
            st.success(f"### 🟢 Triage Decision: RESPOND")
            st.caption("The agent will draft a response")
        elif triage == "ignore":
            st.warning(f"### 🔴 Triage Decision: IGNORE")
            st.caption("This email will be archived")
        elif triage == "notify_human":
            st.info(f"### 🟡 Triage Decision: NOTIFY HUMAN")
            st.caption("This email requires human review")
        else:
            st.info(f"### Triage Decision: {triage}")
        
        st.divider()
        
        # Draft response
        draft = state.get("draft_response")
        if draft:
            st.subheader("📝 Draft Response")
            st.text_area("Agent's Draft", value=draft, height=200, disabled=True, key="draft_display")
            
            # HITL buttons
            st.subheader("👤 Human Review")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Approve and Send", use_container_width=True):
                    st.success("✅ Email approved! (Demo: Would send in production)")
                    # Update state
                    state["hitl_decision"] = "approve"
                    st.session_state.agent_state = state
            
            with col2:
                if st.button("❌ Deny", use_container_width=True):
                    st.warning("❌ Action denied. Email will not be sent.")
                    state["hitl_decision"] = "deny"
                    st.session_state.agent_state = state
            
            with col3:
                edit_key = st.text_input("✏️ Edit Draft:", placeholder="Type your edited version here...", key="edit_input")
                if st.button("Submit Edit", use_container_width=True):
                    if edit_key:
                        from memory import extract_and_save_preferences_from_edit
                        extract_and_save_preferences_from_edit(draft, edit_key)
                        st.success("✅ Edit saved to memory! Agent will learn from this.")
                        state["draft_response"] = edit_key
                        state["hitl_decision"] = "edit"
                        st.session_state.agent_state = state
                        st.rerun()
        else:
            if triage == "respond":
                st.info("No draft response generated. Try again or check the email content.")
        
        # Final outcome
        final = state.get("final_response")
        if final:
            st.divider()
            st.subheader("✉️ Final Outcome")
            if triage == "ignore":
                st.info("📦 Email has been archived.")
            elif triage == "notify_human":
                st.warning("👁️ This email has been flagged for your review.")
            else:
                st.success(f"✅ {final}")
        
        # Debug expander
        with st.expander("🔍 Debug: Full Agent State"):
            st.json({
                "triage_result": state.get("triage_result"),
                "draft_response": state.get("draft_response"),
                "hitl_decision": state.get("hitl_decision"),
                "final_response": state.get("final_response"),
                "memory_context": state.get("memory_context"),
            })
        
        # Reset button
        if st.button("🔄 Process Another Email", use_container_width=True):
            st.session_state.agent_state = None
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()