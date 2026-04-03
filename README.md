#  Ambient Email Agent

An AI-powered autonomous email assistant built using **LangGraph + Google Gemini** that can intelligently classify, respond to, and manage emails.

---

##  Features

* 📩 Email Triage (Ignore / Notify / Respond)
* 🤖 AI-generated email replies
* 🧠 Memory system using SQLite
* 🛠 Tool-based actions (send email, etc.)
* 👨‍💻 Human-in-the-loop approval system
* 🌐 Streamlit Web UI

---

##  Tech Stack

* Python
* LangGraph
* LangChain
* Google Gemini API
* SQLite
* Streamlit

---

## Project Structure

```
EMAILAGENT/
│
├── app.py              # Streamlit UI
├── main.py             # CLI entry
├── graph.py            # LangGraph workflow
├── nodes.py            # Core agent logic
├── memory.py           # SQLite memory
├── tools.py            # Tools (email, etc.)
├── state.py            # Agent state
├── requirements.txt
├── .env                # API keys (not committed)
```

---

## Setup Instructions

### 1️⃣ Clone the repo

```
git clone https://github.com/your-username/email-agent.git
cd email-agent
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Setup environment variables

Create `.env` file:

```
GOOGLE_API_KEY=your_google_api_key
```

---

### 4️⃣ Run the app

#### ▶️ Streamlit UI

```
streamlit run app.py
```

#### ▶️ CLI mode

```
python main.py --demo 0
```

---

## Sample Inputs

* Meeting Request → Respond
* Spam Email → Ignore
* HR Email → Notify Human

---

## How it works

1. Loads memory (SQLite)
2. Classifies email (Triage)
3. Uses tools via LangGraph
4. Generates response
5. Human approval (if needed)
6. Logs interaction

---

## Environment Variables

| Variable               | Description          |
| ---------------------- | -------------------- |
| GOOGLE_API_KEY         | Gemini API key       |
| LANGCHAIN_API_KEY      | (Optional) LangSmith |
| GMAIL_CREDENTIALS_PATH | (Optional) Gmail API |
| EMAIL_ADDRESS          | User email           |

---

##  Future Improvements

* Gmail API integration
* Deployment (Render / Vercel)
* Advanced memory personalization
* Multi-user support

---

##  Author

Built by **MUVVA THIRAPATHA SWAMI**

---
