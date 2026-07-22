# ClearTrace React frontend

This branch adds a polished React web experience without replacing or changing
the existing Streamlit entry point (`app.py`). The React client uses a dedicated
FastAPI backend-for-frontend in `web_api.py`, which calls the same forecast,
source-indicator, exposure, chatbot, and citizen-report modules already used by
Streamlit.

## What is public and what requires an account

Registration is optional. These features work without signing in:

- 24-hour AQI dashboard and station blend
- nearby mapped-source indicators
- personal exposure planner
- AQI chatbot
- location selection and browser geolocation

Citizen reports are protected. A signed-in user is required to:

- view nearby reports
- submit a report and optional photo
- view their own report history
- verify another citizen's report

Accounts, password hashes, revocable sessions, reports, and votes all use the
existing local `cleartrace.db` SQLite database. No new API key or external
authentication service is needed.

## First-time setup

From the ClearTrace repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-web.txt
cd frontend
npm install
cd ..
```

Keep the existing local `.env` configuration. For a real demo, it should include:

```env
MOCK_MODE=false
CLEARTRACE_API_URL=http://127.0.0.1:8000
CLEARTRACE_RAG_URL=http://127.0.0.1:8001
```

No authentication key needs to be added.

## Run the complete application

Open four PowerShell terminals in the repository root.

Terminal 1 — forecast service:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 — RAG/chat service (recommended, but chatbot fallbacks still work
without it):

```powershell
python -m uvicorn rag.main:app --reload --port 8001
```

Terminal 3 — React backend-for-frontend:

```powershell
python -m uvicorn web_api:app --reload --port 8002
```

Terminal 4 — React frontend:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`.

The Streamlit prototype remains available independently:

```powershell
streamlit run app.py
```

## Authentication schema

`database.init_db()` automatically creates these new tables:

### `users`

| Column | Purpose |
|---|---|
| `user_id` | UUID primary key |
| `email` | Case-insensitive unique login |
| `display_name` | Name displayed on reports and votes |
| `password_hash` | PBKDF2-SHA256 password digest |
| `password_salt` | Per-user random salt |
| `created_at` | UTC creation timestamp |
| `last_login_at` | UTC timestamp of the latest successful login |
| `is_active` | Local account disable flag |

### `user_sessions`

| Column | Purpose |
|---|---|
| `session_id` | UUID primary key |
| `user_id` | Account owner |
| `token_hash` | SHA-256 hash of the bearer token; raw tokens are never stored |
| `created_at` | UTC issue timestamp |
| `expires_at` | Seven-day expiry timestamp |
| `revoked_at` | Logout/revocation timestamp |

Passwords are never stored in plaintext. Report photos remain local and are
served only through authenticated API routes.
