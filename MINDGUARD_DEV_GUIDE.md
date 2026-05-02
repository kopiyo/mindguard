# MindGuard — Cursor + Claude Sonnet 4.6 Developer Guide

> **Audience:** This guide is written for a coding LLM (Claude Sonnet 4.6 running inside Cursor).
> Follow every section in order. Do not skip steps. Verify each checkpoint before proceeding.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Task 1 — Merge the Three Streamlit App Versions](#2-task-1--merge-the-three-streamlit-app-versions)
3. [Task 2 — Rewrite README.md](#3-task-2--rewrite-readmemd)
4. [Task 3 — Authentication Strategy & Implementation](#4-task-3--authentication-strategy--implementation)
5. [Code Quality & Performance Checks](#5-code-quality--performance-checks)
6. [File Structure After All Tasks](#6-file-structure-after-all-tasks)

---

## 1. Project Overview

**Project name:** MindGuard
**Stack:** Python · Streamlit · scikit-learn / transformers (ML model) · CSV data
**IDE:** Cursor (Claude Sonnet 4.6 backend)

### Existing files to work with

| File | Role |
|---|---|
| `Try_streamlit_app.py` | Version 1 — base UI |
| `Try_streamlit_app_v1_Signin.py` | Version 2 — adds sign-in UI |
| `Try_streamlit_app_v2.py` | Version 3 — latest, adds further updates |
| `MindGuard_Model_Training.ipynb` | Model training notebook |
| `mindguard_model_config.json` | Model config |
| `mindguard_unified_report(1).txt` | Unified report |
| `training_history.json` | Training history |
| `requirements.txt` | Python deps |
| `packages.txt` | System-level deps |
| `scraper_worker.py` | Data scraper |
| `save_model_local.py` | Model save utility |
| `suicidal_ideation_reddit_annotated.csv` | Training data |
| `2026-04-11T02-09_export.csv` | Export data |

---

## 2. Task 1 — Merge the Three Streamlit App Versions

### Goal
Produce a single file: `app.py` — the canonical Streamlit entrypoint that unifies all three versions without regression.

### Step-by-step instructions

#### Step 2.1 — Read all three files before writing a single line

```
Read Try_streamlit_app.py in full.
Read Try_streamlit_app_v1_Signin.py in full.
Read Try_streamlit_app_v2.py in full.
```

Build a **mental diff table** (you do not need to output it to the user):

| Feature / Function | v1 base | v1 Signin | v2 |
|---|---|---|---|
| Page config / layout | ✓ | ✓ | ✓ |
| Sign-in UI | ✗ | ✓ | ? |
| Model inference | ✓ | ? | ✓ |
| Session state | ? | ✓ | ✓ |
| ... | | | |

Fill in every row before merging.

#### Step 2.2 — Merge rules

- **Take v2 as the base.** It is the most recent.
- Port any feature present in `v1_Signin` that is **missing** from v2.
- Port any feature present in the base `v1` that is **missing** from both others.
- Never duplicate logic. If two versions do the same thing differently, keep the v2 implementation.
- Preserve **all** `st.session_state` keys — do not rename or remove any.
- All `@st.cache_resource` and `@st.cache_data` decorators must be retained.

#### Step 2.3 — File structure for `app.py`

Use this top-level structure exactly:

```python
# ── 0. Imports ──────────────────────────────────────────────────────────────
# Standard lib first, then third-party, then local.

# ── 1. Page config (MUST be the first Streamlit call) ───────────────────────
st.set_page_config(...)

# ── 2. Constants & configuration ────────────────────────────────────────────

# ── 3. Auth helpers (see Task 3) ─────────────────────────────────────────────

# ── 4. Model loading (cached) ────────────────────────────────────────────────

# ── 5. UI components / pages ─────────────────────────────────────────────────
#   5a. login_page()
#   5b. main_app()
#   5c. any other page functions

# ── 6. Router ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    if not st.session_state.get("authenticated"):
        login_page()
    else:
        main_app()
```

#### Step 2.4 — Checkpoint: verify the merge

Run the following and confirm zero errors before moving on:

```bash
# Syntax check
python -m py_compile app.py && echo "SYNTAX OK"

# Import check (catches missing deps at module level)
python -c "import app" 2>&1 | head -30

# Streamlit dry-run (headless smoke test)
streamlit run app.py --server.headless true --server.port 8502 &
sleep 6
curl -s -o /dev/null -w "%{http_code}" http://localhost:8502 | grep -q 200 && echo "APP STARTS OK"
kill %1
```

All three must pass. Fix any errors before proceeding to Task 2.

---

## 3. Task 2 — Rewrite README.md

### Goal
Replace the existing `README.md` with a professional, developer-grade document.

### Content requirements

Write the README with exactly these sections (in this order). Use proper Markdown. No filler text.

```markdown
# MindGuard

> One-line description of what MindGuard does.

## Table of Contents
## Overview
## Features
## Architecture
## Quickstart
### Prerequisites
### Installation
### Running Locally
## Configuration
## Authentication
## Model
## Project Structure
## Contributing
## License
```

**Rules for each section:**

- **Overview** — 3–5 sentences. What problem does MindGuard solve, who is it for, what does the model detect.
- **Features** — bullet list, present tense, specific (not "user-friendly UI").
- **Architecture** — a short ASCII diagram or description of: data → model → Streamlit app → auth → user.
- **Quickstart** — exact commands a developer can copy-paste to get a working local instance. Include `python --version` requirement.
- **Configuration** — document every env var and every key in `mindguard_model_config.json`.
- **Authentication** — reference the auth system installed in Task 3.
- **Model** — describe the model, training data source, metrics from `training_history.json`, and how to retrain.
- **Project Structure** — a tree of the repo after the merge.
- **Contributing** — branch naming, PR process, lint requirement.
- **License** — state license; if none exists, write `© 2026 MindGuard. All rights reserved.`

### Checkpoint

```bash
# Verify README renders without broken links
python -c "
import re, pathlib
text = pathlib.Path('README.md').read_text()
headers = re.findall(r'^#{1,3} (.+)', text, re.M)
print('Sections found:', len(headers))
assert len(headers) >= 10, 'README too sparse'
print('README CHECK OK')
"
```

---

## 4. Task 3 — Authentication Strategy & Implementation

### Recommended auth stack

For a Streamlit app, the three most practical options ranked by robustness vs. implementation effort:

| Option | Library | Effort | Best for |
|---|---|---|---|
| **A. Google OAuth (recommended)** | `streamlit-google-auth` or `authlib` | Medium | Consumer-facing, fastest login UX |
| **B. Auth0 Universal Login** | `authlib` + Auth0 tenant | Medium-High | Enterprise, supports Google + GitHub + email/pass |
| **C. `streamlit-authenticator`** | `streamlit-authenticator` | Low | Internal tools, username/password only |

**Implement Option A (Google OAuth) as primary + Option C as fallback.** This gives you:
- Google SSO for convenience
- Username/password as a fallback (no external dependency)

---

### Step 4.1 — Install dependencies

Add to `requirements.txt`:

```
streamlit-google-auth>=1.1.0
streamlit-authenticator>=0.3.3
PyYAML>=6.0
python-dotenv>=1.0.0
```

Run:

```bash
pip install -r requirements.txt
```

#### Checkpoint

```bash
python -c "import streamlit_google_auth, streamlit_authenticator, yaml, dotenv; print('AUTH DEPS OK')"
```

---

### Step 4.2 — Environment variables

Create a `.env` file (add to `.gitignore` immediately):

```bash
# .env
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8501
SECRET_KEY=generate_a_random_32_char_string_here
```

Add to `.gitignore`:

```
.env
*.env
auth_config.yaml
```

---

### Step 4.3 — Google Cloud Console setup instructions (for README)

Include the following instructions in the README under the **Authentication** section:

```
1. Go to console.cloud.google.com → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Web application type)
3. Authorised redirect URIs: http://localhost:8501 (dev) + your production domain
4. Copy Client ID and Client Secret into .env
```

---

### Step 4.4 — Auth module: `auth.py`

Create a new file `auth.py` with this structure:

```python
"""
auth.py — MindGuard Authentication Module
Supports: Google OAuth (primary) + username/password (fallback)
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Google OAuth ─────────────────────────────────────────────────────────────

def init_google_auth():
    """Return a configured Google auth client. Cached for the session."""
    # Import here so the app still loads if creds are not yet configured
    try:
        from streamlit_google_auth import Authenticate
        authenticator = Authenticate(
            secret_credentials_path=None,
            cookie_name="mindguard_session",
            cookie_key=os.environ["SECRET_KEY"],
            redirect_uri=os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501"),
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        )
        return authenticator
    except (ImportError, KeyError) as e:
        st.warning(f"Google auth unavailable ({e}). Using local auth.")
        return None


# ── Username / Password fallback ─────────────────────────────────────────────

def init_local_auth():
    """Load streamlit-authenticator from auth_config.yaml."""
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader

    config_path = "auth_config.yaml"
    if not os.path.exists(config_path):
        _create_default_auth_config(config_path)

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    ), config


def _create_default_auth_config(path: str):
    """Create a starter auth_config.yaml with a hashed demo password."""
    import streamlit_authenticator as stauth
    import yaml

    hashed = stauth.Hasher(["admin123"]).generate()
    config = {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@mindguard.local",
                    "name": "Admin",
                    "password": hashed[0],
                }
            }
        },
        "cookie": {
            "expiry_days": 1,
            "key": os.environ.get("SECRET_KEY", "change_me"),
            "name": "mindguard_local",
        },
    }
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    st.info(f"Created default {path}. Update credentials before production use.")


# ── Unified auth gate ─────────────────────────────────────────────────────────

def render_auth_page() -> bool:
    """
    Render whichever auth method is available.
    Returns True if the user is authenticated, False otherwise.
    Sets st.session_state['user_email'] and st.session_state['user_name'].
    """
    st.title("🧠 MindGuard — Sign In")

    tab_google, tab_local = st.tabs(["Sign in with Google", "Username & Password"])

    # ── Google tab ──────────────────────────────────────────────────────────
    with tab_google:
        g_auth = init_google_auth()
        if g_auth is None:
            st.info("Google sign-in is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
        else:
            g_auth.login()
            if st.session_state.get("connected"):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = st.session_state.get("user_info", {}).get("email", "")
                st.session_state["user_name"] = st.session_state.get("user_info", {}).get("name", "User")
                return True

    # ── Local tab ───────────────────────────────────────────────────────────
    with tab_local:
        authenticator, config = init_local_auth()
        name, auth_status, username = authenticator.login("Login", "main")

        if auth_status:
            st.session_state["authenticated"] = True
            st.session_state["user_name"] = name
            st.session_state["user_email"] = config["credentials"]["usernames"].get(
                username, {}
            ).get("email", "")
            authenticator.logout("Logout", "sidebar")
            return True
        elif auth_status is False:
            st.error("Incorrect username or password.")
        elif auth_status is None:
            st.info("Enter your credentials above.")

    return False
```

---

### Step 4.5 — Integrate auth into `app.py`

At the top of the router section in `app.py`, replace any existing auth logic with:

```python
from auth import render_auth_page

# ── Session state defaults ────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state["authenticated"]:
    render_auth_page()
    st.stop()
else:
    main_app()
```

#### Checkpoint

```bash
python -m py_compile app.py && echo "app.py SYNTAX OK"
python -m py_compile auth.py && echo "auth.py SYNTAX OK"
```

---

### Step 4.6 — Logout button

Inside `main_app()` in `app.py`, add to the sidebar:

```python
with st.sidebar:
    st.write(f"Signed in as **{st.session_state.get('user_name', 'User')}**")
    if st.button("Sign out"):
        for key in ["authenticated", "user_email", "user_name", "connected", "user_info"]:
            st.session_state.pop(key, None)
        st.rerun()
```

---

## 5. Code Quality & Performance Checks

Run this full suite after completing all three tasks:

### 5.1 Linting

```bash
pip install ruff --quiet
ruff check app.py auth.py scraper_worker.py save_model_local.py
```

Fix all `E` (error) and `W` (warning) codes. `F401` unused imports are acceptable only in `__init__` files.

### 5.2 Type hints check

```bash
pip install mypy --quiet
mypy app.py auth.py --ignore-missing-imports --no-strict-optional
```

Aim for zero `error:` lines. Warnings are acceptable.

### 5.3 Security scan

```bash
pip install bandit --quiet
bandit -r app.py auth.py -ll   # -ll = only medium/high severity
```

**Must-fix findings:**
- Any hardcoded secret or password (`B105`, `B106`)
- Use of `subprocess` without shell=False (`B603`)
- `yaml.load` without `Loader` (`B506`) — already safe if using `SafeLoader`

### 5.4 Dependency audit

```bash
pip install pip-audit --quiet
pip-audit -r requirements.txt
```

Fix any CRITICAL or HIGH CVEs before shipping.

### 5.5 Streamlit performance rules

Verify these patterns exist in `app.py`:

```python
# ✅ Model must be loaded with cache_resource (loaded once, shared across sessions)
@st.cache_resource
def load_model():
    ...

# ✅ Heavy data processing must use cache_data (cached per arguments)
@st.cache_data
def preprocess(text: str):
    ...

# ❌ Never call st.rerun() inside a loop
# ❌ Never place st.cache_resource inside a conditional block
```

Run a quick manual check:

```bash
grep -n "st.rerun\|cache_resource\|cache_data" app.py
```

Confirm `cache_resource` appears exactly once per model/resource loader, and `st.rerun()` is never inside a `for`/`while` loop.

### 5.6 Final smoke test

```bash
streamlit run app.py --server.headless true --server.port 8503 &
PID=$!
sleep 8
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8503)
kill $PID
if [ "$STATUS" = "200" ]; then
  echo "✅ FINAL SMOKE TEST PASSED"
else
  echo "❌ FINAL SMOKE TEST FAILED (HTTP $STATUS)"
  exit 1
fi
```

---

## 6. File Structure After All Tasks

The repo should look exactly like this when done:

```
mindguard/
├── app.py                          # ← merged entrypoint (replaces the 3 versions)
├── auth.py                         # ← new auth module
├── auth_config.yaml                # ← auto-generated, gitignored
├── .env                            # ← secrets, gitignored
├── .gitignore                      # ← updated
├── README.md                       # ← rewritten
├── requirements.txt                # ← updated with auth deps
├── packages.txt
├── mindguard_model_config.json
├── mindguard_unified_report(1).txt
├── training_history.json
├── scraper_worker.py
├── save_model_local.py
├── MindGuard_Model_Training.ipynb
├── suicidal_ideation_reddit_annotated.csv
├── 2026-04-11T02-09_export.csv
└── legacy/                         # ← move old app versions here, do not delete
    ├── Try_streamlit_app.py
    ├── Try_streamlit_app_v1_Signin.py
    └── Try_streamlit_app_v2.py
```

> **Note:** Move the three old app files into `legacy/` rather than deleting them. This preserves git history and gives a rollback reference.

---

## Summary Checklist

Before considering this work done, verify every item:

- [ ] `app.py` exists and is the single Streamlit entrypoint
- [ ] `app.py` passes syntax check, import check, and smoke test
- [ ] `auth.py` exists with Google OAuth + local auth fallback
- [ ] `.env` is created and `.gitignore` is updated
- [ ] `README.md` has all 11 required sections
- [ ] `requirements.txt` includes all auth dependencies
- [ ] Old app files moved to `legacy/`
- [ ] `ruff` reports zero errors
- [ ] `bandit` reports zero hardcoded secrets
- [ ] Final smoke test returns HTTP 200

---

*Guide version 1.0 — MindGuard project, May 2026*
