# MindGuard

> AI-powered clinical decision-support tool for early detection of suicidal ideation in digital text, built on Mental-RoBERTa and Streamlit.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Model](#model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MindGuard is a consent-first, human-in-the-loop clinical decision-support system for mental health professionals. It helps trained practitioners — school psychologists, licensed counselors, and designated mental health staff — identify early signals of suicidal ideation in consented digital text across nine social platforms and file formats.

The system is powered by Mental-RoBERTa, a transformer pre-trained on millions of mental health domain posts and fine-tuned on 12,656 annotated examples, achieving a ROC-AUC of 0.9813 and 92.5% accuracy. MindGuard does not diagnose, does not automate decisions, and does not store data between sessions — every output requires a qualified human reviewer before any action is taken.

---

## Features

- **Multi-platform analysis** — Reddit, Bluesky, Mastodon, YouTube, TikTok, Twitter/X, Facebook (public), and any video URL via Whisper transcription
- **File upload** — WhatsApp exports (.txt), CSV spreadsheets, Facebook/Twitter JSON archives
- **Real-time risk scoring** — four-tier classification: Low / Moderate / High / Critical
- **Socio-economic signal detection** — keyword scan across six distress categories (employment, housing, financial, relationships, health, education)
- **Multi-platform unified profile** — aggregates risk scores across all platforms analysed in a session
- **Crisis resources** — built-in lookup by country and US state (50 states + D.C.)
- **Terms and consent gate** — practitioners must agree to a Practitioner Use and Responsibility Agreement before workspace access
- **Three-method authentication** — Google OAuth, username/password (streamlit-authenticator), and Quick Access prototype fallback
- **Session-only data** — no persistent storage of analysed content

---

## Architecture

```
User
  |
  v
[Sign-in Gate]  <--  auth.py (Google OAuth | username/password | Quick Access)
  |
  v
[Terms Dialog]  <--  Practitioner Use Agreement
  |
  v
[Streamlit app.py]
  |-- Text / Image input  -->  pytesseract (OCR)
  |-- Social platform tabs -->  Reddit API | Bluesky API | Mastodon API | YouTube API | Scraper worker
  |-- Video tab  -->  yt-dlp + faster-whisper (Whisper tiny)
  |-- File upload tab  -->  CSV | JSON | WhatsApp .txt parser
  |
  v
[Mental-RoBERTa] (@st.cache_resource)
  |-- Loaded from HuggingFace (HF_REPO_ID + HF_TOKEN in secrets.toml)
  `-- Falls back to local mindguard_model_local/ directory
  |
  v
[Risk Score 0-1]  -->  4-tier classification  -->  Socio-economic signal overlay
  |
  v
[Practitioner review]  -->  Crisis resources lookup  -->  Download report
```

---

## Quickstart

### Prerequisites

- Python >= 3.12
- `python3.12-venv` installed (`sudo apt install python3.12-venv -y` on Ubuntu/Debian)
- `tesseract-ocr` for image OCR (`sudo apt install tesseract-ocr -y`)
- `ffmpeg` for video audio trimming (`sudo apt install ffmpeg -y`)

### Installation

```bash
git clone https://github.com/kopiyo/mindguard.git
cd mindguard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and fill in your values
cp .env.example .env
```

### Running Locally

```bash
source .venv/bin/activate
streamlit run app.py
```

The app starts at `http://localhost:8501`.

---

## Configuration

### Environment Variables (`.env`)

Copy `.env.example` to `.env` and fill in the values. The file is gitignored and never committed.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CREDENTIALS_FILE` | For Google auth | Path to the JSON file downloaded from Google Cloud Console. Default: `google_credentials.json` if the file exists in the project root |
| `GOOGLE_CLIENT_ID` | Alternative to JSON file | OAuth 2.0 Client ID (use only if you cannot use the JSON file) |
| `GOOGLE_CLIENT_SECRET` | Alternative to JSON file | OAuth 2.0 Client Secret (use only if you cannot use the JSON file) |
| `GOOGLE_REDIRECT_URI` | No (default: `http://localhost:8501`) | OAuth redirect URI — must match what is registered in Google Cloud Console |
| `OAUTHLIB_INSECURE_TRANSPORT` | Local dev only | Set to `1` when running over HTTP locally. Allows the OAuth callback over plain HTTP. Remove in production (HTTPS handles it automatically) |
| `SECRET_KEY` | Yes | Random hex string for cookie signing. Generate with `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `REDDIT_CLIENT_ID` | For Reddit tab | Free at reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | For Reddit tab | Reddit script app secret |
| `YOUTUBE_API_KEY` | For YouTube tab | YouTube Data API v3 key |

The built-in `.env` loader in `auth.py` parses this file at startup — `python-dotenv` is not required, though it is listed as a dependency and works if installed.

### Streamlit Secrets (`streamlit/secrets.toml`)

For HuggingFace model loading (production path):

```toml
HF_REPO_ID = "your-hf-username/mindguard-model"
HF_TOKEN   = "hf_your_token_here"
```

`streamlit/secrets.toml` is gitignored.

### Model Config (`mindguard_model_config.json`)

| Key | Value | Description |
|---|---|---|
| `model_type` | `mental_roberta` | Model family identifier |
| `model_name` | `mental/mental-roberta-base` | HuggingFace model ID for base weights |
| `max_length` | `256` | Tokenizer max sequence length |
| `test_accuracy` | `0.925` | Final test set accuracy |
| `test_auc` | `0.9813` | Final test set ROC-AUC |

---

## Authentication

Authentication is handled in `auth.py` and presented as three tabs in the sign-in screen.

### Tab 1: Google OAuth (primary)

Uses `streamlit-google-auth`. Credential resolution follows this priority order:

1. `GOOGLE_CREDENTIALS_FILE` env var pointing to the downloaded JSON file
2. `google_credentials.json` present in the project root (auto-detected, no env var needed)
3. `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` env vars (JSON built on the fly)

**Google Cloud Console setup:**

1. Go to [console.cloud.google.com](https://console.cloud.google.com) > APIs & Services > Credentials
2. Create an OAuth 2.0 Client ID (Web application type)
3. Add Authorised redirect URIs: `http://localhost:8501` for development, your deployed domain for production
4. Click the download icon next to your client to get the JSON file
5. Save it to the project root as `google_credentials.json` (already in `.gitignore`)
6. Add `OAUTHLIB_INSECURE_TRANSPORT=1` to `.env` for local HTTP development

**Note on client secrets:** Google Cloud Console no longer displays existing client secrets. If a secret is lost or needs rotation, create a new one from the Credentials page and re-download the JSON file. Never copy-paste secrets into code — always use the downloaded JSON file.

### Tab 2: Username / Password

Uses `streamlit-authenticator` >= 0.4.x with `auto_hash=True`. On first run, `auth.py` auto-generates `auth_config.yaml` with demo credentials:

- Username: `admin`
- Password: `admin123`

Replace these before any non-development deployment. `auth_config.yaml` is gitignored.

### Tab 3: Quick Access

Prototype fallback that accepts any email and password. Used to access the workspace during early development without configuring either auth system. Replace or remove before production.

### After Login

A Terms and Conditions dialog requires practitioners to accept a Practitioner Use and Responsibility Agreement before accessing the analysis workspace. This is enforced on every new session.

---

## Model

**Model:** Mental-RoBERTa (`mental/mental-roberta-base`)
**Task:** Binary sequence classification (suicidal ideation vs. non-suicidal)
**Training data:** 12,656 annotated Reddit posts from mental health communities
**Split:** 75% train / 10% validation / 15% test

### Performance (from `training_history.json`)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Bi-LSTM (baseline) | 88.0% | 0.9323 |
| RoBERTa | 91.2% | 0.9762 |
| **Mental-RoBERTa (winner)** | **92.5%** | **0.9813** |

### Mental-RoBERTa Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|
| 1 | 0.3302 | 85.5% | 0.2447 | 91.4% | 0.9703 |
| 2 | 0.1873 | 93.0% | 0.2208 | 91.4% | 0.9787 |
| 3 | 0.1286 | 95.5% | 0.2638 | 92.1% | 0.9794 |
| 4 | 0.0944 | 96.9% | 0.3370 | 92.4% | 0.9781 |

### Retraining

```bash
# Open the training notebook
jupyter notebook MindGuard_Model_Training.ipynb

# Save updated weights locally
python3 save_model_local.py

# Upload to HuggingFace
python3 upload_to_hf.py
```

---

## Project Structure

```
mindguard/
|-- app.py                            # Canonical Streamlit entrypoint (merged from 3 versions)
|-- auth.py                           # Authentication module (Google OAuth + local + Quick Access)
|-- auth_config.yaml                  # Auto-generated on first run — gitignored
|-- google_credentials.json           # Google OAuth credentials JSON — gitignored
|-- .env                              # Secrets and config — gitignored
|-- .env.example                      # Template — safe to commit
|-- .gitignore
|-- CLAUDE.md                         # AI assistant skill file with project rules
|-- MINDGUARD_DEV_GUIDE.md            # Developer guide
|-- README.md                         # This file
|-- requirements.txt                  # Python dependencies
|-- packages.txt                      # System-level dependencies
|-- mindguard_model_config.json       # Model configuration — gitignored
|-- scraper_worker.py                 # Headless browser scraper (Facebook/Twitter)
|-- save_model_local.py               # Save model weights locally
|-- MindGuard_Model_Training.ipynb    # Training notebook
|-- checkpoints/
|   `-- log.txt                       # Dev checkpoint log
|-- assets/                           # Images and screenshots
|-- streamlit/                        # Streamlit config — gitignored
|   `-- secrets.toml
|-- .venv/                            # Virtual environment — gitignored
`-- legacy/                           # Archived original app versions — read-only
    |-- Try_streamlit_app.py
    |-- Try_streamlit_app_v1_Signin.py
    `-- Try_streamlit_app_v2.py
```

---

## Contributing

**Branch naming:**
- `feature/<description>` — new features
- `fix/<description>` — bug fixes
- `chore/<description>` — maintenance, dependency updates

**PR process:**
1. Fork and create a branch from `main`
2. Run `python -m py_compile app.py auth.py` — zero errors required
3. Run `ruff check app.py auth.py` — zero E/W codes
4. Run `bandit -r app.py auth.py -ll` — zero hardcoded secrets
5. Open a PR with a description of what changed and why
6. At least one reviewer must approve before merge

**Lint requirement:** `ruff` with default settings. F401 unused imports are acceptable only in `__init__` files.

---

## License

© 2026 MindGuard. All rights reserved.

MindGuard is a research prototype. It is not a certified medical device and must not be used as the sole basis for any clinical decision, diagnosis, or treatment. If someone is in immediate danger, call emergency services.
