# MindGuard — Claude Agent Skill File

> This file is read by Claude automatically on every session in this repository.
> Follow every rule here before touching any file. Do not skip steps.

---

## Project Context

**Project:** MindGuard — Suicidal Ideation Detector  
**Stack:** Python 3.12 · Streamlit · Mental-RoBERTa (HuggingFace) · scikit-learn · CSV data  
**Entrypoint:** `app.py` (single canonical Streamlit file)  
**Auth module:** `auth.py` (Google OAuth primary + local fallback)  
**Model:** Mental-RoBERTa fine-tuned — ROC-AUC 0.9813, Accuracy 92.5%

---

## Mandatory Pre-Flight Checks

Before editing **any** file, verify:

1. `app.py` passes syntax: `python -m py_compile app.py && echo OK`
2. `auth.py` passes syntax: `python -m py_compile auth.py && echo OK`
3. Read the section you are about to modify in full before writing

---

## File Ownership Rules

| File | Rule |
|---|---|
| `app.py` | Single entrypoint. Never split into multiple pages unless explicitly asked. |
| `auth.py` | Auth module only. No UI logic except the auth/terms pages. |
| `.env` | Never read or print. Never commit. Always gitignored. |
| `auth_config.yaml` | Auto-generated. Never commit. Always gitignored. |
| `streamlit/secrets.toml` | HuggingFace tokens live here. Never commit. Always gitignored. |
| `legacy/` | Read-only archive. Never modify files inside it. |
| `.gitignore` | Never remove existing entries. Only add. |
| `mindguard_best_weights.pt` | 498 MB model weights — hosted on HuggingFace, never git-tracked. |
| `requirements.txt` | Keep pinned versions. Always run `pip install -r requirements.txt` after editing. |

---

## Session State Contract

**Never rename or remove any of these keys.** New keys may be added but existing ones are frozen.

```python
# Auth
"authenticated"     # bool
"auth_user"         # str — email
"auth_role"         # str — workspace role
"terms_accepted"    # bool
"terms_accepted_at" # str — ISO timestamp
"theme_choice"      # str — "Auto" | "Light" | "Dark"

# User input
"user_input"        "should_analyze"    "last_result"
"input_mode"        "download_text"

# Platform results
"reddit_results"    "video_result"      "bluesky_results"
"mastodon_results"  "youtube_results"   "file_results"
"unified_results"   "facebook_results"  "twitter_results"

# Platform pending / triggered flags
"bsky_run"  "bsky_target"  "bsky_min_run"  "bsky_n_run"  "bsky_pending"
"fb_pending"  "tw_pending"  "fb_triggered"  "tw_triggered"

# Analytics
"analytics"  # dict: {total_analyses, positive_count, negative_count, history}
```

---

## Streamlit Performance Rules

```python
# Model MUST use cache_resource (loaded once, shared across sessions)
@st.cache_resource
def load_model_and_tokenizer(): ...

# Heavy per-call data fetching MUST use cache_data
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reddit(...): ...

# NEVER call st.rerun() inside a for/while loop
# NEVER place @st.cache_resource inside a conditional block
```

---

## App Structure (Canonical)

`app.py` must follow this section order exactly:

```
0. Imports (stdlib → third-party → local)
1. st.set_page_config() — MUST be first Streamlit call
2. CSS / theme styles
3. Constants & data (SAMPLE_TWEETS, STOPWORDS, SOCIOECONOMIC_KEYWORDS, RESOURCES, TEAM_MEMBERS, US_STATE_RESOURCES)
4. Session state defaults (all keys initialised here)
5. Auth helpers (reset_auth_state, render_sign_in, render_terms_dialog)
6. Auth gate (if not authenticated → render_sign_in; if not terms → render_terms_dialog)
7. Model loading (@st.cache_resource load_model_and_tokenizer)
8. Utility functions (clean_text, predict_one, predict_batch, risk_label, etc.)
9. main_app() function — all tabs live here
10. Router — calls main_app()
```

---

## Model Loading Strategy

Always try HuggingFace first, fall back to local:

```python
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # 1. Try HuggingFace (production path — uses streamlit/secrets.toml)
        tokenizer = AutoTokenizer.from_pretrained(
            st.secrets["HF_REPO_ID"],
            token=st.secrets["HF_TOKEN"],
            subfolder="mindguard_tokenizer"
        )
        weights_path = hf_hub_download(...)
        ...
    except Exception:
        # 2. Fall back to local files
        tokenizer = AutoTokenizer.from_pretrained("mindguard_tokenizer")
        state_dict = torch.load("mindguard_best_weights.pt", map_location=device)
        ...
```

---

## Auth Module Rules (`auth.py`)

- `render_auth_page()` — unified entry point; returns True when authenticated
- `init_google_auth()` — wraps `streamlit_google_auth`; gracefully degrades if env vars missing
- `init_local_auth()` — wraps `streamlit_authenticator`; auto-creates `auth_config.yaml` on first run
- Never hardcode credentials. Always read from `os.environ` or `st.secrets`

---

## Checkpoint Protocol

After every significant change, create a checkpoint file:

```bash
mkdir -p checkpoints
echo "$(date -Iseconds) | <what changed> | $(python -m py_compile app.py && echo PASS || echo FAIL)" >> checkpoints/log.txt
```

---

## Code Quality Gates

Run before calling any task done:

```bash
python -m py_compile app.py && echo "SYNTAX OK"
python -m py_compile auth.py && echo "SYNTAX OK"
grep -n "cache_resource\|cache_data" app.py   # verify caching
grep -n "st.rerun" app.py                       # verify no rerun in loops
```

Full quality suite (after pip install):

```bash
ruff check app.py auth.py
bandit -r app.py auth.py -ll
```

---

## Git Rules

- Never commit: `.env`, `auth_config.yaml`, `streamlit/`, `*.pt`, `*.csv`, `*.txt` data files
- The `gitignore` file (without dot) is the legacy file — `.gitignore` (with dot) is the active one
- Always verify `.gitignore` is active before committing: `git check-ignore -v <file>`
- Branch naming: `feature/<name>`, `fix/<name>`, `chore/<name>`

---

## Sensitive Content Warning

MindGuard handles suicidal ideation data. When reading training data or analysis results:
- Never print raw post content to logs
- Never store analysis results between sessions
- Always include crisis resource references in any user-facing error about risk detection

---

*CLAUDE.md version 1.0 — MindGuard project, May 2026*
