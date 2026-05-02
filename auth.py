"""
auth.py — MindGuard Authentication Module
Supports: Google OAuth (primary) + username/password fallback via streamlit-authenticator
Compatible with: streamlit-authenticator >= 0.4.x

Google credentials — three options (first match wins):
  1. GOOGLE_CREDENTIALS_FILE=/path/to/file.json  in .env  (explicit path)
  2. google_credentials.json present in the project root  (auto-detected)
  3. GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET in .env      (built on the fly)
"""

import json
import os

import streamlit as st


# .env loader (no python-dotenv required)

def _load_env(path: str = ".env") -> None:
    """Parse a .env file into os.environ. Real env vars are never overwritten."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


_load_env()

# Allow OAuth over plain HTTP for local development.
# oauthlib rejects non-HTTPS callbacks unless this is set.
# In production (HTTPS), this has no effect.
if os.environ.get("GOOGLE_REDIRECT_URI", "").startswith("http://"):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")


# 1. Google OAuth

_DEFAULT_CREDS_PATH = "google_credentials.json"  # gitignored


def _resolve_google_creds_path() -> str | None:
    """
    Return a path to a valid Google client-secrets JSON file, or None.

    Priority:
      1. GOOGLE_CREDENTIALS_FILE env var (or st.secrets) — explicit path
      2. google_credentials.json in the project root — auto-detected
      3. GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET env vars — JSON built on the fly
    """
    # Priority 1: explicit path via env var
    creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "")
    try:
        creds_file = creds_file or st.secrets.get("GOOGLE_CREDENTIALS_FILE", "")
    except Exception:
        pass
    if creds_file and os.path.isfile(creds_file):
        return creds_file

    # Priority 2: well-known filename in project root
    if os.path.isfile(_DEFAULT_CREDS_PATH):
        return _DEFAULT_CREDS_PATH

    # Priority 3: individual env vars → build JSON on the fly
    client_id     = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    redirect_uri  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8502")
    try:
        client_id     = client_id     or st.secrets.get("GOOGLE_CLIENT_ID", "")
        client_secret = client_secret or st.secrets.get("GOOGLE_CLIENT_SECRET", "")
        redirect_uri  = redirect_uri  or st.secrets.get("GOOGLE_REDIRECT_URI", redirect_uri)
    except Exception:
        pass

    if not client_id or not client_secret:
        return None

    payload = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [redirect_uri],
        }
    }
    with open(_DEFAULT_CREDS_PATH, "w") as f:
        json.dump(payload, f)
    return _DEFAULT_CREDS_PATH


def _read_redirect_uri(creds_path: str) -> str:
    """
    Read the first redirect_uri from the credentials JSON.
    Falls back to GOOGLE_REDIRECT_URI env var, then localhost default.
    """
    try:
        with open(creds_path) as f:
            data = json.load(f)
        section = data.get("web", data.get("installed", {}))
        uris = section.get("redirect_uris", [])
        if uris:
            return uris[0]
    except Exception:
        pass
    return os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8502")


def init_google_auth():
    """
    Return a configured streamlit-google-auth Authenticate instance, or None.
    Returns None if no credentials are found — the UI falls back gracefully.
    """
    try:
        from streamlit_google_auth import Authenticate

        creds_path = _resolve_google_creds_path()
        if not creds_path:
            return None

        redirect_uri = _read_redirect_uri(creds_path)
        secret_key   = os.environ.get("SECRET_KEY", "")
        try:
            secret_key = secret_key or st.secrets.get("SECRET_KEY", "")
        except Exception:
            pass
        if not secret_key:
            secret_key = "mindguard_dev_key_replace_before_production"

        return Authenticate(
            secret_credentials_path=creds_path,
            redirect_uri=redirect_uri,
            cookie_name="mindguard_session",
            cookie_key=secret_key,
        )
    except ImportError:
        return None
    except Exception:
        return None


# 2. Username / Password fallback (streamlit-authenticator >= 0.4.x)

def init_local_auth():
    """
    Load streamlit-authenticator from auth_config.yaml.
    Auto-creates the file with demo credentials on first run.
    Passwords are stored plain; auto_hash=True hashes them at runtime.
    """
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader

    config_path = "auth_config.yaml"
    if not os.path.exists(config_path):
        _create_default_auth_config(config_path)

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    secret_key = os.environ.get("SECRET_KEY", "mindguard_dev_key_replace_before_production")

    authenticator = stauth.Authenticate(
        credentials=config["credentials"],
        cookie_name=config["cookie"]["name"],
        cookie_key=secret_key,
        cookie_expiry_days=config["cookie"]["expiry_days"],
        auto_hash=True,
    )
    return authenticator, config


def _create_default_auth_config(path: str) -> None:
    """
    Write a starter auth_config.yaml with plain-text demo password.
    streamlit-authenticator 0.4.x hashes it automatically (auto_hash=True).
    """
    import yaml

    config = {
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@mindguard.local",
                    "name": "Admin",
                    "password": "admin123",
                }
            }
        },
        "cookie": {
            "expiry_days": 1,
            "key": os.environ.get("SECRET_KEY", "mindguard_dev_key_replace_before_production"),
            "name": "mindguard_local",
        },
    }
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    st.info(
        f"Created {path} with demo credentials (username: admin / password: admin123). "
        "Replace before any production deployment."
    )
