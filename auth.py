"""
auth.py — Light-blue dark theme · Login + Register · Animated hero
render_login_page()    → (None, None)
check_authentication() → (bool, str|None, str|None)
render_logout()        → global theme + fixed top-right pill
"""

import streamlit as st
import yaml
import bcrypt
import os
import re
from yaml.loader import SafeLoader

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")



_LOGIN_CSS = """<style>
.stApp { background:linear-gradient(135deg,#00060f 0%,#001524 50%,#000d1a 100%); min-height:100vh; }

/* ── Card ── */
div[data-testid="stForm"] {
  background:linear-gradient(145deg,rgba(14,165,233,0.07) 0%,rgba(56,189,248,0.04) 100%) !important;
  backdrop-filter:blur(20px) !important; -webkit-backdrop-filter:blur(20px) !important;
  border:1px solid rgba(56,189,248,0.22) !important; border-radius:22px !important;
  padding:38px 34px !important; max-width:420px; margin:0 auto !important;
  box-shadow:0 0 0 1px rgba(14,165,233,0.06),0 8px 40px rgba(2,132,199,0.18),0 2px 8px rgba(0,0,0,0.7) !important;
}

/* ── Labels ── */
div[data-testid="stTextInput"] label {
  color:#7dd3fc !important; font-size:0.8rem !important; font-weight:700 !important;
  letter-spacing:0.07em !important; text-transform:uppercase !important;
}

/* ── Inputs ── */
div[data-testid="stTextInput"] input {
  background:rgba(0,15,30,0.75) !important; color:#e0f2fe !important;
  border:1px solid rgba(56,189,248,0.28) !important; border-radius:11px !important;
  padding:11px 15px !important; font-size:0.95rem !important;
  caret-color:#38bdf8 !important;
  transition:border-color 0.2s,box-shadow 0.2s,background 0.2s !important;
}
div[data-testid="stTextInput"] input::placeholder { color:rgba(125,211,252,0.3) !important; }
div[data-testid="stTextInput"] input:focus {
  border-color:rgba(56,189,248,0.6) !important;
  box-shadow:0 0 0 3px rgba(56,189,248,0.14),0 2px 8px rgba(14,165,233,0.12) !important;
  background:rgba(0,20,40,0.88) !important; outline:none !important;
}

/* ── Google-style eye icon: sits inside the input row, right-aligned ── */
div[data-testid="stTextInput"] > div {
  position:relative !important;
  display:flex !important;
  align-items:center !important;
}
div[data-testid="stTextInput"] > div > div:first-child { flex:1 !important; }
div[data-testid="stTextInput"] > div button {
  position:relative !important; right:auto !important; top:auto !important;
  transform:none !important;
  width:32px !important; height:32px !important; min-height:unset !important;
  padding:6px !important; margin-left:4px !important; flex-shrink:0 !important;
  border:none !important; background:transparent !important;
  box-shadow:none !important; border-radius:50% !important;
  display:flex !important; align-items:center !important; justify-content:center !important;
  cursor:pointer !important; opacity:0.65; transition:background 0.15s,opacity 0.15s !important;
  z-index:10 !important;
}
div[data-testid="stTextInput"] > div button:hover { background:rgba(56,189,248,0.15) !important; opacity:1 !important; }
div[data-testid="stTextInput"] > div button svg { width:16px !important; height:16px !important; stroke:#38bdf8 !important; color:#38bdf8 !important; }

/* ── Submit button ── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button[type="submit"] {
  background:linear-gradient(90deg,#0ea5e9 0%,#38bdf8 50%,#0284c7 100%) !important;
  color:#001524 !important; font-weight:800 !important; border-radius:13px !important;
  border:none !important; width:100% !important; font-size:0.96rem !important;
  padding:13px 0 !important; letter-spacing:0.07em !important; text-transform:uppercase !important;
  box-shadow:0 4px 20px rgba(14,165,233,0.4),0 2px 6px rgba(2,132,199,0.25) !important;
  transition:box-shadow 0.22s,transform 0.18s !important; margin-top:6px !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
div[data-testid="stForm"] button[type="submit"]:hover {
  box-shadow:0 8px 30px rgba(14,165,233,0.6),0 4px 14px rgba(56,189,248,0.35) !important;
  transform:translateY(-2px) !important;
}

/* ── Centered link row below form ── */
.auth-footer { display:flex; justify-content:center; align-items:center; gap:6px; margin-top:18px; }
.auth-footer-text { color:rgba(100,116,139,0.85); font-size:0.84rem; }

/* ── Inline link button ── */
div[data-testid="auth-link-btn"] { display:inline-block !important; margin:0 !important; padding:0 !important; }
div[data-testid="auth-link-btn"] > div { margin:0 !important; padding:0 !important; }
div[data-testid="auth-link-btn"] > div > button {
  background:transparent !important; border:none !important; padding:0 2px !important;
  color:#38bdf8 !important; font-size:0.84rem !important; font-weight:700 !important;
  text-decoration:underline !important; text-decoration-color:rgba(56,189,248,0.45) !important;
  text-underline-offset:3px !important; cursor:pointer !important; min-height:unset !important;
  box-shadow:none !important; width:auto !important; display:inline !important;
  transition:color 0.15s !important;
}
div[data-testid="auth-link-btn"] > div > button:hover { color:#7dd3fc !important; background:transparent !important; box-shadow:none !important; }

/* ── Animated hero pill typewriter ── */
@keyframes fadeSlideIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes pillPop { 0%{opacity:0;transform:scale(0.7) translateY(8px);} 80%{transform:scale(1.06) translateY(-1px);} 100%{opacity:1;transform:scale(1) translateY(0);} }
@keyframes glowPulse { 0%,100%{box-shadow:0 0 8px rgba(56,189,248,0.15);} 50%{box-shadow:0 0 20px rgba(56,189,248,0.4);} }
@keyframes shimmer { 0%{background-position:200% center;} 100%{background-position:-200% center;} }

.hero-badge { animation: fadeSlideIn 0.6s ease forwards; }
.hero-title { animation: fadeSlideIn 0.6s ease 0.25s both; }
.hero-subtitle { animation: fadeSlideIn 0.5s ease 0.4s both; }

.pill-1 { opacity:0; animation: pillPop 0.5s cubic-bezier(.34,1.56,.64,1) 0.65s forwards; }
.pill-2 { opacity:0; animation: pillPop 0.5s cubic-bezier(.34,1.56,.64,1) 0.85s forwards; }
.pill-3 { opacity:0; animation: pillPop 0.5s cubic-bezier(.34,1.56,.64,1) 1.05s forwards; }
.pill-4 { opacity:0; animation: pillPop 0.5s cubic-bezier(.34,1.56,.64,1) 1.25s forwards; }

.hero-title-text {
  background: linear-gradient(90deg,#38bdf8,#0ea5e9,#7dd3fc,#38bdf8);
  background-size: 300% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation: shimmer 4s linear 0.8s infinite;
}
</style>"""

_APP_THEME = """<style>
.stApp { background-color:#00060f; color:#cae6ff; }
h1 { background:linear-gradient(90deg,#38bdf8,#0ea5e9,#7dd3fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.2rem !important; font-weight:800 !important; }
h2,h3 { color:#38bdf8 !important; }
hr { border-color:rgba(56,189,248,0.18) !important; }
.section-card { background:linear-gradient(135deg,rgba(14,165,233,0.07) 0%,rgba(2,132,199,0.04) 100%); border:1px solid rgba(56,189,248,0.18); border-radius:16px; padding:24px 28px; margin-bottom:1rem; }
.stat-pill { display:inline-block; background:rgba(14,165,233,0.12); border:1px solid rgba(56,189,248,0.22); border-radius:999px; padding:6px 16px; margin-right:8px; font-size:0.85rem; color:#7dd3fc; }
.stat-pill b { color:#bae6fd; }
div[data-testid="stExpander"] { border:1px solid rgba(56,189,248,0.18) !important; border-radius:12px !important; }
.stButton > button { background:linear-gradient(135deg,rgba(14,165,233,0.12),rgba(56,189,248,0.08)) !important; border:1px solid rgba(56,189,248,0.35) !important; color:#38bdf8 !important; border-radius:10px !important; font-weight:600 !important; transition:box-shadow 0.2s,transform 0.2s !important; }
.stButton > button:hover { box-shadow:0 4px 16px rgba(14,165,233,0.3) !important; transform:translateY(-1px) !important; }
[data-testid="baseButton-primary"] > button,button[kind="primary"] { background:linear-gradient(90deg,#0ea5e9,#38bdf8,#0284c7) !important; color:#001524 !important; border:none !important; font-weight:700 !important; border-radius:10px !important; box-shadow:0 4px 16px rgba(14,165,233,0.35) !important; }
[data-testid="baseButton-primary"] > button:hover,button[kind="primary"]:hover { box-shadow:0 6px 24px rgba(14,165,233,0.55) !important; transform:translateY(-1px) !important; }
div[data-testid="stDataFrame"] { border:1px solid rgba(56,189,248,0.14) !important; border-radius:12px !important; }
div[data-testid="stFileUploader"] { border:1px dashed rgba(56,189,248,0.32) !important; border-radius:12px !important; background:rgba(14,165,233,0.04) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#38bdf8 !important; border-bottom-color:#38bdf8 !important; }
</style>"""

_SIGNOUT_CSS = """<style>
/* ── Outer wrapper: fixed top-right ── */
div[data-testid="so-trigger"] {
    position: fixed !important;
    top: 10px !important;
    right: 16px !important;
    z-index: 99999 !important;
    width: auto !important;
}
div[data-testid="so-trigger"] > div { margin:0 !important; padding:0 !important; }

/* ── The pill button itself ── */
div[data-testid="so-trigger"] > div > button {
    position: static !important;
    width: auto !important;
    height: auto !important;
    min-height: unset !important;
    opacity: 1 !important;
    cursor: pointer !important;
    border-radius: 999px !important;
    padding: 8px 18px 8px 12px !important;
    background: rgba(0, 10, 25, 0.88) !important;
    backdrop-filter: blur(18px) !important;
    border: 1.5px solid rgba(0, 245, 255, 0.40) !important;
    color: #00f5ff !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    box-shadow: 0 2px 20px rgba(0, 245, 255, 0.18), 0 1px 6px rgba(0,0,0,0.5) !important;
    white-space: nowrap !important;
    letter-spacing: 0.02em !important;
    transition: border-color 0.2s, box-shadow 0.2s, color 0.2s, background 0.2s !important;
}
div[data-testid="so-trigger"] > div > button:hover {
    border-color: rgba(239, 68, 68, 0.65) !important;
    box-shadow: 0 4px 24px rgba(239, 68, 68, 0.28), 0 2px 8px rgba(0,0,0,0.4) !important;
    color: #fca5a5 !important;
    background: rgba(0, 12, 30, 0.94) !important;
}
</style>"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.load(f, Loader=SafeLoader)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

def verify_password(plain, hashed):
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False

def hash_password(plain):
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _is_valid_email(email):
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", email))


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def render_login_page():
    """Always returns (None, None)."""
    st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
    )

    if st.session_state.get("auth_status") is True:
        return None, None

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    # Animated hero — single-line HTML strings only
    hero = (
        '<div style="text-align:center;margin-top:44px;margin-bottom:28px;">'

        # Badge
        '<div class="hero-badge" style="display:inline-flex;align-items:center;gap:8px;'
        'background:linear-gradient(135deg,rgba(14,165,233,0.14),rgba(56,189,248,0.08));'
        'border:1px solid rgba(56,189,248,0.28);border-radius:999px;'
        'padding:6px 18px;margin-bottom:14px;">'
        '<span style="font-size:0.85rem;">⚡</span>'
        '<span style="background:linear-gradient(90deg,#38bdf8,#7dd3fc);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'font-size:0.75rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;">'
        'Secure Analytics Workspace</span>'
        '</div><br>'

        # Title with shimmer
        '<span class="hero-title" style="display:inline-block;font-size:2.3rem;font-weight:900;line-height:1.1;margin-bottom:12px;">'
        '<span class="hero-title-text">🤖 Multi-Agent Data Analyzer</span>'
        '</span><br>'

        # Subtitle
        '<span class="hero-subtitle" style="display:inline-block;color:rgba(125,211,252,0.55);font-size:0.82rem;margin-bottom:14px;">'
        'Your intelligent workspace for data insights</span><br>'

        # Animated pills
        '<div style="display:inline-flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:6px;">'

        '<span class="pill-1" style="display:inline-flex;align-items:center;gap:5px;'
        'color:#7dd3fc;font-size:0.82rem;font-weight:600;'
        'background:rgba(14,165,233,0.12);border:1px solid rgba(56,189,248,0.25);'
        'border-radius:999px;padding:5px 14px;">🧹 Clean</span>'

        '<span class="pill-2" style="display:inline-flex;align-items:center;gap:5px;'
        'color:#7dd3fc;font-size:0.82rem;font-weight:600;'
        'background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);'
        'border-radius:999px;padding:5px 14px;">📊 Analyze</span>'

        '<span class="pill-3" style="display:inline-flex;align-items:center;gap:5px;'
        'color:#7dd3fc;font-size:0.82rem;font-weight:600;'
        'background:rgba(14,165,233,0.12);border:1px solid rgba(56,189,248,0.25);'
        'border-radius:999px;padding:5px 14px;">📈 Visualize</span>'

        '<span class="pill-4" style="display:inline-flex;align-items:center;gap:5px;'
        'color:#7dd3fc;font-size:0.82rem;font-weight:600;'
        'background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);'
        'border-radius:999px;padding:5px 14px;">🔮 Forecast</span>'

        '</div>'
        '</div>'
    )
    st.markdown(hero, unsafe_allow_html=True)
    return None, None


def check_authentication(authenticator=None, config=None):
    """Always returns (bool, str|None, str|None)."""

    if st.session_state.get("auth_status") is True:
        return True, st.session_state.get("auth_name"), st.session_state.get("auth_username")

    try:
        cfg   = load_config()
        users = cfg.get("credentials", {}).get("usernames", {})
    except Exception as e:
        st.error(f"Could not load config.yaml: {e}")
        return False, None, None

    if "auth_tab" not in st.session_state:
        st.session_state["auth_tab"] = "login"

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    if st.session_state["auth_tab"] == "login":

        with st.form("login_form", clear_on_submit=False):
            st.markdown(
                '<div style="margin-bottom:26px;text-align:center;">'
                '<div style="font-size:1.9rem;margin-bottom:8px;filter:drop-shadow(0 0 10px rgba(56,189,248,0.45));">🔐</div>'
                '<div style="color:#bae6fd;font-size:1.25rem;font-weight:800;letter-spacing:-0.01em;">Welcome back</div>'
                '<div style="color:rgba(125,211,252,0.5);font-size:0.84rem;margin-top:5px;">Sign in to your analytics workspace</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            username_input = st.text_input("Username", placeholder="Enter your username")
            password_input = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In  →", use_container_width=True)

        if submitted:
            if not username_input or not password_input:
                st.error("Please fill in all fields.")
            elif username_input not in users:
                st.error("Incorrect username or password.")
            else:
                user_data    = users[username_input]
                stored_hash  = user_data.get("password", "")
                display_name = user_data.get("first_name") or user_data.get("name") or username_input
                if verify_password(password_input, stored_hash):
                    st.session_state["auth_status"]   = True
                    st.session_state["auth_name"]     = display_name
                    st.session_state["auth_username"] = username_input
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        # Centered footer row — single centered div, no columns
        st.markdown(
            '<div class="auth-footer">'
            '<span class="auth-footer-text">Don\'t have an account?</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.markdown('<div data-testid="auth-link-btn">', unsafe_allow_html=True)
            if st.button("Register here", key="go_register", use_container_width=True):
                st.session_state["auth_tab"] = "register"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── REGISTER ──────────────────────────────────────────────────────────────
    else:
        with st.form("register_form", clear_on_submit=False):
            st.markdown(
                '<div style="margin-bottom:26px;text-align:center;">'
                '<div style="font-size:1.9rem;margin-bottom:8px;filter:drop-shadow(0 0 10px rgba(56,189,248,0.45));">✨</div>'
                '<div style="color:#bae6fd;font-size:1.25rem;font-weight:800;letter-spacing:-0.01em;">Create your account</div>'
                '<div style="color:rgba(125,211,252,0.5);font-size:0.84rem;margin-top:5px;">Join your analytics workspace</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            col_fn, col_ln = st.columns(2)
            with col_fn:
                first_name = st.text_input("First Name", placeholder="Albert")
            with col_ln:
                last_name = st.text_input("Last Name", placeholder="Benjamin")
            reg_email    = st.text_input("Email Address",    placeholder="you@example.com")
            reg_username = st.text_input("Username",          placeholder="Choose a username")
            reg_password = st.text_input("Password",          type="password", placeholder="Minimum 8 characters")
            reg_confirm  = st.text_input("Confirm Password",  type="password", placeholder="Repeat your password")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            reg_submitted = st.form_submit_button("Create Account  →", use_container_width=True)

        if reg_submitted:
            errors = []
            if not all([first_name, last_name, reg_email, reg_username, reg_password, reg_confirm]):
                errors.append("Please fill in all fields.")
            if reg_username and " " in reg_username:
                errors.append("Username cannot contain spaces.")
            if reg_username and reg_username in users:
                errors.append(f"Username '{reg_username}' is already taken.")
            if reg_email and not _is_valid_email(reg_email):
                errors.append("Please enter a valid email address.")
            if reg_password and len(reg_password) < 8:
                errors.append("Password must be at least 8 characters.")
            if reg_password and reg_confirm and reg_password != reg_confirm:
                errors.append("Passwords do not match.")
            if errors:
                for err in errors:
                    st.error(err)
            else:
                try:
                    cfg.setdefault("credentials", {}).setdefault("usernames", {})[reg_username] = {
                        "email": reg_email, "first_name": first_name, "last_name": last_name,
                        "password": hash_password(reg_password), "roles": ["user"],
                    }
                    save_config(cfg)
                    st.session_state["auth_status"]   = True
                    st.session_state["auth_name"]     = first_name
                    st.session_state["auth_username"] = reg_username
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save account: {e}")

        st.markdown(
            '<div class="auth-footer">'
            '<span class="auth-footer-text">Already have an account?</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _, mid2, _ = st.columns([1, 1, 1])
        with mid2:
            st.markdown('<div data-testid="auth-link-btn">', unsafe_allow_html=True)
            if st.button("Sign in here", key="go_login", use_container_width=True):
                st.session_state["auth_tab"] = "login"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    return False, None, None


def render_logout(authenticator=None, name: str = ""):
    """Injects global app theme + fixed top-right sign-out pill."""
    st.markdown(_APP_THEME, unsafe_allow_html=True)
    display  = name or st.session_state.get("auth_name", "User")
    st.markdown(_SIGNOUT_CSS, unsafe_allow_html=True)
    btn_key = f"so_btn_{st.session_state.get('auth_username', 'user')}"
    with st.container():
        st.markdown('<div data-testid="so-trigger">', unsafe_allow_html=True)
        if st.button(f"🟢 {display}  ⏻", key=btn_key, help="Sign out"):
            for k in ["auth_status", "auth_name", "auth_username"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)