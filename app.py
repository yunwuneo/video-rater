"""
Video Annotation WebUI — Taste Prediction Database (Digital Twin).
Review videos, inspect AI-generated metadata, rate them, and save preferences to PostgreSQL.
Run from the parent directory that contains both `videos/` and `analysis/` folders.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Optional

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
from dotenv import load_dotenv

# Load .env from directory containing app.py
_env_path = Path(__file__).resolve().parent / ".env"
_env_loaded = load_dotenv(_env_path)

# -----------------------------------------------------------------------------
# Database connection config (edit as needed, or set in .env)
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", "5432")),
    "dbname": os.environ.get("PGDATABASE", "video_rater"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", ""),
}

# Base path: parent of videos/ and analysis/ (default: current working directory)
BASE_PATH = Path(os.environ.get("VIDEO_RATER_BASE", ".")).resolve()
VIDEOS_DIR = BASE_PATH / "1"
ANALYSIS_DIR = BASE_PATH / "2"

# Cloud LLM config for extracting tag-like feature phrases (optional)
# When set, uses LLM to extract 标签化特征短语 from frame descriptions and summary.
_raw_url = (os.environ.get("LLM_APP_URL", "").strip() or "").rstrip("/")
# OpenRouter 正确 base URL 为 https://openrouter.ai/api/v1（需含 /api）
if _raw_url and "openrouter.ai" in _raw_url and "/api/" not in _raw_url:
    if _raw_url.endswith("/v1"):
        _raw_url = _raw_url[:-3] + "/api/v1"
    else:
        _raw_url = _raw_url.rstrip("/") + "/api/v1"
LLM_APP_URL = _raw_url or None
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip() or None
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "").strip() or "gpt-4o-mini"

# -----------------------------------------------------------------------------
# 身份认证（公网部署时启用）
# -----------------------------------------------------------------------------
AUTH_ENABLED = os.environ.get("VIDEO_RATER_AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes")
AUTH_ADMIN_USER = (os.environ.get("VIDEO_RATER_ADMIN_USER", "").strip() or "admin")
AUTH_ADMIN_PASSWORD = os.environ.get("VIDEO_RATER_ADMIN_PASSWORD", "").strip()

# OIDC（用于对接 Casdoor 等）
OIDC_ISSUER = (os.environ.get("VIDEO_RATER_OIDC_ISSUER", "").strip() or "").rstrip("/") or None
OIDC_CLIENT_ID = (os.environ.get("VIDEO_RATER_OIDC_CLIENT_ID", "").strip() or None)
OIDC_CLIENT_SECRET = (os.environ.get("VIDEO_RATER_OIDC_CLIENT_SECRET", "").strip() or None)
OIDC_REDIRECT_URI = (os.environ.get("VIDEO_RATER_OIDC_REDIRECT_URI", "").strip() or None)
OIDC_SCOPE = (os.environ.get("VIDEO_RATER_OIDC_SCOPE", "").strip() or "openid profile email")
OIDC_ENABLED = bool(
    AUTH_ENABLED
    and OIDC_ISSUER
    and OIDC_CLIENT_ID
    and OIDC_REDIRECT_URI
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    """Create and cache a single DB connection for the session."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def ensure_table(conn):
    """Create video_preferences table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_preferences (
                id SERIAL PRIMARY KEY,
                video_path VARCHAR(512) UNIQUE NOT NULL,
                json_path VARCHAR(512) NOT NULL,
                overall_score NUMERIC(4, 2) NOT NULL,
                liked_features JSONB,
                raw_analysis JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()


# -----------------------------------------------------------------------------
# 身份认证：用户表与密码校验（直接使用 bcrypt，避免 passlib 与新版 bcrypt 不兼容）
# -----------------------------------------------------------------------------
def _bcrypt_secret(password: str) -> bytes:
    """Encode password for bcrypt; bcrypt 仅支持最多 72 字节，超出部分截断。"""
    raw = password.encode("utf-8")
    return raw[:72] if len(raw) > 72 else raw


def _hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(_bcrypt_secret(password), bcrypt.gensalt()).decode("ascii")


def _verify_password(password: str, password_hash: str) -> bool:
    import bcrypt
    try:
        return bcrypt.checkpw(_bcrypt_secret(password), password_hash.encode("ascii"))
    except Exception:
        return False


def ensure_users_table(conn):
    """Create users table for auth if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_rater_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(128) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()


def ensure_auth_admin(conn):
    """If no users exist and admin password is set in env, create initial admin user."""
    if not AUTH_ADMIN_PASSWORD:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM video_rater_users")
        if cur.fetchone()["n"] > 0:
            return
        cur.execute(
            "INSERT INTO video_rater_users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING",
            (AUTH_ADMIN_USER, _hash_password(AUTH_ADMIN_PASSWORD)),
        )
        conn.commit()
    logger.info("Created initial admin user: %s", AUTH_ADMIN_USER)


def get_user_by_username(conn, username: str) -> Optional[dict]:
    """Return user row (with password_hash) or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash FROM video_rater_users WHERE username = %s",
            (username.strip(),),
        )
        return cur.fetchone()


def verify_login(conn, username: str, password: str) -> bool:
    """Verify credentials; return True if valid."""
    user = get_user_by_username(conn, username)
    if not user or not password:
        return False
    return _verify_password(password, user["password_hash"])


def update_username(conn, current_username: str, new_username: str, current_password: str) -> tuple[bool, str]:
    """Update username; verify current password first. Returns (success, error_message)."""
    user = get_user_by_username(conn, current_username)
    if not user or not _verify_password(current_password, user["password_hash"]):
        return False, "当前密码错误。"
    new_username = new_username.strip()
    if not new_username:
        return False, "新用户名不能为空。"
    if new_username == current_username:
        return False, "新用户名与当前相同。"
    existing = get_user_by_username(conn, new_username)
    if existing is not None:
        return False, "该用户名已被使用。"
    with conn.cursor() as cur:
        cur.execute("UPDATE video_rater_users SET username = %s WHERE id = %s", (new_username, user["id"]))
        conn.commit()
    return True, ""


def update_password(conn, username: str, current_password: str, new_password: str) -> tuple[bool, str]:
    """Update password; verify current password first. Returns (success, error_message)."""
    user = get_user_by_username(conn, username)
    if not user or not _verify_password(current_password, user["password_hash"]):
        return False, "当前密码错误。"
    if not new_password or len(new_password) < 6:
        return False, "新密码至少 6 位。"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE video_rater_users SET password_hash = %s WHERE id = %s",
            (_hash_password(new_password), user["id"]),
        )
        conn.commit()
    return True, ""


# -----------------------------------------------------------------------------
# OIDC（Casdoor 等）发现、授权 URL、code 交换与 userinfo
# state 存服务端，避免从 IdP 跳回时 session 丢失导致校验失败
# -----------------------------------------------------------------------------
_OIDC_STATE_TTL_SEC = 600  # 10 分钟


@st.cache_resource
def _oidc_state_store() -> tuple[dict[str, float], threading.Lock]:
    """
    跨会话共享的 OIDC state 存储。
    使用 cache_resource 避免新会话回调时拿不到此前登记的 state。
    """
    return {}, threading.Lock()


def _oidc_register_state(state: str) -> None:
    """生成授权 URL 前将 state 登记到服务端存储。"""
    pending, lock = _oidc_state_store()
    now = time.time()
    with lock:
        # 顺带清理过期 state，避免常驻进程内存累积
        expired = [k for k, ts in pending.items() if now - ts > _OIDC_STATE_TTL_SEC]
        for k in expired:
            pending.pop(k, None)
        pending[state] = now


def _oidc_consume_state(state: str) -> bool:
    """回调时校验并消费 state：存在且未过期返回 True 并删除，否则返回 False。"""
    pending, lock = _oidc_state_store()
    with lock:
        created = pending.pop(state, None)
    if created is None:
        return False
    if time.time() - created > _OIDC_STATE_TTL_SEC:
        return False
    return True


@st.cache_data(ttl=3600)
def _oidc_discovery(issuer: str) -> Optional[dict]:
    """获取 OIDC 发现文档；失败返回 None。"""
    url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning("OIDC discovery failed for %s: %s", url, e)
        return None


def _oidc_auth_url(state: str) -> Optional[str]:
    """构造 OIDC 授权 URL；若 discovery 失败返回 None。"""
    if not OIDC_ISSUER or not OIDC_CLIENT_ID or not OIDC_REDIRECT_URI:
        return None
    doc = _oidc_discovery(OIDC_ISSUER)
    if not doc:
        return None
    auth_endpoint = doc.get("authorization_endpoint")
    if not auth_endpoint:
        return None
    from urllib.parse import urlencode
    params = {
        "response_type": "code",
        "client_id": OIDC_CLIENT_ID,
        "redirect_uri": OIDC_REDIRECT_URI,
        "scope": OIDC_SCOPE,
        "state": state,
    }
    return f"{auth_endpoint}?{urlencode(params)}"


def _oidc_exchange_and_userinfo(code: str) -> tuple[Optional[str], Optional[str]]:
    """
    用 code 换取 token，再拉取 userinfo，返回 (username, error_message)。
    成功时 error_message 为 None；失败时 username 为 None。
    """
    if not OIDC_ISSUER or not OIDC_CLIENT_ID or not OIDC_REDIRECT_URI:
        return None, "OIDC 未配置完整"
    doc = _oidc_discovery(OIDC_ISSUER)
    if not doc:
        return None, "无法获取 OIDC 发现文档"
    token_endpoint = doc.get("token_endpoint")
    userinfo_endpoint = doc.get("userinfo_endpoint")
    if not token_endpoint or not userinfo_endpoint:
        return None, "发现文档缺少 token_endpoint 或 userinfo_endpoint"
    try:
        with httpx.Client(timeout=15.0) as client:
            # 使用 application/x-www-form-urlencoded 交换 code
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": OIDC_REDIRECT_URI,
                "client_id": OIDC_CLIENT_ID,
            }
            if OIDC_CLIENT_SECRET:
                token_data["client_secret"] = OIDC_CLIENT_SECRET
            r = client.post(
                token_endpoint,
                data=token_data,
                headers={"Accept": "application/json"},
            )
            r.raise_for_status()
            token_json = r.json()
            access_token = token_json.get("access_token")
            if not access_token:
                return None, "未返回 access_token"
            # 拉取 userinfo
            u = client.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            u.raise_for_status()
            userinfo = u.json()
            # 优先 preferred_username，其次 sub、name
            username = (
                userinfo.get("preferred_username")
                or userinfo.get("sub")
                or userinfo.get("name")
                or userinfo.get("email")
            )
            if not username:
                username = str(userinfo.get("sub", "oidc_user"))
            return str(username).strip(), None
    except httpx.HTTPStatusError as e:
        return None, f"OIDC 请求失败: {e.response.status_code}"
    except Exception as e:
        logger.warning("OIDC exchange/userinfo failed: %s", e)
        return None, str(e)


def render_login_page(conn):
    """Show login form; on success set session and rerun. Caller should return after this."""
    st.set_page_config(page_title="登录 — Video Rater", layout="centered")
    st.title("Video Rater — 登录")
    st.caption("请使用账号密码登录后使用标注功能。")

    # OIDC 回调：URL 带 code 和 state 时，用服务端存储校验 state（不依赖 session）
    q = st.query_params
    if OIDC_ENABLED and q.get("code") and q.get("state"):
        incoming_state = q.get("state", "")
        if _oidc_consume_state(incoming_state):
            code = q.get("code", "")
            username, err = _oidc_exchange_and_userinfo(code)
            if username and not err:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["auth_provider"] = "oidc"
                # 清除 URL 中的 code/state，避免刷新时重复使用
                for k in ("code", "state"):
                    if k in st.query_params:
                        del st.query_params[k]
                st.rerun()
            else:
                st.error(f"OIDC 登录失败: {err or '未知错误'}")
        else:
            st.error("OIDC state 不匹配或已过期，请重新点击「使用 Casdoor 登录」。")

    with st.form("login_form"):
        username = st.text_input("用户名", key="login_username", autocomplete="username")
        password = st.text_input("密码", type="password", key="login_password", autocomplete="current-password")
        submitted = st.form_submit_button("登录")
    if submitted:
        if not username or not password:
            st.error("请输入用户名和密码。")
            return
        if verify_login(conn, username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username.strip()
            st.rerun()
        else:
            st.error("用户名或密码错误。")

    if OIDC_ENABLED:
        import secrets
        state = secrets.token_urlsafe(32)
        _oidc_register_state(state)
        auth_url = _oidc_auth_url(state)
        if auth_url:
            from urllib.parse import urlparse
            redirect_path = (urlparse(OIDC_REDIRECT_URI).path or "/").rstrip("/") or "/"
            st.divider()
            st.caption("或使用 Casdoor (OIDC) 登录")
            if redirect_path != "/":
                st.warning("当前 OIDC 回调地址不是根路径 `/`，如出现回调异常请改为应用首页地址（例如 http://localhost:8501/）。")
            st.link_button("使用 Casdoor 登录", url=auth_url, type="secondary")
        else:
            st.caption("OIDC 未就绪（请检查 VIDEO_RATER_OIDC_ISSUER 与发现文档）。")


def render_account_sidebar(conn):
    """Show current user, change username/password, and logout in sidebar."""
    if not st.session_state.get("authenticated"):
        return
    username = st.session_state.get("username", "")
    with st.sidebar:
        st.caption(f"已登录: **{username}**")
        with st.expander("修改账号 / 密码"):
            # 修改用户名
            st.caption("修改用户名")
            with st.form("change_username_form", clear_on_submit=True):
                new_username = st.text_input("新用户名", key="acct_new_username", autocomplete="username")
                current_pw_for_name = st.text_input("当前密码（验证身份）", type="password", key="acct_pw_for_name")
                if st.form_submit_button("保存用户名"):
                    if new_username and current_pw_for_name:
                        ok, err = update_username(conn, username, new_username, current_pw_for_name)
                        if ok:
                            st.session_state["username"] = new_username.strip()
                            st.success("用户名已更新。")
                            st.rerun()
                        else:
                            st.error(err)
                    else:
                        st.warning("请填写新用户名和当前密码。")
            st.divider()
            # 修改密码
            st.caption("修改密码")
            with st.form("change_password_form", clear_on_submit=True):
                current_pw = st.text_input("当前密码", type="password", key="acct_current_pw")
                new_pw = st.text_input("新密码（至少 6 位）", type="password", key="acct_new_pw")
                new_pw_confirm = st.text_input("确认新密码", type="password", key="acct_new_pw_confirm")
                if st.form_submit_button("保存密码"):
                    if current_pw and new_pw and new_pw_confirm:
                        if new_pw != new_pw_confirm:
                            st.error("两次输入的新密码不一致。")
                        else:
                            ok, err = update_password(conn, username, current_pw, new_pw)
                            if ok:
                                st.success("密码已更新。")
                            else:
                                st.error(err)
                    else:
                        st.warning("请填写当前密码、新密码并确认。")
        if st.button("退出登录", key="logout_btn"):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()


def get_rated_video_paths(conn):
    """Return set of video_path values that are already rated."""
    with conn.cursor() as cur:
        cur.execute("SELECT video_path FROM video_preferences")
        return {row["video_path"] for row in cur.fetchall()}


def insert_preference(conn, video_path: str, json_path: str, overall_score: float, liked_features: list, raw_analysis: dict):
    """Insert one annotation row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_preferences (video_path, json_path, overall_score, liked_features, raw_analysis)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_path) DO NOTHING
            """,
            (video_path, json_path, overall_score, json.dumps(liked_features), json.dumps(raw_analysis)),
        )
        conn.commit()


def count_rated(conn) -> int:
    """Return total number of rated videos."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM video_preferences")
        return cur.fetchone()["n"]


# -----------------------------------------------------------------------------
# Data loading & matching
# -----------------------------------------------------------------------------
def collect_video_analysis_pairs():
    """
    Traverse videos/ for all .mp4, build analysis path, skip if JSON missing.
    Returns list of (video_path, json_path) as Path objects (relative to BASE_PATH for DB storage).
    """
    pairs = []
    if not VIDEOS_DIR.is_dir():
        logger.warning("Videos directory not found: %s", VIDEOS_DIR)
        return pairs

    for mp4_path in VIDEOS_DIR.rglob("*.mp4"):
        # relative: e.g. example1/1.mp4
        rel = mp4_path.relative_to(VIDEOS_DIR)
        subfolder = rel.parent
        stem = rel.stem  # 1
        json_name = f"{stem}_analysis.json"
        json_path = ANALYSIS_DIR / subfolder / json_name

        if not json_path.is_file():
            logger.warning("Analysis file missing for video %s, skipping: %s", mp4_path, json_path)
            continue
        # Store as forward-slash paths for DB (portable)
        video_path_str = str(rel).replace("\\", "/")
        json_path_str = str(json_path.relative_to(BASE_PATH)).replace("\\", "/")
        pairs.append((video_path_str, json_path_str, mp4_path, json_path))

    return pairs


def load_analysis(json_path: Path) -> Optional[dict]:
    """Load and parse analysis JSON; return None on error."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", json_path, e)
        return None


def _collect_descriptive_text(obj, texts: list[str]) -> None:
    """
    Recursively collect descriptive text from analysis JSON.
    Targets: 帧描述、总结、summary、description、frame descriptions, video_description, etc.
    """
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 5 and len(s) <= 8000:  # 5–8000 字，过短为噪音、过长截断后送 LLM
            texts.append(s)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect_descriptive_text(v, texts)
    elif isinstance(obj, list):
        for item in obj:
            _collect_descriptive_text(item, texts)


def collect_descriptive_text(analysis: dict) -> str:
    """Gather frame descriptions, summary, and other descriptive text from analysis JSON."""
    texts = []
    _collect_descriptive_text(analysis, texts)
    combined = "\n\n".join(dict.fromkeys(texts))
    return combined.strip()


def _resp_to_debug_str(resp) -> str:
    """将 API 响应转为可读字符串以便调试。"""
    try:
        if hasattr(resp, "model_dump"):
            return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
        if hasattr(resp, "__dict__"):
            return json.dumps(
                {k: str(v)[:200] for k, v in vars(resp).items()},
                ensure_ascii=False,
                indent=2,
            )
        return str(resp)
    except Exception:
        return repr(resp)


_PLACEHOLDER_PATTERN = re.compile(
    r"^短语\d+$|^实际特征\d+$|^占位|^placeholder", re.I
)


def _parse_llm_features(content: str) -> list[str]:
    """
    从 LLM 返回文本解析特征短语。支持：① JSON {"features": [...]}
    ② 逗号分隔；③ 从长文本中提取 2-6 字中文短语（如 reasoning 回退）。
    """
    def _valid(p: str) -> bool:
        if 2 <= len(p) <= 12 and not _PLACEHOLDER_PATTERN.search(p):
            return True
        return False

    content = content.strip()
    if not content:
        return []
    # 1. 尝试 JSON {"features": ["a","b",...]}
    m = re.search(r'"features"\s*:\s*\[(.*?)\]', content, re.DOTALL)
    if m:
        inner = m.group(1)
        parts = re.findall(r'"([^"]{2,12})"', inner)
        if parts:
            return [p for p in parts if _valid(p)]
        try:
            obj = json.loads("{" + content[content.find('"features"'):] + "}")
            arr = obj.get("features", [])
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if _valid(str(x).strip())]
        except (json.JSONDecodeError, ValueError):
            pass
    # 2. 逗号/顿号分隔
    if len(content) < 500:
        parts = re.split(r"[,，、\s]+", content)
        out = [p.strip() for p in parts if _valid(p.strip())]
        if out:
            return out
    # 3. 从长文本提取中文短语（含 reasoning 中引号内 "xxx"、「xxx」）
    phrases = re.findall(r'"([\u4e00-\u9fff]{2,6})"', content)
    phrases.extend(re.findall(r'[「\u300c]([\u4e00-\u9fff]{2,6})[」\u300d]', content))
    phrases.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", content))
    seen = set()
    result = []
    for p in phrases:
        p = p.strip()
        if 2 <= len(p) <= 6 and p not in seen and _valid(p):
            seen.add(p)
            result.append(p)
    return result[:40]


def _store_llm_debug(tag: str, value: str) -> None:
    """存储 LLM 调试信息到 session_state，供 Debug 面板展示。"""
    if "session_state" in dir(st):
        st.session_state["_llm_last_raw"] = {"tag": tag, "value": value[:6000]}


def _call_llm_extract(text: str) -> tuple[list[str], str | None]:
    """
    Call cloud LLM to extract 标签化的特征短语. Returns (features, error_msg).
    """
    if not text:
        return [], "无描述文本"
    if not LLM_APP_URL or not LLM_API_KEY:
        return [], f"LLM 未配置: APP_URL={'已设置' if LLM_APP_URL else '未设置'}, API_KEY={'已设置' if LLM_API_KEY else '未设置'}"
    try:
        from openai import OpenAI

        logger.info("LLM 调用开始: url=%s model=%s text_len=%d", LLM_APP_URL, LLM_MODEL_NAME, len(text))
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_APP_URL)
        prompt = f"""从以下视频分析文本（包含帧描述、总结等）中，提炼出标签化的特征短语。
要求：
- 每个短语 2-6 个字，便于用户选择
- 示例：活泼、短裙、室外、舞蹈、复古、暖色调、慢节奏、特写镜头
- 只输出 JSON：{{"features": ["实际特征1", "实际特征2", ...]}}，用你提炼的真实特征词填充

文本：
{text[:6000]}
"""
        # stream=False 确保拿到完整响应；推理模型会先思考再输出 content
        # max_tokens 不足时 finish_reason=length，content 为空；需增大或限制 reasoning
        create_kw = {
            "model": LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False,
        }
        if "openrouter" in (LLM_APP_URL or "").lower():
            create_kw["extra_body"] = {"reasoning": {"max_tokens": 256}}
        resp = client.chat.completions.create(**create_kw)
        if isinstance(resp, str):
            logger.info("LLM 原始返回(字符串): %s", resp[:500])
            _store_llm_debug("api_response", resp[:2000])
            return [], f"API 返回了字符串: {resp[:200]}…"
        if not hasattr(resp, "choices") or not resp.choices:
            raw = _resp_to_debug_str(resp)
            logger.info("LLM 原始返回(无choices): %s", raw[:500])
            _store_llm_debug("api_response", raw[:2000])
            return [], f"响应无 choices: {raw[:200]}"
        msg = resp.choices[0].message
        raw_content = getattr(msg, "content", None)

        if isinstance(raw_content, list):
            content = " ".join(
                p.get("text", str(p)) for p in raw_content if isinstance(p, dict)
            ).strip()
        elif raw_content is not None:
            content = str(raw_content).strip()
        else:
            content = ""
        if not content and hasattr(msg, "model_dump"):
            d = msg.model_dump()
            c = d.get("content") or d.get("text")
            if isinstance(c, list):
                content = " ".join(x.get("text", str(x)) for x in c if isinstance(x, dict)).strip()
            else:
                content = str(c or "").strip()

        raw_resp = _resp_to_debug_str(resp)
        _store_llm_debug("api_response", raw_resp)
        logger.info("LLM 返回: content_len=%d", len(content))
        if not content:
            logger.info("LLM 返回空内容: %s", raw_resp[:500])
            return [], "LLM 返回空内容"

        features = _parse_llm_features(content)
        if not features:
            return [], "LLM 仅返回占位符或无效内容，已回退本地"
        return features, None
    except Exception as e:
        import traceback
        err = str(e)
        tb = traceback.format_exc()
        logger.warning("LLM feature extraction failed: %s\n%s", err, tb)
        if "session_state" in dir(st):
            st.session_state["_llm_last_traceback"] = tb
            st.session_state["_llm_last_error"] = err
        return [], f"LLM 调用异常: {err}"


def extract_features_locally(text: str) -> list[str]:
    """
    Local fallback: extract short Chinese phrases (2-4 chars) that look like feature tags.
    """
    if not text or len(text) < 10:
        return []
    matches = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    seen = set()
    result = []
    for m in matches:
        if 2 <= len(m) <= 6 and m not in seen:
            seen.add(m)
            result.append(m)
    return result[:30]


@st.cache_data(ttl=3600)
def _extract_features_cached(combined_text: str, use_llm: bool) -> tuple[list[str], str]:
    """
    Cached extraction. Returns (options, debug_status).
    """
    if use_llm:
        options, err = _call_llm_extract(combined_text)
        if err:
            options = extract_features_locally(combined_text)
            status = f"LLM 失败({err})，已回退本地提取，得到 {len(options)} 个"
        else:
            status = f"LLM 成功，得到 {len(options)} 个"
    else:
        options = extract_features_locally(combined_text)
        status = f"本地提取，得到 {len(options)} 个（未配置 LLM）"
    return sorted(set(options)), status


def extract_feature_options(analysis: dict, debug_out: dict | None = None) -> list[str]:
    """
    Extract tag-like feature phrases from frame descriptions and summary text.
    If debug_out dict is provided, fills it with debug info for UI display.
    """
    combined_text = collect_descriptive_text(analysis)
    use_llm = bool(LLM_APP_URL and LLM_API_KEY)

    if debug_out is not None:
        debug_out["llm_configured"] = use_llm
        debug_out["llm_url"] = (LLM_APP_URL or "")[:50] + ("..." if (LLM_APP_URL or "")[50:] else "")
        debug_out["llm_model"] = LLM_MODEL_NAME
        debug_out["text_len"] = len(combined_text)
        debug_out["text_preview"] = (combined_text[:200] + "…") if len(combined_text) > 200 else combined_text

    if not combined_text:
        if debug_out is not None:
            debug_out["status"] = "未收集到描述文本（analysis.json 中无 5–2000 字的字符串）"
            debug_out["text_preview"] = ""
        return []

    options, status = _extract_features_cached(combined_text, use_llm)
    if debug_out is not None:
        debug_out["status"] = status
    return options


# 模块级缓存，供后台线程写入，避免在非主线程访问 session_state
_feature_cache: dict[str, list[str]] = {}
_feature_loading: set[str] = set()
_feature_cache_lock = threading.Lock()


def _load_features_background(video_path_str: str, json_path: Path) -> None:
    """在后台线程中加载特征并写入模块级缓存（session_state 由 fragment 同步）"""
    try:
        analysis = load_analysis(json_path)
        if analysis:
            options = extract_feature_options(analysis, debug_out=None)
            with _feature_cache_lock:
                _feature_cache[video_path_str] = options
    finally:
        with _feature_cache_lock:
            _feature_loading.discard(video_path_str)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    # 身份认证：公网部署时在入口处校验
    if AUTH_ENABLED:
        conn_auth = get_db_connection()
        ensure_users_table(conn_auth)
        ensure_auth_admin(conn_auth)
        if not st.session_state.get("authenticated"):
            render_login_page(conn_auth)
            return

    st.set_page_config(page_title="Video Rater — Taste DB", layout="wide")
    conn = get_db_connection()
    if AUTH_ENABLED:
        render_account_sidebar(conn)
    # 视频播放器适配视口：PC/手机均无需上下滚动即可看到完整内容
    st.markdown(
        """
        <style>
        /* 视频播放器适配视口：PC/手机均无需上下滚动即可看到完整内容 */
        [data-testid="stVideo"],
        [data-testid="stVideo"] video,
        .stVideo,
        .stVideo video {
            max-height: calc(100vh - 180px) !important;
            width: 100% !important;
            object-fit: contain !important;
        }
        /* 确保视频元素正确缩放 */
        [data-testid="stVideo"] video,
        .stVideo video {
            display: block !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Video Annotation — Taste Prediction Database")

    ensure_table(conn)

    pairs = collect_video_analysis_pairs()
    rated_paths = get_rated_video_paths(conn)
    # (video_path_str, json_path_str, mp4_path, json_path)
    unrated = [
        (vp, jp, mp, jpath)
        for vp, jp, mp, jpath in pairs
        if vp not in rated_paths
    ]

    total_videos = len(pairs)
    num_rated = count_rated(conn)
    num_remaining = len(unrated)

    st.caption(f"Progress: **{num_rated}** rated · **{num_remaining}** remaining (of {total_videos} videos with analysis)")

    if not unrated:
        st.info("No unrated videos. All videos with valid analysis have been rated.")
        st.stop()

    # Use session state to track current index (stable across reruns after submit)
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    idx = min(st.session_state.current_index, len(unrated) - 1)
    video_path_str, json_path_str, mp4_path, json_path = unrated[idx]

    analysis = load_analysis(json_path)
    if analysis is None:
        st.error(f"Could not load analysis for: {json_path_str}")
        st.stop()

    # Full path for st.video (Streamlit needs filesystem path)
    video_full_path = str(mp4_path)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Video")
        if Path(video_full_path).is_file():
            st.video(video_full_path, autoplay=True, loop=True, muted=False)
        else:
            st.warning("Video file not found.")

        # 切换视频按钮：在 LLM 加载标签期间也可快速切换
        col_prev, _, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← 上一视频", key="btn_prev_video", disabled=(idx <= 0)):
                st.session_state.current_index = max(0, idx - 1)
                st.rerun()
        with col_next:
            if st.button("下一视频 →", key="btn_next_video", disabled=(idx >= len(unrated) - 1)):
                st.session_state.current_index = min(len(unrated) - 1, idx + 1)
                st.rerun()

    with right:
        st.subheader("Metadata & Annotation")

        tags = analysis.get("tags") or []
        if tags:
            st.markdown("**Tags**")
            st.write(", ".join(str(t) for t in tags[:20]))

        color_palette = analysis.get("color_palette")
        if color_palette:
            st.markdown("**Color palette**")
            st.write(color_palette)

        # Any other top-level keys we didn't display
        shown = {"summary", "description", "video_description", "tags", "color_palette"}
        for k, v in analysis.items():
            if k not in shown and v is not None and not isinstance(v, (dict, list)):
                st.markdown(f"**{k}**")
                st.write(v)

        st.divider()
        st.markdown("**Annotations**")

        overall_score = st.slider(
            "Overall Score",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key="overall_score",
        )

        session_key = f"liked_features_{video_path_str}"
        if session_key not in st.session_state:
            st.session_state[session_key] = []

        @st.fragment(run_every=timedelta(seconds=2))
        def feature_section():
            """标签区域：优先从 session_state/缓存读取，未命中时后台加载，加载期间可切换视频"""
            # 优先用 session_state，避免 fragment-only rerun 时闭包/模块缓存不同步
            if "feature_options_cache" not in st.session_state:
                st.session_state.feature_options_cache = {}
            feature_options = st.session_state.feature_options_cache.get(video_path_str)
            is_loading = False

            if not feature_options:
                with _feature_cache_lock:
                    feature_options = _feature_cache.get(video_path_str)
                    is_loading = video_path_str in _feature_loading
                if feature_options:
                    st.session_state.feature_options_cache[video_path_str] = feature_options

            if not feature_options and not is_loading:
                with _feature_cache_lock:
                    _feature_loading.add(video_path_str)
                t = threading.Thread(
                    target=_load_features_background,
                    args=(video_path_str, json_path),
                    daemon=True,
                )
                t.start()
                is_loading = True

            if is_loading and not feature_options:
                st.info("标签加载中… 可先使用左侧按钮切换视频。")
                return

            if feature_options:
                llm_debug = {}
                extract_feature_options(analysis, debug_out=llm_debug)
                if "llm_debug" not in st.session_state:
                    st.session_state.llm_debug = {}
                st.session_state.llm_debug[video_path_str] = llm_debug

                liked_features = st.session_state[session_key]
                st.markdown("**Which specific features did you particularly like?**")
                cols_per_row = 6
                for i in range(0, len(feature_options), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(feature_options):
                            feat = feature_options[i + j]
                            with col:
                                is_selected = feat in liked_features
                                if st.button(
                                    feat,
                                    type="primary" if is_selected else "secondary",
                                    key=f"feat_{idx}_{i}_{j}",
                                ):
                                    if feat in liked_features:
                                        liked_features.remove(feat)
                                    else:
                                        liked_features.append(feat)
                                    st.rerun()
                st.caption(
                    "选项由帧描述、总结文本中提取的特征短语组成。点击按钮可多选。"
                    + (" 已启用云端 LLM 提取。" if (LLM_APP_URL and LLM_API_KEY) else " 使用本地规则提取。配置 LLM_APP_URL、LLM_API_KEY 可启用云端提取。")
                )
            else:
                st.session_state[session_key] = []
                st.caption("未能从帧描述和总结文本中提取到特征短语。可配置 LLM_APP_URL、LLM_API_KEY 使用云端模型提取。")

        llm_debug = st.session_state.get("llm_debug", {}).get(video_path_str, {})
        feature_section()
        liked_features = st.session_state[session_key]

        # Summary 置于标签区域下方，默认折叠
        vd = analysis.get("video_description")
        summary = (
            analysis.get("summary")
            or analysis.get("description")
            or (vd.get("response") if isinstance(vd, dict) else None)
            or "—"
        )
        with st.expander("📄 Summary", expanded=False):
            st.text(summary[:500] + ("…" if len(str(summary)) > 500 else ""))

        with st.expander("🔧 LLM Debug", expanded=False):
            st.write("**配置**")
            st.code(
                f".env 文件: {_env_path} ({'已加载' if _env_loaded else '未找到或为空'})\n"
                f"LLM_APP_URL: {LLM_APP_URL or '(未设置)'}\n"
                f"LLM_API_KEY: {'已设置' if LLM_API_KEY else '(未设置)'}\n"
                f"LLM_MODEL_NAME: {LLM_MODEL_NAME}"
            )
            if llm_debug:
                st.write("**本次提取**")
                st.write(f"- 收集文本长度: {llm_debug.get('text_len', 0)}")
                st.write(f"- 状态: {llm_debug.get('status', '-')}")
                if llm_debug.get("text_preview"):
                    st.write("**收集到的文本预览**")
                    st.text(llm_debug["text_preview"])
                if llm_debug.get("llm_resp_type"):
                    st.write("**API 响应**")
                    st.code(f"类型: {llm_debug.get('llm_resp_type')}\n{llm_debug.get('llm_resp_repr', '')}")
                if llm_debug.get("llm_raw"):
                    st.write("**原始返回值**")
                    st.text(llm_debug["llm_raw"])
            else:
                st.write("（当前视频无特征提取记录）")
            if "_llm_last_traceback" in st.session_state:
                st.write("**最近一次 LLM 异常堆栈**")
                st.text(st.session_state["_llm_last_traceback"])
            if "_llm_last_raw" in st.session_state:
                r = st.session_state["_llm_last_raw"]
                st.write("**LLM API 返回信息**（每次提取都会更新）")
                st.code(r.get("value", ""))
            if LLM_APP_URL and LLM_API_KEY:
                if st.button("🧪 测试 API 连接", key="llm_test_btn"):
                    try:
                        url = f"{LLM_APP_URL.rstrip('/')}/chat/completions"
                        body = json.dumps({
                            "model": LLM_MODEL_NAME,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 5,
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            url,
                            data=body,
                            headers={
                                "Authorization": f"Bearer {LLM_API_KEY}",
                                "Content-Type": "application/json",
                            },
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=15) as r:
                            raw = r.read().decode("utf-8")
                        st.session_state["_llm_test_result"] = raw[:1500]
                    except Exception as e:
                        st.session_state["_llm_test_result"] = f"错误: {e}"
                    st.rerun()
                if "_llm_test_result" in st.session_state:
                    st.write("**API 测试原始返回**")
                    st.text(st.session_state["_llm_test_result"])
            st.caption("💡 修改 .env 后需重启应用才能生效；LLM 调用日志会输出到终端")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Submit & Next", type="primary"):
                insert_preference(
                    conn,
                    video_path=video_path_str,
                    json_path=json_path_str,
                    overall_score=overall_score,
                    liked_features=liked_features,
                    raw_analysis=analysis,
                )
                st.session_state.current_index = idx + 1
                st.rerun()
        with col_btn2:
            if st.button("直接跳过", help="不分析当前视频，跳到下一个；该视频之后还会再出现"):
                st.session_state.current_index = idx + 1
                st.rerun()

    # Optional: show current position in queue
    st.caption(f"Current: video {idx + 1} of {len(unrated)} unrated.")


if __name__ == "__main__":
    import os
    import subprocess
    import sys

    # 由我们拉起的 streamlit 子进程会带上此环境变量；只有这时才执行 main()，否则会再次 spawn 导致无限开浏览器。
    if os.environ.get("VIDEO_RATER_STREAMLIT") == "1":
        main()
    else:
        # 用 python app.py 时在子进程中启动 streamlit run，并标记为“我们启动的”，避免子进程里再次 spawn。
        script_path = str(Path(__file__).resolve())
        env = os.environ.copy()
        env["VIDEO_RATER_STREAMLIT"] = "1"
        sys.exit(
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", script_path, *sys.argv[1:]],
                env=env,
                check=False,
            ).returncode
        )
