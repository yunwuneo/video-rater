"""
Video Annotation WebUI — Taste Prediction Database (Digital Twin).
Review videos, inspect AI-generated metadata, rate them, and save preferences to PostgreSQL.
Run from the parent directory that contains both `videos/` and `analysis/` folders.
"""

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

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


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Video Rater — Taste DB", layout="wide")
    st.title("Video Annotation — Taste Prediction Database")

    conn = get_db_connection()
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
            st.video(video_full_path)
        else:
            st.warning("Video file not found.")

    with right:
        st.subheader("Metadata & Annotation")

        # Metadata from JSON (safe .get); summary may be in video_description.response
        st.markdown("**Summary**")
        vd = analysis.get("video_description")
        summary = (
            analysis.get("summary")
            or analysis.get("description")
            or (vd.get("response") if isinstance(vd, dict) else None)
            or "—"
        )
        st.text(summary[:500] + ("…" if len(str(summary)) > 500 else ""))

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

        llm_debug = {}
        feature_options = extract_feature_options(analysis, debug_out=llm_debug)
        session_key = f"liked_features_{video_path_str}"
        if session_key not in st.session_state:
            st.session_state[session_key] = []
        liked_features = st.session_state[session_key]

        if feature_options:
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
            liked_features = []
            st.caption("未能从帧描述和总结文本中提取到特征短语。可配置 LLM_APP_URL、LLM_API_KEY 使用云端模型提取。")

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
