#!/usr/bin/env python3
"""
独立脚本：从 PostgreSQL 读取已评分视频，将未同步到 Mem0 的记录同步到 Mem0。

- 使用表 video_preferences 的列 mem0_synced_at 标记是否已同步（不存在则自动添加）。
- 仅处理 mem0_synced_at IS NULL 的已评分记录。
- 一条记录对应一次 Mem0 API 请求；同步成功后更新该条 mem0_synced_at = NOW()。

环境变量（.env）：
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD  — PostgreSQL
  MEM0_API_KEY                                   — Mem0 API Key（必填）
  MEM0_USER_ID                                   — 可选，默认为 video_rater
  MEM0_ORG_ID, MEM0_PROJECT_ID                   — 可选，Mem0 项目/组织
"""

import json
import logging
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# 从项目根目录加载 .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", "5432")),
    "dbname": os.environ.get("PGDATABASE", "video_rater"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", ""),
}

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "").strip()
MEM0_USER_ID = os.environ.get("MEM0_USER_ID", "video_rater").strip() or "video_rater"
MEM0_ORG_ID = os.environ.get("MEM0_ORG_ID", "").strip() or None
MEM0_PROJECT_ID = os.environ.get("MEM0_PROJECT_ID", "").strip() or None


def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def ensure_mem0_synced_column(conn):
    """确保 video_preferences 存在 mem0_synced_at 列。"""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE video_preferences
            ADD COLUMN IF NOT EXISTS mem0_synced_at TIMESTAMPTZ DEFAULT NULL
        """)
        conn.commit()
    logger.info("已确认表 video_preferences 存在列 mem0_synced_at")


def fetch_rated_unsynced(conn):
    """返回已评分且未同步到 Mem0 的所有行。"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, video_path, json_path, overall_score, liked_features, raw_analysis, created_at
            FROM video_preferences
            WHERE mem0_synced_at IS NULL
            ORDER BY id
        """)
        return cur.fetchall()


def build_memory_content(row):
    """
    构造送给 Mem0 的文本：明确这是用户「喜欢看的视频」+ 喜欢程度评分，
    并强调评分与视频特征的关联，便于 Mem0 做偏好记忆与检索。
    """
    video_path = row["video_path"]
    score = float(row["overall_score"])
    liked = row.get("liked_features") or []
    if isinstance(liked, str):
        try:
            liked = json.loads(liked) if liked else []
        except json.JSONDecodeError:
            liked = []
    raw = row.get("raw_analysis") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raw = {}

    summary = (
        raw.get("summary")
        or raw.get("description")
        or (raw.get("video_description") or {}).get("response")
        or ""
    )
    if isinstance(summary, dict):
        summary = summary.get("response") or summary.get("text") or str(summary)[:500]
    summary = (summary or "")[:2000]

    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [tags] if tags else []
    tags = [str(t) for t in tags[:15]]
    color_palette = raw.get("color_palette")
    if isinstance(color_palette, list):
        color_palette = ", ".join(str(c) for c in color_palette[:10])
    color_palette = str(color_palette or "").strip() or None

    # 明确语义：用户喜欢该视频；评分 1–10 表示喜欢程度；评分与下列特征强关联
    lines = [
        "The user liked this video and gave it a preference score of {score}/10 (1=dislike, 10=strong like). "
        "This score reflects how much they liked the video and should be associated with the following video characteristics:".format(
            score=score
        ),
    ]
    feature_parts = []
    if liked:
        feature_parts.append("user-selected liked features: " + ", ".join(str(x) for x in liked[:20]))
    if tags:
        feature_parts.append("video tags: " + ", ".join(tags))
    if color_palette:
        feature_parts.append("color palette: " + color_palette)
    if feature_parts:
        lines.append(" ".join(feature_parts) + ".")
    if summary:
        lines.append("Video content summary: " + summary)
    lines.append("(Video identifier: " + video_path + ")")

    return "\n".join(lines)


def sync_one_to_mem0(client, row):
    """将一条评分记录写入 Mem0，返回是否成功。"""
    messages = [{"role": "user", "content": build_memory_content(row)}]
    metadata = {"source": "video_rater", "video_path": row["video_path"]}

    kwargs = {
        "user_id": MEM0_USER_ID,
        "metadata": metadata,
        "version": "v2",
        "async_mode": False,
    }
    if MEM0_ORG_ID:
        kwargs["org_id"] = MEM0_ORG_ID
    if MEM0_PROJECT_ID:
        kwargs["project_id"] = MEM0_PROJECT_ID

    try:
        client.add(messages, **kwargs)
        return True
    except Exception as e:
        logger.warning("Mem0 add 失败 video_path=%s: %s", row["video_path"], e)
        return False


def mark_synced(conn, video_path):
    """将指定 video_path 标记为已同步。"""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE video_preferences SET mem0_synced_at = NOW() WHERE video_path = %s",
            (video_path,),
        )
        conn.commit()


def main():
    if not MEM0_API_KEY:
        logger.error("未设置 MEM0_API_KEY，请在 .env 中配置")
        sys.exit(1)

    try:
        from mem0 import MemoryClient
    except ImportError:
        logger.error("请安装 mem0ai: pip install mem0ai")
        sys.exit(1)

    client_kw = {"api_key": MEM0_API_KEY}
    if MEM0_ORG_ID:
        client_kw["org_id"] = MEM0_ORG_ID
    if MEM0_PROJECT_ID:
        client_kw["project_id"] = MEM0_PROJECT_ID
    client = MemoryClient(**client_kw)

    conn = get_conn()
    try:
        ensure_mem0_synced_column(conn)
        rows = fetch_rated_unsynced(conn)
    finally:
        conn.close()

    if not rows:
        logger.info("没有需要同步的已评分记录（mem0_synced_at 均为已设置）")
        return

    logger.info("待同步记录数: %d（一条记录对应一次 API 请求）", len(rows))
    ok_count, fail_count = 0, 0
    for row in rows:
        if sync_one_to_mem0(client, row):
            conn = get_conn()
            try:
                mark_synced(conn, row["video_path"])
                ok_count += 1
            finally:
                conn.close()
        else:
            fail_count += 1

    logger.info("同步完成: 成功 %d, 失败 %d", ok_count, fail_count)
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
