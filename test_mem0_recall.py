#!/usr/bin/env python3
"""
测试脚本：用自然语言向 Mem0 提问，看能否召回与视频偏好相关的记忆。

用法:
  python test_mem0_recall.py
    跑两轮测试：① 通用偏好 ② 精准测试「用户是否喜欢 户外 类型视频，喜欢的评分有多少？」
  python test_mem0_recall.py "用户喜欢什么类型的视频"
    只跑你指定的问题
  python test_mem0_recall.py --precise 舞蹈
    只跑精准测试，把「舞蹈」代入：用户是否喜欢 舞蹈 类型视频，喜欢的评分有多少？

环境变量（.env）：与 sync_rated_to_mem0.py 相同
  MEM0_API_KEY（必填）, MEM0_USER_ID, MEM0_ORG_ID, MEM0_PROJECT_ID
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "").strip()
MEM0_USER_ID = os.environ.get("MEM0_USER_ID", "video_rater").strip() or "video_rater"
MEM0_ORG_ID = os.environ.get("MEM0_ORG_ID", "").strip() or None
MEM0_PROJECT_ID = os.environ.get("MEM0_PROJECT_ID", "").strip() or None


def get_client():
    try:
        from mem0 import MemoryClient
    except ImportError:
        print("请先安装: pip install mem0ai", file=sys.stderr)
        sys.exit(1)
    kwargs = {"api_key": MEM0_API_KEY}
    if MEM0_ORG_ID:
        kwargs["org_id"] = MEM0_ORG_ID
    if MEM0_PROJECT_ID:
        kwargs["project_id"] = MEM0_PROJECT_ID
    return MemoryClient(**kwargs)


def _print_memory(i, m, title=None):
    """打印单条记忆的摘要。"""
    mem_text = m.get("memory") or m.get("text") or "(无内容)"
    meta = m.get("metadata") or {}
    vid = meta.get("video_path", "")
    created = m.get("created_at", "")
    mid = m.get("id", "")[:8] + "..." if m.get("id") else ""
    if title:
        print(title)
    print(f"  [{i}] id={mid} video_path={vid} created={created}")
    print(f"      memory: {mem_text[:200]}{'...' if len(mem_text) > 200 else ''}")
    print()


def run_get_all(client):
    """拉取当前用户全部记忆。"""
    print("--- get_all (user_id=%s) ---" % MEM0_USER_ID)
    try:
        # v2 API 要求必须传 filters，且不能为空（见 https://docs.mem0.ai/api-reference/memory/get-memories）
        out = client.get_all(
            filters={"AND": [{"user_id": MEM0_USER_ID}]},
            version="v2",
        )
    except Exception as e:
        print("get_all 失败:", e)
        return
    results = out.get("results") if isinstance(out, dict) else out
    if not results:
        print("无记忆。")
        return
    print("共 %d 条记忆:\n" % len(results))
    for i, m in enumerate(results, 1):
        _print_memory(i, m)


def run_search(client, query: str):
    """按自然语言查询检索相关记忆。"""
    print('--- search (query="%s", user_id=%s) ---' % (query, MEM0_USER_ID))
    try:
        # v2 search API 要求必须传 filters（见 https://docs.mem0.ai/api-reference/memory/search-memories）
        out = client.search(
            query,
            user_id=MEM0_USER_ID,
            version="v2",
            filters={"AND": [{"user_id": MEM0_USER_ID}]},
        )
    except Exception as e:
        print("search 失败:", e)
        return
    # 返回可能是 {"results": [...]} 或 {"memories": [...]} 或直接 list
    if isinstance(out, dict):
        memories = out.get("results") or out.get("memories") or []
    else:
        memories = out if isinstance(out, list) else []
    if not memories:
        print("未检索到相关记忆。")
        return
    print("共 %d 条相关记忆:\n" % len(memories))
    for i, m in enumerate(memories, 1):
        _print_memory(i, m)


# 无参数时跑两轮：通用偏好 + 精准测试
DEFAULT_QUERY = "用户喜欢什么类型的视频？有哪些评分高或偏好的视频特征？"
# 精准测试：是否喜欢某类视频及评分（可替换类型词做更精准查找）
PRECISE_QUERY_TEMPLATE = "用户是否喜欢 {} 类型视频，喜欢的评分有多少？"
DEFAULT_PRECISE_TYPE = "户外"


def main():
    if not MEM0_API_KEY:
        print("未设置 MEM0_API_KEY，请在 .env 中配置", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    argv = [a.strip() for a in sys.argv[1:] if a.strip()]

    if len(argv) >= 2 and argv[0] == "--precise":
        # python test_mem0_recall.py --precise 舞蹈
        video_type = argv[1]
        query = PRECISE_QUERY_TEMPLATE.format(video_type)
        run_search(client, query)
        return

    if len(argv) >= 1:
        # 自定义单条问题
        run_search(client, argv[0])
        return

    # 无参数：跑两轮测试
    print("========== 测试 1：通用偏好 ==========\n")
    run_search(client, DEFAULT_QUERY)
    print("\n========== 测试 2：精准查找（类型 + 评分） ==========\n")
    run_search(client, PRECISE_QUERY_TEMPLATE.format(DEFAULT_PRECISE_TYPE))


if __name__ == "__main__":
    main()
