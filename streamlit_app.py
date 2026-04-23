"""
隐私保护轻量级 RAG 系统 - Streamlit 可视化界面
Privacy-Preserving Lightweight RAG System - Streamlit UI

用法：
  streamlit run streamlit_app.py

前置条件：
  1. Ollama 服务已启动：ollama serve
  2. Qdrant 向量库已就绪
  3. 依赖已安装：pip install streamlit

界面功能：
  - RAG 问答（支持加密/明文双通道切换）
  - 检索来源可视化（展示 Rerank 分数、来源文件）
  - 性能指标展示（检索延迟、生成延迟、总耗时）
  - 系统状态监控（Ollama / Qdrant / Collection）
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

import streamlit as st
import yaml

# ─────────────────────────────────────────────────────────────
# 项目路径设置
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# 懒加载 RAG 运行时（避免 Streamlit 重载时重复初始化）
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_runtime(
    config_path: str,
    key_file: str,
    collection_name: str,
    plaintext_mode: bool,
):
    """缓存 RAG 运行时，避免每次 rerun 重建。"""
    from src.rag_pipeline.rag_system import _build_runtime

    if plaintext_mode:
        from unencrypted.build_plaintext_rag import build_plaintext_runtime
        rag_system, _ = build_plaintext_runtime(
            config_path=config_path,
            key_file=key_file,
            collection_name=collection_name,
            allow_empty_collection=False,
        )
    else:
        rag_system, _ = _build_runtime(
            config_path=config_path,
            key_file=key_file,
            collection_name=collection_name,
            allow_empty_collection=False,
        )
    return rag_system


@st.cache_resource(show_spinner=False)
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_collection_info(collection_name: str, storage_path: str) -> dict | None:
    """获取 Qdrant Collection 信息。"""
    try:
        from src.retrieval import VectorStore
        from src.embedding import EmbeddingModel

        cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
        em = EmbeddingModel(model_name=cfg["embedding"]["model_name"])
        vs = VectorStore(
            collection_name=collection_name,
            dimension=em.get_dimension(),
            distance_metric=cfg["vector_db"]["distance_metric"],
            storage_path=storage_path,
        )
        return vs.get_collection_info()
    except Exception as e:
        return {"error": str(e)}


def check_ollama(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """检查 Ollama 是否可用。"""
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, ", ".join(models) if models else "(无已加载模型)"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="隐私保护 RAG 系统",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 侧边栏：配置
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 系统配置")

    # ── 模式切换 ──
    plaintext_mode = st.toggle(
        "🔓 明文模式（对比实验）",
        value=False,
        help="关闭时使用加密 RAG；开启时使用明文 RAG（plaintext_documents_lihua_world）",
    )
    mode_label = "明文 RAG" if plaintext_mode else "加密 RAG"
    mode_emoji = "🔓" if plaintext_mode else "🔒"

    # ── Collection 选择 ──
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg = load_config(str(config_path))

    if plaintext_mode:
        collection_options = [
            "plaintext_documents_lihua_world",
            "plaintext_documents_test1",
        ]
        storage_path = str(PROJECT_ROOT / "unencrypted" / "qdrant_storage_plaintext")
        default_col = "plaintext_documents_lihua_world"
    else:
        collection_options = [
            "encrypted_documents_lihua",
            "encrypted_documents_test1",
            cfg["vector_db"]["collection_name"],
        ]
        storage_path = str(PROJECT_ROOT / cfg["vector_db"]["storage_path"])
        default_col = cfg["vector_db"]["collection_name"]

    collection_name = st.selectbox(
        f"{mode_emoji} Collection",
        options=collection_options,
        index=0,
    )

    # ── 参数调节 ──
    st.markdown("---")
    st.markdown("### 🔧 检索参数")
    top_k = st.slider("Top-K（检索块数）", 1, 20, 5)
    temperature = st.slider("Temperature（生成随机性）", 0.0, 1.0, 0.2, 0.05)

    # ── 系统状态 ──
    st.markdown("---")
    st.markdown("### 🖥️ 系统状态")

    ollama_ok, ollama_models = check_ollama()
    if ollama_ok:
        st.success(f"✅ Ollama 在线\n\n已加载模型：{ollama_models}")
    else:
        st.error(f"❌ Ollama 不可用\n\n{ollama_models}\n\n请先运行：`ollama serve`")

    col_info = get_collection_info(collection_name, storage_path)
    if col_info and "error" not in col_info:
        pts = col_info.get("points_count", "?")
        st.info(f"📦 Collection：`{collection_name}`\n\n向量数：{pts}")
    elif col_info and "error" in col_info:
        st.warning(f"⚠️ Collection 未找到\n\n`{col_info['error']}`")
    else:
        st.info(f"📦 Collection：`{collection_name}`\n\n（状态未知）")

    # ── 模型信息 ──
    st.markdown("---")
    st.markdown("### 🤖 模型信息")
    st.caption(f"Embedding：`{cfg['embedding']['model_name']}`")
    llm_name = cfg.get("llm", {}).get("default_model") or cfg["llm"].get("model_name", "mistral")
    st.caption(f"LLM：`{llm_name}`（Ollama）")
    st.caption(f"量化：{cfg['llm']['quantization']['bits']}-bit {cfg['llm']['quantization']['type'].upper()}")


# ─────────────────────────────────────────────────────────────
# 主界面：标题
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"# {mode_emoji} 隐私保护轻量级 RAG 系统\n"
    f"**当前模式：{mode_label}** | 毕设演示系统"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 初始化 Session State
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "last_collection" not in st.session_state:
    st.session_state.last_collection = None

if "last_plaintext_mode" not in st.session_state:
    st.session_state.last_plaintext_mode = None


# ─────────────────────────────────────────────────────────────
# 加载 / 重载 RAG 运行时
# ─────────────────────────────────────────────────────────────
current_key = (collection_name, plaintext_mode)
if (
    st.session_state.last_collection != current_key
    or st.session_state.last_plaintext_mode != plaintext_mode
):
    st.session_state.last_collection = current_key
    st.session_state.last_plaintext_mode = plaintext_mode
    st.session_state.rag_system = None  # 强制重建

key_file = PROJECT_ROOT / "encryption.key"
if not ollama_ok:
    st.error("⚠️ Ollama 服务未启动，请先在终端运行 `ollama serve` 后刷新页面。")
    st.stop()

if st.session_state.rag_system is None:
    with st.spinner("🔄 初始化 RAG 系统（首次加载需数秒）..."):
        try:
            st.session_state.rag_system = load_rag_runtime(
                config_path=str(config_path),
                key_file=str(key_file),
                collection_name=collection_name,
                plaintext_mode=plaintext_mode,
            )
            st.success("✅ RAG 系统就绪！")
        except FileNotFoundError as e:
            st.error(f"文件未找到：{e}")
            st.stop()
        except Exception as e:
            st.error(f"初始化失败：{e}")
            st.stop()

rag = st.session_state.rag_system


# ─────────────────────────────────────────────────────────────
# 聊天历史渲染
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metrics"):
            m = msg["metrics"]
            cols = st.columns(4)
            cols[0].metric("🔍 检索延迟", f"{m['retrieval_time']:.3f}s")
            cols[1].metric("🧠 生成延迟", f"{m['generation_time']:.1f}s")
            cols[2].metric("⏱️ 总耗时", f"{m['total_time']:.1f}s")
            cols[3].metric("📄 检索块数", m["num_chunks"])
        if msg.get("sources"):
            with st.expander("📂 检索来源详情", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score", 0)
                    rr_score = src.get("rerank_score")
                    src_file = src.get("metadata", {}).get("source_file", "unknown")
                    chunk_id = src.get("metadata", {}).get("chunk_id", "?")
                    text_preview = (src.get("text") or "")[:200].replace("\n", " ")
                    score_str = f"向量={score:.3f}"
                    if rr_score is not None:
                        score_str += f" | Rerank={rr_score:.3f}"
                    st.markdown(
                        f"**[{i}]** `{src_file}` (chunk_id={chunk_id})\n"
                        f"   分数：{score_str}\n"
                        f"   内容：{text_preview}..."
                    )


# ─────────────────────────────────────────────────────────────
# 用户输入
# ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("请输入您的问题..."):
    # ── 显示用户消息 ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── 调用 RAG ──
    with st.chat_message("assistant"):
        with st.spinner("🤔 RAG 推理中（Mistral-7B 首次推理需加载模型，请耐心等待）..."):
            try:
                result = rag.answer_question(
                    question=prompt,
                    top_k=top_k,
                    temperature=temperature,
                )
                answer = result.get("answer", "")
                retrieval_time = result.get("retrieval_time", 0)
                generation_time = result.get("generation_time", 0)
                total_time = result.get("total_time", retrieval_time + generation_time)
                num_chunks = result.get("num_chunks_retrieved", 0)
                used_chunks = result.get("used_chunks") or []
                error = result.get("error")

                if error:
                    st.error(f"❌ 推理出错：{error}")
                else:
                    st.markdown(answer or "_（未返回答案）_")

                    # ── 性能指标 ──
                    cols = st.columns(4)
                    cols[0].metric("🔍 检索延迟", f"{retrieval_time:.3f}s")
                    cols[1].metric("🧠 生成延迟", f"{generation_time:.1f}s")
                    cols[2].metric("⏱️ 总耗时", f"{total_time:.1f}s")
                    cols[3].metric("📄 检索块数", num_chunks)

                    # ── 检索来源 ──
                    if used_chunks:
                        with st.expander("📂 检索来源详情", expanded=False):
                            for i, ch in enumerate(used_chunks, 1):
                                score = float(ch.get("score") or 0)
                                rr_score = ch.get("rerank_score")
                                meta = ch.get("metadata") or {}
                                src_file = meta.get("source_file", "unknown")
                                chunk_id = meta.get("chunk_id", "?")
                                text_preview = (ch.get("text") or "")[:200].replace("\n", " ")
                                score_str = f"向量={score:.3f}"
                                if rr_score is not None:
                                    score_str += f" | Rerank={float(rr_score):.3f}"
                                st.markdown(
                                    f"**[{i}]** `{src_file}` (chunk_id={chunk_id})\n"
                                    f"   分数：{score_str}\n"
                                    f"   内容：{text_preview}..."
                                )

                    # ── 保存历史 ──
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metrics": {
                            "retrieval_time": retrieval_time,
                            "generation_time": generation_time,
                            "total_time": total_time,
                            "num_chunks": num_chunks,
                        },
                        "sources": used_chunks,
                    })

            except Exception as e:
                st.error(f"❌ 发生错误：{e}")
                logging.exception("RAG inference error")


# ─────────────────────────────────────────────────────────────
# 底部说明
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🔒 本系统为本科毕业设计演示系统 | "
    "加密方案：AES-256-GCM | "
    "Embedding：all-MiniLM-L6-v2 | "
    "LLM：Mistral-7B Q4（Ollama）"
)
