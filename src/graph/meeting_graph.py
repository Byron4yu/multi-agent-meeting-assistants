"""
LangGraph 会议处理图 —— 多Agent编排核心

编排模式: Pipeline + 并行 (Fan-out / Fan-in)

    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │ Transcription│  ← Pipeline 阶段
    │    Agent     │
    └──────┬───────┘
           │
    ┌──────┼───────┐  ← Fan-out (并行)
    │      │       │
    ▼      ▼       ▼
  Summary Action Insight
  Agent   Agent  Agent
    │      │       │
    └──────┼───────┘  ← Fan-in (汇聚)
           │
           ▼
    ┌──────────────┐
    │  Follow-up   │
    │    Agent     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │     END      │
    └──────────────┘

面试考点:
- LangGraph 的 State/Node/Edge 分别是什么？
- 并行执行是怎么实现的？（Fan-out + Fan-in）
- 如果某个并行节点失败了怎么办？（错误写入state，不阻塞其他节点）
"""

from __future__ import annotations

import base64
from operator import add
from typing import Any, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from loguru import logger

from ..agents.transcription_agent import TranscriptionAgent, TranscriptionConfig
from ..agents.summary_agent import SummaryAgent
from ..agents.action_agent import ActionAgent
from ..agents.insight_agent import InsightAgent
from ..agents.followup_agent import FollowUpAgent
from ..integrations.minimax_client import MiniMaxClient
from ..integrations.feishu_client import FeishuClient
from ..models.schemas import (
    MeetingState,
    MeetingStatus,
    create_initial_state,
)


# ============================================================
# LangGraph 状态类型定义
# ============================================================

class GraphState(TypedDict, total=False):
    """
    LangGraph 使用 TypedDict 定义状态结构。
    每个 Node（Agent）都读写这个共享状态。
    """
    meeting_id: str
    status: str
    audio_data: bytes

    # Transcription 输出
    transcript: Any
    transcript_text: str

    # 并行 Agent 输出
    summary: Any
    actions: Any
    insights: Any

    # Follow-up 输出
    followup: Any

    # 错误记录（使用 add reducer 支持并行 Agent 追加错误）
    errors: Annotated[list[str], add]


# ============================================================
# 构建 Meeting Graph
# ============================================================

def build_meeting_graph(
    llm_client: MiniMaxClient | None = None,
    feishu_client: FeishuClient | None = None,
    transcription_config: TranscriptionConfig | None = None,
) -> StateGraph:
    """
    构建会议处理 StateGraph

    这是整个系统的编排核心：
    1. 创建 5 个 Agent 实例
    2. 将它们注册为 Graph 的 Node
    3. 定义 Edge（流转关系）
    4. 编译为可执行的 Graph

    Args:
        llm_client: LLM 客户端（共享，避免重复创建）
        feishu_client: 飞书客户端
        transcription_config: 转写配置

    Returns:
        编译后的 StateGraph
    """
    # 共享依赖
    llm = llm_client or MiniMaxClient()
    feishu = feishu_client or FeishuClient()

    # 创建 Agent 实例
    transcription_agent = TranscriptionAgent(transcription_config)
    summary_agent = SummaryAgent(llm)
    action_agent = ActionAgent(llm, feishu)
    insight_agent = InsightAgent(llm)
    followup_agent = FollowUpAgent(feishu)

    # ---- 构建 StateGraph ----
    graph = StateGraph(GraphState)

    # 注册节点（Node = Agent）
    graph.add_node("transcription", transcription_agent.process)
    graph.add_node("summary", summary_agent.process)
    graph.add_node("action", action_agent.process)
    graph.add_node("insight", insight_agent.process)
    graph.add_node("followup", followup_agent.process)

    # ---- 定义边（Edge = 流转关系）----

    # Pipeline 阶段: START → Transcription
    graph.add_edge(START, "transcription")

    # Fan-out 并行: Transcription → [Summary, Action, Insight]
    graph.add_edge("transcription", "summary")
    graph.add_edge("transcription", "action")
    graph.add_edge("transcription", "insight")

    # Fan-in 汇聚: [Summary, Action, Insight] → Follow-up
    graph.add_edge("summary", "followup")
    graph.add_edge("action", "followup")
    graph.add_edge("insight", "followup")

    # 结束: Follow-up → END
    graph.add_edge("followup", END)

    logger.info("Meeting graph built successfully")
    return graph


def compile_meeting_graph(**kwargs) -> Any:
    """构建并编译 Graph（编译后可直接调用）"""
    graph = build_meeting_graph(**kwargs)
    compiled = graph.compile()
    logger.info("Meeting graph compiled successfully")
    return compiled


async def run_meeting_pipeline(
    meeting_id: str,
    audio_base64: str = "",
    **kwargs,
) -> dict:
    """
    执行完整的会议处理 Pipeline

    这是对外暴露的主入口函数：
    1. 创建初始状态
    2. 编译 Graph
    3. 执行 Graph
    4. 返回最终状态

    Args:
        meeting_id: 会议ID
        audio_data: 音频数据（为空则使用演示数据）

    Returns:
        最终的 MeetingState 字典
    """
    logger.info(f"Starting meeting pipeline: {meeting_id}")

    initial_state: dict[str, Any] = {
        "meeting_id": meeting_id,
        "status": MeetingStatus.PENDING,
        "audio_data": base64.b64decode(audio_base64) if audio_base64 else b"",
        "errors": [],
    }
    compiled_graph = compile_meeting_graph(**kwargs)

    final_state = await compiled_graph.ainvoke(initial_state)

    errors = final_state.get("errors", [])
    if errors:
        logger.warning(f"Pipeline completed with errors: {errors}")
    else:
        logger.info(f"Pipeline completed successfully for: {meeting_id}")

    return final_state
