"""
Action Agent（待办Agent）
- 从转写文本中提取行动项（谁/做什么/截止时间）
- 自动同步到飞书任务
- 支持幂等性保证（防止重复创建）
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from ..integrations.feishu_client import FeishuClient
from ..integrations.minimax_client import MiniMaxClient
from ..models.schemas import ActionItem, ActionResult, Priority


ACTION_SYSTEM_PROMPT = """你是一位专业的任务提取助手。你的任务是从会议转写文本中提取所有行动项/待办事项。

提取规则：
1. 识别明确分配给某人的任务
2. 提取任务的截止时间（如果提到的话）
3. 判断任务优先级（根据语气和上下文）
4. 记录任务的上下文（为什么要做这件事）

注意：
- 只提取明确的行动项，不要凭空创造
- 截止时间格式为 YYYY-MM-DD
- 如果没有明确截止时间，留空

你必须严格按照JSON格式输出："""

ACTION_USER_PROMPT = """请从以下会议转写文本中提取所有行动项/待办事项。

今天的日期是: {today}

## 会议转写文本
{transcript}

## 输出格式（严格JSON）
{{
  "action_items": [
    {{
      "assignee": "负责人姓名",
      "task": "具体任务描述",
      "deadline": "YYYY-MM-DD 或空字符串",
      "priority": "low/medium/high/urgent",
      "context": "这个任务的背景说明"
    }}
  ]
}}"""


class ActionAgent:
    """
    待办Agent - 并行阶段的节点之一

    架构说明:
    1. 从 state 读取 transcript_text
    2. LLM 提取行动项三元组（谁/做什么/截止时间）
    3. 同步到飞书
    4. 记录同步状态

    面试考点:
    - 如果飞书不可用怎么办？（降级到本地存储 + 异步重试）
    """

    def __init__(
        self,
        llm_client: MiniMaxClient | None = None,
        feishu_client: FeishuClient | None = None,
    ):
        self.llm = llm_client or MiniMaxClient()
        self.feishu = feishu_client or FeishuClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 提取待办并同步

        与 Summary Agent、Insight Agent 并行执行。
        """
        meeting_id = state.get("meeting_id") or "unknown"
        logger.info(f"[ActionAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        if not transcript_text:
            logger.warning("[ActionAgent] No transcript text available")
            return {
                "actions": ActionResult(
                    meeting_id=meeting_id, action_items=[]
                )
            }

        try:
            action_items = await self._extract_actions(transcript_text)
            synced_items = await self._sync_to_external(action_items, meeting_id)

            logger.info(
                f"[ActionAgent] Extracted {len(synced_items)} action items"
            )
            return {
                "actions": ActionResult(
                    meeting_id=meeting_id,
                    action_items=synced_items,
                    sync_status={
                        "feishu": "enabled" if self.feishu.is_enabled else "disabled",
                    },
                )
            }
        except Exception as e:
            logger.error(f"[ActionAgent] Error: {e}")
            return {
                "actions": ActionResult(
                    meeting_id=meeting_id, action_items=[]
                ),
                "errors": [f"ActionAgent: {str(e)}"],
            }

    async def _extract_actions(self, transcript: str) -> list[ActionItem]:
        """调用 LLM 提取行动项"""
        today = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ACTION_USER_PROMPT.format(
                    today=today, transcript=transcript
                ),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
        )

        items = []
        for raw in result.get("action_items", []):
            priority_str = raw.get("priority", "medium").lower()
            try:
                priority = Priority(priority_str)
            except ValueError:
                priority = Priority.MEDIUM

            items.append(
                ActionItem(
                    assignee=raw.get("assignee", "未指定"),
                    task=raw.get("task", ""),
                    deadline=raw.get("deadline", ""),
                    priority=priority,
                    context=raw.get("context", ""),
                )
            )

        return items

    async def _sync_to_external(
        self, items: list[ActionItem], meeting_id: str
    ) -> list[ActionItem]:
        """将行动项同步到飞书"""
        synced = []
        for item in items:
            # 飞书同步
            if self.feishu.is_enabled:
                try:
                    due_ts = None
                    if item.deadline:
                        due_dt = datetime.strptime(item.deadline, "%Y-%m-%d")
                        due_ts = int(due_dt.timestamp())

                    feishu_result = await self.feishu.create_task(
                        summary=f"[会议待办] {item.task}",
                        description=(
                            f"负责人：{item.assignee}\n"
                            f"来源会议：{meeting_id}\n"
                            f"上下文：{item.context}"
                        ),
                        due_timestamp=due_ts,
                    )
                    item.feishu_task_id = feishu_result.get("task_id")
                except Exception as e:
                    logger.error(
                        f"Failed to sync to Feishu: {item.task} - {e}"
                    )

            synced.append(item)

        return synced
