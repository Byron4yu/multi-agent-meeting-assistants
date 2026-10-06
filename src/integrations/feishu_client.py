"""飞书 Open API 集成客户端 - 消息推送和任务管理"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class FeishuClient:
    """
    飞书开放平台 API 客户端

    职责:
    - 发送群消息（推送会议纪要）
    - 创建任务（同步待办事项）
    - Webhook 机器人消息

    API 文档: https://open.feishu.cn/document/home/index
    """

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        app_id: str | None = None,
        app_secret: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ):
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL", "")
        self.webhook_secret = webhook_secret or os.getenv("FEISHU_WEBHOOK_SECRET", "")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._tenant_token: str = ""
        self._token_expires_at: float = 0
        self._enabled = bool(
            (self.app_id and self.app_secret) or self.webhook_url
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def _get_tenant_token(self) -> str:
        """获取 tenant_access_token（自动缓存和刷新）"""
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token

        if not (self.app_id and self.app_secret):
            return ""

        resp = await self._client.post(
            f"{self.BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        data = resp.json()
        self._tenant_token = data.get("tenant_access_token", "")
        self._token_expires_at = time.time() + data.get("expire", 7200) - 300
        return self._tenant_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_webhook_message(
        self,
        title: str,
        content: str,
        msg_type: str = "interactive",
    ) -> bool:
        """
        通过 Webhook 机器人发送消息到飞书群

        Args:
            title: 消息标题
            content: 消息内容（Markdown格式）
            msg_type: 消息类型

        Returns:
            是否发送成功
        """
        if not self.webhook_url:
            logger.warning("Feishu webhook not configured, skipping")
            return False

        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content,
                    }
                ],
            },
        }

        # 如果配置了签名密钥，生成签名并放入请求 body（飞书要求 timestamp 和 sign 在 body 中）
        if self.webhook_secret:
            timestamp = str(int(time.time()))
            string_to_sign = f"{timestamp}\n{self.webhook_secret}"
            hmac_code = hmac.new(
                self.webhook_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
            sign = base64.b64encode(hmac_code).decode("utf-8")
            card["timestamp"] = timestamp
            card["sign"] = sign
            logger.debug(
                f"[FeishuClient] Webhook signing: timestamp={timestamp}, "
                f"sign_len={len(sign)}, secret_len={len(self.webhook_secret)}"
            )

        resp = await self._client.post(self.webhook_url, json=card)
        data = resp.json()
        code = data.get("code", -1)
        success = code == 0
        if success:
            logger.info(f"Feishu webhook message sent: {title}")
        else:
            # 提供更有针对性的错误提示
            hints = {
                19001: "webhook URL 中的 access token 无效，请检查 FEISHU_WEBHOOK_URL 是否完整",
                19021: "签名校验失败！请检查 FEISHU_WEBHOOK_SECRET 是否与机器人设置中的签名密钥一致，或关闭机器人的签名校验",
                19022: "卡片格式有误，请检查 card 结构",
            }
            hint = hints.get(code, "")
            logger.error(
                f"Feishu webhook failed (code={code}): {data.get('msg', '')}"
                + (f" | 提示: {hint}" if hint else "")
            )
        return success

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_message(
        self,
        receive_id: str,
        content: str,
        receive_id_type: str = "chat_id",
        msg_type: str = "text",
    ) -> dict[str, Any]:
        """
        通过 API 发送消息

        Args:
            receive_id: 接收者ID（群ID或用户ID）
            content: 消息内容
            receive_id_type: ID类型 chat_id/open_id/user_id/email
            msg_type: 消息类型 text/interactive/post
        """
        token = await self._get_tenant_token()
        if not token:
            return {"success": False, "error": "No token"}

        resp = await self._client.post(
            f"{self.BASE_URL}/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": content if isinstance(content, str) else str(content),
            },
        )
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def create_task(
        self,
        summary: str,
        description: str = "",
        due_timestamp: int | None = None,
        assignee_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        创建飞书任务

        Args:
            summary: 任务标题
            description: 任务描述
            due_timestamp: 截止时间戳（秒）
            assignee_ids: 负责人open_id列表
        """
        token = await self._get_tenant_token()
        if not token:
            return {"success": False, "error": "No token"}

        task_body: dict[str, Any] = {
            "summary": summary,
            "description": description or f"来源：会议助手自动创建",
        }
        if due_timestamp:
            task_body["due"] = {"timestamp": str(due_timestamp), "is_all_day": True}

        resp = await self._client.post(
            f"{self.BASE_URL}/task/v2/tasks",
            headers={"Authorization": f"Bearer {token}"},
            json=task_body,
        )
        data = resp.json()
        task_id = data.get("data", {}).get("task", {}).get("id", "")
        logger.info(f"Created Feishu task: {task_id} - {summary}")
        return {"task_id": task_id, "data": data}

    async def send_meeting_summary(
        self,
        title: str,
        summary_md: str,
        action_items_md: str,
        insights_md: str,
    ) -> bool:
        """发送完整的会议纪要卡片消息"""
        content = (
            f"**会议主题**: {title}\n\n"
            f"---\n\n"
            f"**📋 会议纪要**\n{summary_md}\n\n"
            f"---\n\n"
            f"**✅ 待办事项**\n{action_items_md}\n\n"
            f"---\n\n"
            f"**📊 会议洞察**\n{insights_md}"
        )
        return await self.send_webhook_message(
            title=f"📝 会议纪要 | {title}",
            content=content,
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
