"""MiniMax LLM 客户端 - 支持 M2.7 模型调用"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class MiniMaxClient:
    """
    MiniMax API 客户端

    支持 MiniMax M2.7 模型，兼容 OpenAI 接口格式。
    文档: https://platform.minimaxi.com/document/guides/chat-model/chat/api
    """

    BASE_URL = "https://api.minimax.chat/v1"

    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self.model = model or os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            logger.warning(
                "MiniMaxClient initialized without API key — "
                "set MINIMAX_API_KEY env var to enable LLM calls"
            )

        self._client = httpx.AsyncClient(timeout=60.0, headers=headers)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        """
        调用 MiniMax 聊天接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            response_format: 输出格式约束 (如 {"type": "json_object"})

        Returns:
            模型生成的文本
        """
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY is not set. "
                "Configure it in environment variables or .env file."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.BASE_URL}/text/chatcompletion_v2"
        if self.group_id:
            url = f"{url}?GroupId={self.group_id}"

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        logger.error(f"MiniMax API unexpected response: {data}")
        raise ValueError(f"Unexpected API response: {data}")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict:
        """调用聊天接口并解析 JSON 输出"""
        text = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # 尝试用 json_repair 库（如果已安装）——它能处理绝大多数 LLM JSON 问题
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except ImportError:
            pass
        except Exception:
            pass

        # 第一次尝试：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 第二次尝试：提取 {} 包裹的内容（去掉可能的 markdown 代码块标记）
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = cleaned[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # 第三次尝试：我们自己的修复逻辑
                repaired = self._repair_json(json_str)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    logger.error(
                        f"[MiniMaxClient] JSON parse failed after repair. "
                        f"Error: {e}\n"
                        f"Raw text (first 800 chars): {text[:800]}\n"
                        f"Extracted JSON: {json_str[:800]}"
                    )
                    raise

        logger.error(
            f"[MiniMaxClient] No JSON found in response. "
            f"Raw text (first 500 chars): {text[:500]}"
        )
        raise ValueError(f"Model did not return valid JSON: {text[:200]}")

    @staticmethod
    def _repair_json(json_str: str) -> str:
        """尝试修复常见的 JSON 格式问题（未转义引号、截断、尾随逗号等）"""
        import re

        repaired = json_str

        # 1. 移除尾随逗号 (}, ] 前的逗号)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # 2. 处理未转义的引号：扫描字符串，转义掉字符串值内部的 ASCII 双引号
        #    策略：逐字符遍历，跟踪是否在字符串内；如果在字符串内遇到 " 且后面不是
        #    结构字符（, : } ] 或空白+这些），就认为是内容引号，需要转义
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(repaired):
            ch = repaired[i]
            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue
            if ch == "\\":
                result.append(ch)
                escape_next = True
                i += 1
                continue
            if ch == '"':
                if not in_string:
                    in_string = True
                    result.append(ch)
                else:
                    # 我们在字符串内，判断这个 " 是结束引号还是内容引号
                    # 向后看，跳过空白，看下一个非空白字符
                    j = i + 1
                    while j < len(repaired) and repaired[j] in " \t\n\r":
                        j += 1
                    next_char = repaired[j] if j < len(repaired) else ""
                    # 如果下一个字符是结构字符，说明这是字符串结束引号
                    if next_char in (",", ":", "}", "]", ""):
                        in_string = False
                        result.append(ch)
                    else:
                        # 否则认为是内容引号，转义它
                        result.append('\\"')
                i += 1
                continue
            result.append(ch)
            i += 1

        repaired = "".join(result)

        # 3. 处理截断：补齐未闭合的字符串、数组、对象
        # 统计未闭合的结构
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        # 如果最后一个非空白字符不是 } 或 ]，可能被截断了
        stripped = repaired.rstrip()
        if stripped and stripped[-1] not in "}]":
            # 尝试补齐：先关闭可能打开的字符串
            if stripped.count('"') % 2 == 1:
                stripped += '"'
            # 关闭数组
            for _ in range(max(0, open_brackets)):
                stripped += "]"
            # 关闭对象
            for _ in range(max(0, open_braces)):
                stripped += "}"
            repaired = stripped

        return repaired

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
