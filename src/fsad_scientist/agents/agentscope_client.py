from __future__ import annotations

import json
import os
from typing import Any

from openai import APIConnectionError, APIStatusError


class AgentScopeUnavailableError(RuntimeError):
    pass


class AgentScopeJsonClient:
    """Thin adapter around AgentScope 2.x and Alibaba Cloud Model Studio.

    The workflow stores durable state outside the model. This adapter is only
    responsible for one structured reasoning turn and can therefore be replaced
    without changing the scientific ledger or experiment runner.
    """

    def __init__(
        self,
        *,
        model: str = "qwen3.7-plus",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    async def complete(
        self,
        *,
        role_name: str,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        api_key = self._api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise AgentScopeUnavailableError("DASHSCOPE_API_KEY is not configured")

        try:
            from agentscope.agent import Agent
            from agentscope.credential import DashScopeCredential
            from agentscope.message import UserMsg
            from agentscope.model import DashScopeChatModel
            from agentscope.tool import Toolkit
        except ImportError as exc:
            raise AgentScopeUnavailableError(
                "Install the optional 'agent' dependencies before using AgentScope"
            ) from exc

        agent = Agent(
            name=role_name,
            system_prompt=(
                f"{system_prompt}\n"
                "Return one valid JSON object only. Do not invent citations, metrics, "
                "experiment runs, or verification status."
            ),
            model=DashScopeChatModel(
                credential=DashScopeCredential(api_key=api_key),
                model=self.model,
            ),
            toolkit=Toolkit(),
        )
        try:
            reply = await agent.reply(
                UserMsg(name="workflow", content=json.dumps(payload, ensure_ascii=False))
            )
        except APIStatusError as exc:
            error_code = _api_error_code(exc)
            if error_code == "Arrearage":
                message = "DashScope 账户欠费或余额不足，请充值后重试。"
            elif exc.status_code in {401, 403}:
                message = "DashScope API Key 无效或没有当前模型的访问权限。"
            else:
                message = f"DashScope 请求失败（HTTP {exc.status_code}）。"
            raise AgentScopeUnavailableError(message) from exc
        except APIConnectionError as exc:
            raise AgentScopeUnavailableError(
                "无法连接 DashScope，请检查网络和服务地址。"
            ) from exc
        return _parse_json_object(reply.get_text_content())


def _api_error_code(exc: APIStatusError) -> str | None:
    body = exc.body
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    return str(error.get("code")) if isinstance(error, dict) and error.get("code") else None


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Agent output must be one JSON object")
    return parsed
