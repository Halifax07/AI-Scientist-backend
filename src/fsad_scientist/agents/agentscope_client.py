from __future__ import annotations

import json
import os
from typing import Any


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
        reply = await agent.reply(
            UserMsg(name="workflow", content=json.dumps(payload, ensure_ascii=False))
        )
        return _parse_json_object(reply.get_text_content())


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
