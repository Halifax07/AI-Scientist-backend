from pydantic import SecretStr

from fsad_scientist.agents.agentscope_client import AgentScopeJsonClient
from fsad_scientist.config import Settings


def test_dashscope_key_loads_from_unprefixed_env_file_without_repr_leak(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AISCIENTIST_DASHSCOPE_API_KEY", raising=False)
    secret = "test-dashscope-secret-value-that-must-not-appear-in-repr"
    env_file = tmp_path / ".env"
    env_file.write_text(f"DASHSCOPE_API_KEY={secret}\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert isinstance(settings.dashscope_api_key, SecretStr)
    assert settings.dashscope_api_key_value == secret
    assert secret not in repr(settings)


def test_explicit_dashscope_key_is_passed_to_agent_client() -> None:
    client = AgentScopeJsonClient(api_key="explicit-test-key")

    assert client._api_key == "explicit-test-key"
