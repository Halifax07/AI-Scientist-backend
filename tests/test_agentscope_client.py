from __future__ import annotations

import pytest

from fsad_scientist.agents.agentscope_client import _parse_json_object


def test_parse_json_object_accepts_prose_and_markdown_fences() -> None:
    response = (
        "Here is the structured result:\n"
        "```json\n"
        '{"decision": "continue", "confidence": 0.8}\n'
        "```\n"
        "The result is ready for review."
    )

    assert _parse_json_object(response) == {
        "decision": "continue",
        "confidence": 0.8,
    }


@pytest.mark.parametrize(
    "response",
    [
        "No structured response was returned.",
        "```json\n{not valid json}\n```",
        "[{'decision': 'continue'}]",
        '[{"decision": "continue"}]',
    ],
)
def test_parse_json_object_rejects_non_object_or_invalid_output(response: str) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_object(response)
