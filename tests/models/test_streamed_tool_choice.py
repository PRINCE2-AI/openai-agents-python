"""Streamed Chat Completions runs whose function tool_choice matches a Responses builtin.

`web_search` and `mcp` are natural function names for Chat Completions models, which cannot
use the hosted tools of the same name. The placeholder response built for the stream used to
report them as hosted selectors and failed validation.
"""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

pytestmark = [pytest.mark.asyncio, pytest.mark.allow_call_model_methods]


def _sse_body() -> bytes:
    chunk = {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


@function_tool
def web_search(query: str) -> str:
    """Search the app's own index."""
    return f"results for {query}"


@function_tool
def mcp(query: str) -> str:
    """Call the app's own MCP bridge."""
    return f"results for {query}"


@pytest.mark.parametrize(("tool_name", "tool"), [("web_search", web_search), ("mcp", mcp)])
async def test_streamed_run_accepts_function_tool_choice_named_like_a_hosted_tool(
    tool_name: str, tool: Any
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200, content=_sse_body(), headers={"content-type": "text/event-stream"}
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://api.openai.com/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler), trust_env=False),
    ) as client:
        result = Runner.run_streamed(
            Agent(
                name="synthetic",
                model=OpenAIChatCompletionsModel("gpt-4o-mini", client),
                tools=[tool],
                model_settings=ModelSettings(tool_choice=tool_name),
            ),
            "search for x",
            run_config=RunConfig(tracing_disabled=True),
        )
        created_tool_choice = None
        async for event in result.stream_events():
            if event.type == "raw_response_event" and event.data.type == "response.created":
                created_tool_choice = event.data.response.tool_choice

    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": tool_name},
    }
    assert created_tool_choice is not None
    assert created_tool_choice.model_dump() == {"type": "function", "name": tool_name}
    assert result.final_output == "done"
