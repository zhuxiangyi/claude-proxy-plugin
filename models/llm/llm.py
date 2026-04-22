from collections.abc import Generator
from typing import Optional, Union

import anthropic

from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel


def _build_client(credentials: dict) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=credentials.get("api_key", ""),
        base_url=credentials.get("endpoint_url", "").rstrip("/"),
    )


def _convert_messages(prompt_messages: list[PromptMessage]) -> tuple[Optional[str], list[dict]]:
    system: Optional[str] = None
    messages: list[dict] = []

    for msg in prompt_messages:
        if isinstance(msg, SystemPromptMessage):
            system = msg.content if isinstance(msg.content, str) else str(msg.content)
            continue

        role = "user" if isinstance(msg, UserPromptMessage) else "assistant"

        if isinstance(msg, AssistantPromptMessage) and msg.tool_calls:
            content = [
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": tc.function.arguments if isinstance(tc.function.arguments, dict) else {},
                }
                for tc in msg.tool_calls
            ]
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            content = [
                {"type": "text", "text": b.data}
                for b in msg.content
                if isinstance(b, TextPromptMessageContent)
            ]
        else:
            content = str(msg.content)

        messages.append({"role": role, "content": content})

    return system, messages


def _convert_tools(tools: list[PromptMessageTool]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in (tools or [])
    ]


def _make_usage(usage: anthropic.types.Usage) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=usage.input_tokens,
        prompt_unit_price=0,
        prompt_price_unit=0,
        prompt_price=0,
        completion_tokens=usage.output_tokens,
        completion_unit_price=0,
        completion_price_unit=0,
        completion_price=0,
        total_tokens=usage.input_tokens + usage.output_tokens,
        total_price=0,
        currency="USD",
        latency=0,
    )


class ClaudeProxyLargeLanguageModel(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        client = _build_client(credentials)
        api_model = model
        system, messages = _convert_messages(prompt_messages)

        params: dict = {
            "model": api_model,
            "messages": messages,
            "max_tokens": model_parameters.get("max_tokens", 8192),
        }
        if system:
            params["system"] = system
        if stop:
            params["stop_sequences"] = stop
        if tools:
            params["tools"] = _convert_tools(tools)

        try:
            if stream:
                return self._stream_invoke(client, params, model)
            return self._sync_invoke(client, params, model)
        except anthropic.AuthenticationError as e:
            raise InvokeAuthorizationError(str(e))
        except anthropic.BadRequestError as e:
            raise InvokeBadRequestError(str(e))
        except anthropic.RateLimitError as e:
            raise InvokeRateLimitError(str(e))
        except anthropic.APIConnectionError as e:
            raise InvokeConnectionError(str(e))
        except anthropic.APIStatusError as e:
            raise InvokeServerUnavailableError(str(e))

    def _sync_invoke(self, client: anthropic.Anthropic, params: dict, model: str) -> LLMResult:
        response = client.messages.create(**params)
        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    AssistantPromptMessage.ToolCall(
                        id=block.id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )
        message = AssistantPromptMessage(content=text, tool_calls=tool_calls)
        return LLMResult(model=model, prompt_messages=[], message=message, usage=_make_usage(response.usage))

    def _stream_invoke(self, client: anthropic.Anthropic, params: dict, model: str) -> Generator[LLMResultChunk, None, None]:
        with client.messages.stream(**params) as stream:
            current_tool_id: Optional[str] = None
            current_tool_name: Optional[str] = None
            current_tool_json: str = ""
            usage_data: Optional[anthropic.types.Usage] = None

            for event in stream:
                etype = type(event).__name__

                if etype == "RawContentBlockStartEvent":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_json = ""

                elif etype == "RawContentBlockDeltaEvent":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=[],
                            delta=LLMResultChunkDelta(
                                index=event.index,
                                message=AssistantPromptMessage(content=delta.text),
                            ),
                        )
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json

                elif etype == "RawContentBlockStopEvent":
                    if current_tool_id and current_tool_name:
                        import json
                        try:
                            tool_input = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=[],
                            delta=LLMResultChunkDelta(
                                index=event.index,
                                message=AssistantPromptMessage(
                                    content="",
                                    tool_calls=[
                                        AssistantPromptMessage.ToolCall(
                                            id=current_tool_id,
                                            type="function",
                                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                                name=current_tool_name,
                                                arguments=tool_input,
                                            ),
                                        )
                                    ],
                                ),
                            ),
                        )
                        current_tool_id = None
                        current_tool_name = None

                elif etype == "RawMessageDeltaEvent":
                    if hasattr(event, "usage") and event.usage:
                        usage_data = event.usage

                elif etype == "RawMessageStopEvent":
                    if usage_data is None:
                        final = stream.get_final_message()
                        usage_data = final.usage
                    usage = _make_usage(usage_data) if usage_data else LLMUsage.empty_usage()
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=""),
                            finish_reason="stop",
                            usage=usage,
                        ),
                    )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            client = _build_client(credentials)
            client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        except anthropic.AuthenticationError as e:
            raise CredentialsValidateFailedError(f"Authentication failed: {e}")
        except anthropic.APIConnectionError as e:
            raise CredentialsValidateFailedError(f"Cannot connect: {e}")
        except Exception as e:
            raise CredentialsValidateFailedError(str(e))

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        return sum(len(m.content) // 4 if isinstance(m.content, str) else 0 for m in prompt_messages)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        max_tokens = int(credentials.get("max_tokens_to_sample", 8192))
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.LLM,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            features=[ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL, ModelFeature.STREAM_TOOL_CALL],
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 200000,
                ModelPropertyKey.MODE: "chat",
            },
            parameter_rules=[
                ParameterRule(
                    name=DefaultParameterName.MAX_TOKENS.value,
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大 Token"),
                    type=ParameterType.INT,
                    default=8192,
                    min=1,
                    max=max_tokens,
                ),
            ],
        )

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [anthropic.APIConnectionError],
            InvokeAuthorizationError: [anthropic.AuthenticationError],
            InvokeRateLimitError: [anthropic.RateLimitError],
            InvokeBadRequestError: [anthropic.BadRequestError],
            InvokeServerUnavailableError: [anthropic.APIStatusError],
        }
