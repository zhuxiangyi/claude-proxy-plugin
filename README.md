# Claude Proxy Plugin for Dify

A Dify model provider plugin that connects to Anthropic Claude models through any custom proxy endpoint that implements the native Anthropic Messages API.

## Setup

1. Install the plugin in your Dify instance.
2. Go to **Settings → Model Providers → Claude Proxy** and click **Add Model**.
3. Fill in the following fields:

| Field | Description |
|-------|-------------|
| **Model Name** | The model identifier your proxy expects (e.g. `claude-sonnet-4-6`) |
| **API Key** | Your API key for the proxy service |
| **API Base URL** | The base URL of your proxy (e.g. `https://your-proxy.example.com/v1`) |
| **Max Tokens** | Maximum output tokens (default: 8192) |

## Requirements

- A proxy service that implements the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- An API key issued by your proxy provider

## Features

- Supports any proxy endpoint compatible with the native Anthropic Messages API
- Streaming and non-streaming responses
- Tool / function calling
- Configurable max tokens per model
- Context window: 200,000 tokens

## Repository

https://github.com/zhuxiangyi/claude-proxy-plugin
