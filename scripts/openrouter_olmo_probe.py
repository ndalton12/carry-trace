#!/usr/bin/env python3
"""Probe OpenRouter for OLMo Think availability and run a tiny chat completion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "allenai/olmo-3-32b-think"


def main() -> int:
    """Run the OpenRouter OLMo availability probe."""
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY before running this script.", file=sys.stderr)
        return 2

    try:
        models = fetch_models(args.base_url, api_key)
    except RuntimeError as exc:
        print(f"Could not fetch OpenRouter model list: {exc}", file=sys.stderr)
        models = []

    olmo_models = find_olmo_models(models)
    print_model_candidates(olmo_models)
    if args.models_only:
        return 0

    model_id = args.model
    print(f"Using model: {model_id}")

    response = chat_completion(
        base_url=args.base_url,
        api_key=api_key,
        model_id=model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print("\nRaw response:")
    print(json.dumps(response, indent=2, sort_keys=True))
    print("\nText:")
    print(extract_text(response))
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the OpenRouter probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Explicit OpenRouter model ID to call.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only print matching OLMo models without making a chat completion call.",
    )
    parser.add_argument(
        "--prompt",
        default="What is 1124 + 922? Give only the final answer.",
        help="Prompt for the tiny completion test.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens for the tiny test call.",
    )
    return parser.parse_args()


def fetch_models(base_url: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch the OpenRouter model catalog."""
    response = request_json(
        f"{base_url.rstrip('/')}/models",
        api_key=api_key,
        payload=None,
    )
    data = response.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected models response shape: {response}")
    return [item for item in data if isinstance(item, dict)]


def find_olmo_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return model catalog entries that look like OLMo models."""
    matches = []
    for model in models:
        haystack = " ".join(
            str(model.get(key, "")) for key in ("id", "name", "description")
        ).lower()
        if "olmo" in haystack:
            matches.append(model)
    return sorted(matches, key=lambda model: str(model.get("id", "")))


def print_model_candidates(models: list[dict[str, Any]]) -> None:
    """Print concise OLMo model candidates from the OpenRouter catalog."""
    if not models:
        print("No OLMo models found in OpenRouter /models response.")
        return
    print("OLMo-like OpenRouter models:")
    for model in models:
        model_id = model.get("id")
        name = model.get("name")
        context_length = model.get("context_length")
        pricing = model.get("pricing")
        print(f"- id={model_id!r} name={name!r} context_length={context_length!r}")
        if pricing:
            print(f"  pricing={pricing}")


def chat_completion(
    base_url: str,
    api_key: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Call OpenRouter chat completions with a tiny prompt."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    return request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        payload=payload,
    )


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Send an authenticated JSON request and return the JSON response."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ndalton12/carry-trace",
            "X-Title": "carry-trace OpenRouter OLMo probe",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def extract_text(response: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
