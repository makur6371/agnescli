"""HTTP client wrapping the Agnes AI API with retry logic."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from typing import Any

import httpx

BASE_URL = "https://apihub.agnes-ai.com/v1"
TIMEOUT = 120
MAX_RETRIES = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class AgnesAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"Agnes API error {status_code}: {message}")


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        body = resp.json()
        msg = body.get("error", {}).get("message", "") or body.get("detail", "") or resp.text[:500]
    except Exception:
        msg = resp.text[:500]
    raise AgnesAPIError(resp.status_code, msg)


def _retry(fn, *args, **kwargs):
    """Execute with exponential backoff retry for transient errors."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except AgnesAPIError as e:
            if e.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES - 1:
                raise
            last_exc = e
            delay = 2**attempt
            print(
                f"  [retry] {e.status_code} error, retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore


class AgnesClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=TIMEOUT,
        )

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "agnes-2.0-flash",
        stream: bool = False,
        thinking: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        if stream:
            return self._stream_chat(body)
        return _retry(self._post_json, "/chat/completions", body)

    def _post_json(self, path: str, body: dict) -> dict[str, Any]:
        resp = self._http.post(path, json=body)
        _raise_for_status(resp)
        return resp.json()

    def _stream_chat(self, body: dict) -> Iterator[dict[str, Any]]:
        with self._http.stream("POST", "/chat/completions", json=body) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                yield json.loads(payload)

    # ── Image ─────────────────────────────────────────────────────────

    def image_generate(
        self,
        prompt: str,
        *,
        model: str = "agnes-image-2.1-flash",
        size: str = "1024x768",
        input_images: list[str] | None = None,
        response_format: str = "url",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }
        if input_images:
            body["extra_body"] = {
                "image": input_images,
                "response_format": response_format,
            }
        return _retry(self._post_json, "/images/generations", body)

    # ── Video ─────────────────────────────────────────────────────────

    def video_create(
        self,
        prompt: str,
        *,
        model: str = "agnes-video-v2.0",
        image: str | None = None,
        input_images: list[str] | None = None,
        keyframes: bool = False,
        width: int = 1152,
        height: int = 768,
        num_frames: int = 121,
        frame_rate: int = 24,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }
        if image and not input_images:
            body["image"] = image
        if input_images:
            extra: dict[str, Any] = {"image": input_images}
            if keyframes:
                extra["mode"] = "keyframes"
            body["extra_body"] = extra
        if seed is not None:
            body["seed"] = seed
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        return _retry(self._post_json, "/videos", body)

    def video_status(self, task_id: str) -> dict[str, Any]:
        resp = self._http.get(f"/videos/{task_id}")
        _raise_for_status(resp)
        return resp.json()

    def download(self, url: str, dest: str) -> None:
        with self._http.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
