"""Client for an OpenAI-style chat completions API (OpenAI Python SDK, via Portkey)."""
import os
import re
import time
from pathlib import Path

import openai
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Load .env from the repo folder first, then from the folder above it."""
    for folder in (ROOT, ROOT.parent):
        f = folder / ".env"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k, v)


class LLMError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def provider():
    """Pick the API to call. Returns (base_url, headers, model, label)."""
    base_override = os.getenv("LLM_BASE_URL")  # used by tests
    if base_override:
        return base_override.rstrip("/"), {}, os.getenv("HUB_MODEL", "test-model"), "custom"
    pk = os.getenv("PORTKEY_API_KEY")
    if pk:
        headers = {"x-portkey-api-key": pk}
        if os.getenv("PORTKEY_PROVIDER"):
            headers["x-portkey-provider"] = os.environ["PORTKEY_PROVIDER"]
        if os.getenv("PORTKEY_VIRTUAL_KEY"):
            headers["x-portkey-virtual-key"] = os.environ["PORTKEY_VIRTUAL_KEY"]
        base = os.getenv("PORTKEY_BASE_URL", "https://api.portkey.ai/v1").rstrip("/")
        return base, headers, os.getenv("HUB_MODEL", "gpt-6-luna"), "portkey"
    ok = os.getenv("OPENAI_API_KEY")
    if ok:
        return ("https://api.openai.com/v1", {"Authorization": f"Bearer {ok}"},
                os.getenv("HUB_MODEL", "gpt-4o-mini"), "openai")
    raise LLMError("No API key found. Add PORTKEY_API_KEY or OPENAI_API_KEY to your .env file "
                   "(see .env.example).", status=500)


def has_key():
    try:
        provider()
        return True
    except LLMError:
        return False


class Usage:
    def __init__(self):
        self.input = 0
        self.cached = 0
        self.output = 0
        self.calls = 0

    def add_response(self, usage):
        usage = usage or {}
        self.calls += 1
        details = usage.get("prompt_tokens_details") or {}
        self.input += int(usage.get("prompt_tokens", 0))
        self.cached += int(details.get("cached_tokens", 0) or 0)
        self.output += int(usage.get("completion_tokens", 0))

    def merge(self, other):
        self.input += other.input
        self.cached += other.cached
        self.output += other.output
        self.calls += other.calls

    @property
    def total(self):
        return self.input + self.output

    def cost(self):
        """Dollar cost if prices are configured, else None."""
        try:
            p_in = float(os.environ["PRICE_IN_PER_M"])
            p_out = float(os.environ["PRICE_OUT_PER_M"])
        except (KeyError, ValueError):
            return None
        p_cached = float(os.getenv("PRICE_CACHED_PER_M") or p_in)
        fresh = self.input - self.cached
        return round((fresh * p_in + self.cached * p_cached + self.output * p_out) / 1e6, 6)

    def as_dict(self):
        return {"input_tokens": self.input, "cached_tokens": self.cached,
                "output_tokens": self.output, "total_tokens": self.total,
                "llm_calls": self.calls, "cost_usd": self.cost()}


def chat(messages, tools=None, max_tokens=800, temperature=0.3, timeout=60):
    """One chat completion via the OpenAI SDK. Returns (assistant_message_dict, usage_dict)."""
    base, headers, model, label = provider()
    api_key = os.getenv("PORTKEY_API_KEY") if label == "portkey" else (
        os.getenv("OPENAI_API_KEY") if label == "openai" else "test-key")
    client = OpenAI(api_key=api_key, base_url=base, default_headers=headers,
                    timeout=timeout, max_retries=1)
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    # gpt-6-luna only allows function tools on /chat/completions with reasoning off. Off also keeps
    # output tokens (and cost) predictable. Set HUB_REASONING_EFFORT to change it, or to "" to omit it.
    effort = os.getenv("HUB_REASONING_EFFORT", "none")
    # only reasoning models take this setting (gpt-4o-mini, the OPENAI_API_KEY default, rejects it)
    if effort and label != "custom" and re.match(r"(gpt-[5-9]|o\d)", model):
        body["reasoning_effort"] = effort
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    try:
        resp = client.chat.completions.create(**body)
    except openai.APIStatusError as e:
        detail = str(getattr(e, "body", "") or e.message)[:300]
        raise LLMError(f"The AI service returned an error ({e.status_code}). {detail}", status=e.status_code)
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise LLMError(f"Could not reach the AI service: {e}", status=502)
    try:
        msg = resp.choices[0].message.model_dump(exclude_none=True)
    except (AttributeError, IndexError, TypeError):
        raise LLMError("The AI service sent a reply in an unexpected format.", status=502)
    usage = resp.usage.model_dump(exclude_none=True) if resp.usage else {}
    return msg, usage


def web_search(prompt, timeout=90):
    """One web-search answer through the Responses API. Returns (text, cited_urls, usage_dict).

    Usage is converted to the chat-completions shape so Usage.add_response can count it.
    """
    base, headers, model, label = provider()
    if label == "custom":
        raise LLMError("Web search is not available with a custom test endpoint.", status=501)
    api_key = os.getenv("PORTKEY_API_KEY") if label == "portkey" else os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base, default_headers=headers, timeout=timeout, max_retries=1)
    try:
        resp = client.responses.create(model=model, input=prompt, tools=[{"type": "web_search_preview"}],
                                       include=["web_search_call.action.sources"])
    except openai.APIStatusError as e:
        raise LLMError(f"Web search failed ({e.status_code}).", status=e.status_code)
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise LLMError(f"Could not reach the web search service: {e}", status=502)
    cited = []  # pages the search really returned or opened, plus inline citations
    for item in resp.output or []:
        action = getattr(item, "action", None)
        for url in [getattr(s, "url", None) for s in (getattr(action, "sources", None) or [])] + [getattr(action, "url", None)]:
            if url and url not in cited:
                cited.append(url)
        for part in getattr(item, "content", None) or []:
            for a in getattr(part, "annotations", None) or []:
                url = getattr(a, "url", None)
                if url and url not in cited:
                    cited.append(url)
    u = resp.usage
    details = getattr(u, "input_tokens_details", None)
    usage = {"prompt_tokens": getattr(u, "input_tokens", 0) or 0,
             "completion_tokens": getattr(u, "output_tokens", 0) or 0,
             "prompt_tokens_details": {"cached_tokens": getattr(details, "cached_tokens", 0) or 0}}
    return resp.output_text or "", cited, usage


def model_label():
    try:
        _, _, model, label = provider()
        return f"{model} via {label}"
    except LLMError:
        return "no key configured"


def timed():
    return time.time()
