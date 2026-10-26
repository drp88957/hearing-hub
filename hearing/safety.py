"""Safety helpers: prompt-injection defense, spending limits, output checks."""
import json
import os
import re
import threading
import time
import unicodedata


class SafetyError(Exception):
    """Raised for inputs we refuse. The message is safe to show to the user."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ injection

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|your|these)[^.\n]{0,40}(instruction|prompt|rule|guideline)s?",
    r"disregard (all |any |the )?(previous|prior|above|earlier|your)[^.\n]{0,40}",
    r"forget (everything|all|your|the (above|previous))[^.\n]{0,30}",
    r"(reveal|show|print|repeat|output|leak|tell me) (me )?(your|the) (system |hidden |initial |secret )?(prompt|instructions)",
    r"system prompt",
    r"you are now\b",
    r"developer mode|jailbreak",
    r"act as (an? )?(unrestricted|different|new|evil)",
    r"override (your|the) (rules|instructions|safety|guidelines)",
    r"new instructions?\s*:",
    r"api[ _-]?key|portkey|\.env\b|secret key",
    r"</?\s*(system|assistant|untrusted_reviews|user_input|tool)\b",
    r"(always|must) (recommend|rate|score)[^.\n]{0,40}(10|perfect|best|safe)",
    r"(rate|score) (it|this|the place)[^.\n]{0,20}\b10\b",
]
# "DAN" (the jailbreak persona) only in capitals, so a person called Dan is not refused
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS] + [re.compile(r"\bDAN\b")]
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def _normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def scan(text):
    """Return the list of pattern hits found in text."""
    t = _normalize(text)
    return [p.pattern for p in _COMPILED if p.search(t)]


def neutralize(text):
    """Remove sentences that look like instructions to an AI.

    Returns (clean_text, removed_snippets). Used for untrusted text such as
    pasted reviews, where we keep the useful parts instead of rejecting all.
    """
    t = _normalize(text)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", t)
    kept, removed = [], []
    for s in sentences:
        if not s.strip():
            continue
        if any(p.search(s) for p in _COMPILED):
            removed.append(s.strip()[:90])
        else:
            kept.append(s.strip())
    return "\n".join(kept), removed


def escape_tags(text):
    """Stop text from closing our <untrusted_*> wrappers."""
    return text.replace("<", "\u2039").replace(">", "\u203a")


def clean_field(name, value, max_len, required=False):
    """Validate a free-text field the person typed. Rejects injection attempts."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise SafetyError(f"{name} must be text.")
    value = _normalize(value).strip()
    if required and not value:
        raise SafetyError(f"Please fill in: {name}.")
    if len(value) > max_len:
        raise SafetyError(f"{name} is too long. Keep it under {max_len} characters.")
    if scan(value):
        raise SafetyError(
            f"{name} contains wording that looks like an instruction to the AI "
            "(for example, 'ignore your instructions'). Please rephrase what you need."
        )
    return escape_tags(value)


# ------------------------------------------------------------- spending limits

class BudgetExceeded(SafetyError):
    def __init__(self):
        super().__init__(
            "The token budget for this server is used up, so AI requests are paused. "
            "Restart the app or raise HUB_TOKEN_BUDGET.", status=429)


class Budget:
    def __init__(self, cap):
        self.cap = int(cap)
        self.used = 0
        self._lock = threading.Lock()

    def check(self):
        with self._lock:
            if self.used >= self.cap:
                raise BudgetExceeded()

    def add(self, tokens):
        with self._lock:
            self.used += int(tokens)

    @property
    def remaining(self):
        return max(0, self.cap - self.used)


class RateLimiter:
    def __init__(self, per_minute):
        self.per_minute = int(per_minute)
        self._hits = {}
        self._lock = threading.Lock()

    def allow(self, key):
        now = time.time()
        with self._lock:
            if len(self._hits) > 5000:  # forget visitors with no hits in the last minute
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < 60}
            hits = [t for t in self._hits.get(key, []) if now - t < 60]
            if len(hits) >= self.per_minute:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


def make_budget():
    return Budget(os.getenv("HUB_TOKEN_BUDGET", "400000"))  # a web contact lookup is ~15k tokens


def make_limiter():
    return RateLimiter(os.getenv("HUB_REQ_PER_MIN", "12"))


# --------------------------------------------------------------- output checks

def extract_json(text):
    """Pull the first JSON object out of a model reply, tolerating code fences."""
    if not text:
        raise ValueError("empty reply")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def validate_keys(obj, required):
    """Return a list of problems. required maps key -> expected python type."""
    problems = []
    for k, typ in required.items():
        if k not in obj:
            problems.append(f"missing '{k}'")
        elif not isinstance(obj[k], typ):
            problems.append(f"'{k}' should be {typ.__name__}")
    return problems


# ------------------------------------------------------------ report notes

_PII = [
    re.compile(r"\b\d[\d\s\-/.]{5,}\d\b"),                  # ID, phone, registration numbers
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                   # email addresses
    re.compile(r"\b(?:mr|mrs|ms|shri|smt|kum)\.?\s+[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*", re.I),  # titled names
]


_DATE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}")
_DB_RANGE = re.compile(r"\d{1,3}\s*-\s*\d{1,3}")
_HZ = {125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000}


def _hearing_numbers(t):
    """True for runs like '60 65 65 70' (dB in steps of 5) or '250-8000' (test frequencies)."""
    nums = re.findall(r"\d+", t)
    return all(int(n) in _HZ or (len(n) <= 3 and int(n) % 5 == 0 and int(n) <= 120) for n in nums)


def strip_pii(text):
    def sub(m):
        t = m.group(0).strip()
        # keep dates (30-04-2015), dB values and test frequencies; remove ID and phone numbers
        keep = _DATE.fullmatch(t) or _DB_RANGE.fullmatch(t) or _hearing_numbers(t)
        return m.group(0) if keep else "[removed]"
    for p in _PII:
        text = p.sub(sub, text)
    return text


def clean_notes(items, max_items=6, max_len=200):
    """Clean short notes that came from an uploaded report (via the model or the browser).

    Drops anything that looks like an instruction to the AI, removes ID numbers, emails and
    titled names, escapes tags and caps the size. Returns (notes, removed_count).
    """
    if not isinstance(items, list):
        return [], 0
    notes, removed = [], 0
    for item in items[:max_items * 2]:
        if not isinstance(item, str):
            continue
        t = _normalize(item).strip()
        if not t:
            continue
        if scan(t):
            removed += 1
            continue
        notes.append(escape_tags(strip_pii(t))[:max_len])
        if len(notes) == max_items:
            break
    return notes, removed
