#!/usr/bin/env python3
"""Hearing Hub server. Standard library only: run with `python server.py`."""
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hearing import llm
llm.load_env()  # before anything reads os.environ

from hearing import agents, documents, news, safety, tools  # noqa: E402

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "static" / "index.html"
# The only other files the server will send (no directory browsing, no user paths).
STATIC_FILES = {"/static/examples/example-audiogram.jpg": (ROOT / "static" / "examples" / "example-audiogram.jpg",
                                                         "image/jpeg")}
MAX_BODY = 6_500_000
MAX_REPORT_BODY = 36_000_000  # several documents, base64-encoded
IMAGE_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$")

BUDGET = safety.make_budget()
LIMITER = safety.make_limiter()
SESSION = llm.Usage()


def hearing_from(body):
    """Hearing inputs for tools.analyze_audiogram. An old 'aids' checkbox maps to hearing aids."""
    device = body.get("device")
    if not device and body.get("aids"):
        device = "hearing_aids"
    return {"right": body.get("right") or {}, "left": body.get("left") or {},
            "right_bc": body.get("right_bc") or {}, "left_bc": body.get("left_bc") or {},
            "device": device or "none", "device_side": body.get("device_side")}


def finish(module, t0, result, trace, usage, flags=0, extra=None):
    ms = round((time.time() - t0) * 1000)
    agents.log_run(module, True, usage, trace, ms, flags)
    SESSION.merge(usage)
    out = {"result": result, "run": {"trace": trace, "usage": usage.as_dict(), "ms": ms,
                                    "model": llm.model_label()}}
    out["session"] = session_info()
    if extra:
        out["warnings"] = extra
    return out


def session_info():
    d = SESSION.as_dict()
    d["budget_remaining"] = BUDGET.remaining
    d["budget_cap"] = BUDGET.cap
    return d


# ------------------------------------------------------------------ handlers

def report_notes_from(body):
    """Report facts the browser kept from an earlier upload. Re-cleaned: never trust the client."""
    notes, _ = safety.clean_notes(body.get("report_notes"), max_items=8, max_len=300)
    return notes


def h_report(body, client):
    t0 = time.time()
    try:
        docs = documents.read_files(body.get("files"))
    except documents.DocumentError as e:
        raise safety.SafetyError(str(e))
    notes, _ = safety.clean_notes([ln for ln in str(body.get("my_notes") or "").splitlines()], max_items=6, max_len=200)
    report, warnings, trace, usage = agents.read_report(docs["texts"], docs["images"], BUDGET, notes)
    trace.insert(0, {"agent": "report", "step": 0, "tool": "read_documents",
                     "args": {"files": [f["name"] for f in docs["files"]]},
                     "note": f"read {len(docs['files'])} file(s) in code: {len(docs['images'])} image(s), "
                             f"{sum(len(t['text']) for t in docs['texts'])} characters of text"})
    analysis = None
    if report["found_audiogram"]:
        try:
            analysis = tools.analyze_audiogram(report["right_air"], report["left_air"], report["right_bone"],
                                               report["left_bone"], report["device"], report["device_side"])
        except tools.AudiogramError:
            analysis = None  # fewer than 3 readable points; the person fills in the rest
    return finish("report", t0, {"report": report, "files": docs["files"], "analysis": analysis},
                  trace, usage, flags=sum(w["count"] for w in warnings), extra=warnings)


def h_analyze(body, client):
    return {"analysis": tools.analyze_audiogram(**hearing_from(body))}  # free: no AI call


def h_route(body, client):
    t0 = time.time()
    text = safety.clean_field("Your request", body.get("text"), 500, required=True)
    obj, trace, usage = agents.route(text, BUDGET)
    return finish("router", t0, obj, trace, usage)


def h_profile(body, client):
    t0 = time.time()
    name = safety.clean_field("First name", body.get("first_name"), 40)
    res, trace, usage = agents.profile_card(hearing_from(body), name, body.get("lipreading"), BUDGET,
                                            report_notes_from(body))
    return finish("profile", t0, res, trace, usage)


def profile_context(body):
    """Recompute the profile line from numbers (never trust client-written summaries)."""
    hearing = hearing_from(body)
    if not (hearing["right"] or hearing["left"]):
        return None, None
    try:
        analysis = tools.analyze_audiogram(**hearing)
    except tools.AudiogramError:
        return None, None
    return analysis, tools.profile_summary(analysis)


def h_call(body, client):
    t0 = time.time()
    goal = safety.clean_field("What you need to get done", body.get("goal"), 800, required=True)
    name = safety.clean_field("First name", body.get("first_name"), 40)
    org = body.get("org_type") if body.get("org_type") in tools.ORG_TYPES else "other"
    urgency = body.get("urgency") if body.get("urgency") in ("today", "this week", "no rush") else "this week"
    comfort = body.get("comfort") if body.get("comfort") in (
        "cannot use the phone", "can use a captioned phone or relay", "can talk but struggle to hear") \
        else "can talk but struggle to hear"
    _, line = profile_context(body)
    notes = report_notes_from(body)
    if notes:
        line = (line or "Hearing numbers not entered") + ". From their hearing report: " + "; ".join(notes)
    obj, trace, usage = agents.call_kit(goal, org, urgency, comfort, name, line, BUDGET)
    return finish("call", t0, obj, trace, usage)


def h_place(body, client):
    t0 = time.time()
    venue = safety.clean_field("Type of place", body.get("venue_type"), 60, required=True)
    purpose = safety.clean_field("What is happening there", body.get("purpose"), 200)
    desc = safety.clean_field("Description", body.get("description"), 1500)
    raw_reviews = body.get("reviews") or ""
    if not isinstance(raw_reviews, str) or len(raw_reviews) > 6000:
        raise safety.SafetyError("Pasted reviews must be text under 6,000 characters.")
    reviews, removed = safety.neutralize(raw_reviews)
    reviews = safety.escape_tags(reviews)
    image = body.get("image")
    if image:
        if not isinstance(image, str) or len(image) > 5_500_000 or not IMAGE_RE.match(image):
            raise safety.SafetyError("The photo must be a JPEG, PNG or WebP under 4 MB.")
    if not (desc or reviews or image):
        raise safety.SafetyError("Add a description, a photo, or some reviews so there is something to assess.")
    analysis, _ = profile_context(body)
    obj, trace, usage = agents.place_scout(venue, purpose, desc, reviews, image, analysis,
                                           bool(body.get("lipreading")), BUDGET, report_notes_from(body))
    warnings = []
    if removed:
        warnings.append({"type": "injection_removed", "count": len(removed), "snippets": removed,
                         "message": f"Removed {len(removed)} sentence(s) from the pasted text that looked "
                                    "like instructions to the AI. The rest was used."})
    return finish("place", t0, obj, trace, usage, flags=len(removed), extra=warnings)


def h_circle(body, client):
    t0 = time.time()
    event = safety.clean_field("What is the gathering", body.get("event"), 200, required=True)
    raw = body.get("people") or ""
    if not isinstance(raw, str):
        raise safety.SafetyError("People must be text, one per line.")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise safety.SafetyError("Add at least one person who will be there.")
    if len(lines) > tools.MAX_PEOPLE:
        raise safety.SafetyError(f"Add up to {tools.MAX_PEOPLE} people.")
    lines = [safety.clean_field("A person in the list", ln, 90) for ln in lines]
    people = tools.parse_people(lines)
    shape = body.get("shape") if body.get("shape") in tools.SHAPES else "round"
    noise = [x for x in (body.get("noise") or []) if x in tools.NOISES] if isinstance(body.get("noise"), list) else []
    tone = body.get("tone") if body.get("tone") in ("casual", "warm", "professional") else "warm"
    analysis, _ = profile_context(body)
    obj, trace, usage = agents.circle_plan(event, shape, people, noise, tone, analysis,
                                           bool(body.get("lipreading")), report_notes_from(body), BUDGET)
    return finish("circle", t0, obj, trace, usage)


def h_news(body, client):
    t0 = time.time()
    device = body.get("device") if body.get("device") in tools.DEVICES else None
    payload, web_usage = news.search_newer(device, BUDGET)
    usage = llm.Usage()
    if web_usage:
        usage.add_response(web_usage)
    trace = [{"agent": "news", "step": 1, "tool": "web_search",
              "note": ("from the 24-hour news cache (no tokens used)" if payload.get("cached") else
                       f"kept {len(payload['items'])} stor(ies) whose link the search returned; "
                       f"dropped {payload.get('dropped', 0)}")}]
    return finish("news", t0, payload, trace, usage)


POST_ROUTES = {"/api/analyze": (h_analyze, False), "/api/route": (h_route, True),
               "/api/news": (h_news, True),
               "/api/circle": (h_circle, True),
               "/api/report": (h_report, True),
               "/api/profile": (h_profile, True), "/api/call": (h_call, True),
               "/api/place": (h_place, True)}


class Handler(BaseHTTPRequestHandler):
    server_version = "HearingHub/0.1"

    def log_message(self, fmt, *args):
        if os.getenv("HUB_VERBOSE"):
            super().log_message(fmt, *args)

    def _send(self, status, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        if path in STATIC_FILES:
            file, ctype = STATIC_FILES[path]
            return self._send(200, file.read_bytes(), ctype)
        if path == "/api/news":
            return self._send(200, news.curated())  # hand-checked list: no AI, no tokens
        if path == "/api/config":
            return self._send(200, {"model": llm.model_label(), "has_key": llm.has_key(),
                                    "session": session_info()})
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        path = self.path.split("?")[0]
        route = POST_ROUTES.get(path)
        if not route:
            return self._send(404, {"error": "Not found."})
        handler, uses_ai = route
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > (MAX_REPORT_BODY if path == "/api/report" else MAX_BODY):
                raise safety.SafetyError("That upload is too large.", 413)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise safety.SafetyError("Send a JSON object.")
            client = self.client_address[0]
            if uses_ai:
                if not LIMITER.allow(client):
                    raise safety.SafetyError("Too many requests. Wait a minute and try again.", 429)
                if not llm.has_key():
                    raise safety.SafetyError("No API key found. Add one to .env (see .env.example).", 500)
                BUDGET.check()
            self._send(200, handler(body, client))
        except safety.SafetyError as e:
            self._send(e.status, {"error": str(e), "kind": "refused"})
        except tools.AudiogramError as e:
            self._send(400, {"error": str(e), "kind": "input"})
        except json.JSONDecodeError:
            self._send(400, {"error": "Could not read that request.", "kind": "input"})
        except llm.LLMError as e:
            agents.log_run(path, False, None, [], 0, error=str(e)[:120])
            self._send(e.status if e.status and e.status >= 400 else 502,
                       {"error": str(e), "kind": "service"})
        except agents.AgentError as e:
            agents.log_run(path, False, None, [], 0, error=str(e)[:120])
            self._send(e.status, {"error": str(e), "kind": "agent"})
        except Exception as e:  # last resort: never leak a traceback to the browser
            print("Unexpected error:", repr(e), file=sys.stderr)
            self._send(500, {"error": "Something went wrong on the server.", "kind": "server"})


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    srv = make_server("127.0.0.1", port)
    print(f"Hearing Hub running at http://localhost:{port}  ({llm.model_label()})")
    if not llm.has_key():
        print("Note: no API key found. The audiogram preview works, but AI features need a key in .env.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
