# Hearing Hub

An AI agent app for Deaf and hard-of-hearing (DHH) people. *Every conversation is a campfire. Take your seat.* It is **not** a transcription app. It helps with
what happens *around* conversations, where people get stuck.

| Tool | What it does |
|---|---|
| **My hearing** | **Start here:** upload your hearing report and related pictures (PNG, JPG, WebP, PDF, DOC, DOCX, PPT, PPTX, TXT; several at once). A report-reader agent fills in your numbers, device and key facts, which the other two tools then use. Or draw your audiogram (air and, optionally, bone conduction) and pick your device and its side. A free-text box lets you add facts in your own words (for example "Bonebridge only on my left ear"); every tool and the share card use them. The app tells conductive, sensorineural and mixed loss apart, shows which speech sounds you likely catch, then writes a plain-language "how I hear" card you can share with teachers, coworkers and family. |
| **Phone calls** | Say what you need to get done. You get a written-route plan, a message ready to send, a backup call script with likely questions, and a follow-up email that puts the answer in writing. |
| **Places** | Add a photo, a description or pasted reviews of a restaurant, classroom or meeting room. You get a 1-10 listening-effort score, the best seat, and what to ask the venue. |
| **The circle** | Group conversations ("dinner table syndrome"). List who will be there, with notes like "soft voice" or "talks most". Code works out who sits where (people you most need on your hearing side or across where you can see their face, noise behind you) and draws the seating map; the agent writes a friendly message to the group, lines to catch up without stopping the talk, and a follow-up. |
| **What's new** | Recent hearing-tech news (fully implantable cochlear implants, OTOF gene therapy, bone-conduction processors, AI hearing aids, hearing glasses), each with sources, why it matters and its limits. Stories that fit your device are marked "For you". A hand-checked list costs no tokens; **Search for newer news** runs a live web search and keeps only stories whose link the search really opened (cached 24 hours). |

## Run it (about 2 minutes)

You need Python 3.9+ and one package, the OpenAI Python SDK (AI calls go through Portkey with it).

```bash
git clone <this repo>
cd hearing-hub
pip install -r requirements.txt
cp .env.example .env        # then paste a key into .env
python server.py
```

Open **http://localhost:8000**.

**API keys:** put `PORTKEY_API_KEY` (course key, model `gpt-6-luna`) or `OPENAI_API_KEY` in `.env`.
The app looks for `.env` in this folder first, then in the folder one level above it.
`.env.example` lists every setting. Without a key the audiogram chart still works; the AI tools show a clear message.

**Quick demo path (for graders):**
1. Open *My hearing*. Upload an audiogram photo or PDF in **Step 1** and press **Read my report** (or keep the example audiogram), then press **Write my "how I hear" card**.
2. Open *Phone calls*, press **Fill in an example**, then **Plan my call**.
3. Open *Places*, press **Load sample reviews (one has a hidden attack)**, then **Check this place**.
   The "Safety filter" banner shows the hidden attack being removed before the AI sees it.

Under every result, **How this was made** shows the tool calls, tokens (input, cached, output) and time.

## How it works

```
                      +----------------+
 "I have to call      |  Router agent  |  picks a specialist (no tools, ~150 output tokens)
  my dentist"  -----> +-------+--------+
                              |
        +---------------------+----------------------+
        v                     v                      v
 Profile agent           Call agent             Place agent
 tool: analyze_audiogram  tool: find_call_       tool: score_acoustics
 (reads verified numbers)      alternatives      (+ optional photo input)
```

Design choices worth knowing:

* **Tools are plain code, not model guesses.** Audiogram analysis, the 1-10 acoustics score and the
  call-alternatives lookup are deterministic Python (`hearing/tools.py`).
* **The model cannot change your numbers.** The profile tool is bound to your validated input, so the model
  calls it with no arguments.
* **The score is never invented.** In *Places*, only features the model backed with evidence count, and the app
  computes the score from exactly those. If the model skips the tool or counts unsupported features, the app
  corrects it and records that in the run trace.
* **The chart costs zero tokens.** Live speech-sound analysis runs in code.

## Safety

| Risk | Defense | Where |
|---|---|---|
| Direct prompt injection ("ignore your instructions") | Pattern scan on every typed field, with Unicode normalization (catches zero-width and full-width tricks). Request is refused **before any AI call**. | `safety.clean_field` |
| Indirect injection (instructions hidden in pasted reviews) | Offending sentences are stripped; useful text is kept; the user is told what was removed. Text is also wrapped in `<untrusted_reviews>` tags and the model is told it is data. | `safety.neutralize` |
| Tag break-out | `<` and `>` are escaped in user text. | `safety.escape_tags` |
| Invented numbers or facts | Tool-computed scores; "never invent facts" rules; `[bracketed placeholders]` for unknown personal details. | `agents.py` |
| Runaway agents | Max tool-use steps (`HUB_MAX_STEPS`), duplicate tool-call blocking, one JSON repair attempt only. | `agents.run_agent` |
| Overspending | Server-wide token budget (`HUB_TOKEN_BUDGET`), per-visitor rate limit (`HUB_REQ_PER_MIN`), per-request `max_tokens`, upload size limits. | `safety.Budget`, `RateLimiter` |
| Privacy | Logs in `logs/runs.jsonl` record token counts and tool names only, never what you typed. Audiogram is stored in your own browser. | `agents.log_run` |
| Unsafe rendering | All model output is HTML-escaped in the browser. | `esc()` in `index.html` |
| Hidden instructions in uploaded files | Document text is filtered like reviews and wrapped in `<untrusted_document>` tags; extracted notes are re-filtered on every request. | `agents.read_report`, `safety.clean_notes` |
| Misread reports | Every number the model reads is range-checked and snapped to 5 dB in code. Statements that contradict the numbers (wrong loss type, degree, or "no loss") are moved to "Please double-check" and never reach the other tools. The person can remove any note. | `tools.clean_report_ear`, `agents._drop_conflicting_text` |
| Bad or hostile files | Up to 8 files, 10 MB each, 25 MB total; file contents must match the extension; zip-bomb guard for DOCX/PPTX; files are read in memory and never saved. | `documents.read_files` |
| Wrong or scam contact details | When the person names an organization, the app searches the web, then checks in code: the website must be a domain the search really visited (no look-alikes), each email and phone must come from a page on that domain, and the app downloads that page to confirm it is written there (✓). Details on pages it cannot open are labeled "listed, not double-checked"; details missing from a readable page are dropped. Any other email the model writes is replaced. Lookups are cached for 7 days. | `contacts.verify`, `agents._scrub_unknown_emails` |
| Personal data in reports | The model is told never to copy names, addresses or ID numbers, and code strips ID/phone numbers, emails and titled names from notes. | `safety.strip_pii` |

## Tests (no API key needed)

```bash
python -m unittest discover -s tests -t .
```

37 tests cover the audiogram math, scoring, injection defenses, budget and rate limits, and an end-to-end run of
the real server against a **fake model** (`tests/fake_llm.py`) to check the tool loop, safety layer and fallbacks.
The fake model is only for testing the plumbing. It produces canned text, not real answers.

## Limits (honest)

* Speech-sound levels are approximate teaching values, not clinical data. The app says so and never diagnoses.
* The call tool lists US relay services (711, IP CTS, VRS). Verify availability for your location.
* Photo analysis depends on whether the model accepts images; if not, the app says so and uses text only.
* The 1-10 score is a transparent heuristic, not a measurement. Validate it against places you know.

## Layout

```
server.py            HTTP server and API (standard library only)
hearing/tools.py     deterministic tools
hearing/agents.py    router + three agents, tool loop, JSON checks
hearing/safety.py    injection defense, budget, rate limit
hearing/llm.py       API client, token accounting, .env loading
hearing/contacts.py  finds an organization's official contacts by web search and verifies them on its own site
hearing/documents.py reads uploaded PDF, Word, PowerPoint, text and picture files (no AI)
static/index.html    the whole UI
tests/               unit tests, fake model, end-to-end tests
```
