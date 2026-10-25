# Hearing Hub

An AI agent app for Deaf and hard-of-hearing (DHH) people. *Every conversation is a campfire. Take your seat.* It is **not** a transcription app. It helps with
what happens *around* conversations, where people get stuck.

| Tool | What it does |
|---|---|
| **My hearing** | **Start here:** upload your hearing report and related pictures (PNG, JPG, WebP, PDF, DOC, DOCX, PPT, PPTX, TXT; several at once). A report-reader agent fills in your numbers, device and key facts, which the other tools then use. Or draw your audiogram (air and, optionally, bone conduction) and pick your device and its side. A free-text box lets you add facts in your own words (for example "Bonebridge only on my left ear"); every tool and the share card use them. The app tells conductive, sensorineural and mixed loss apart, shows which speech sounds you likely catch, then writes a plain-language **share card** ("how I hear") for professors, coworkers, family and doctors. |
| **Phone calls** | Say what you need to get done. For every organization you name, the app finds its official website, written channels, email and phone, and checks them on the organization's own site. You get a written-route plan, a message ready to send, a backup call script with likely questions, and a follow-up email that puts the answer in writing. |
| **Places** | Add a photo, a description or pasted reviews of a restaurant, classroom or meeting room. You get a 1-10 listening-effort score, the best seat, and what to ask the venue. |
| **The circle** | Group conversations ("dinner table syndrome"). List who will be there, with notes like "soft voice" or "talks most". Code works out who sits where (people you most need on your hearing side or across where you can see their face, noise behind you) and draws the seating map; the agent writes a friendly message to the group, lines to catch up without stopping the talk, and a follow-up. |
| **What's new** | Recent hearing-tech news (fully implantable cochlear implants, OTOF gene therapy, bone-conduction processors, AI hearing aids, hearing glasses), each with sources, why it matters and its limits. Stories that fit your device are marked "For you". A hand-checked list costs no tokens; **Search for newer news** runs a live web search and keeps only stories whose link the search really opened (cached 24 hours). |

## Run it (about 3 minutes, tested from a fresh clone)

You need **Python 3.9 or newer**. Two packages are installed from `requirements.txt`: the OpenAI Python SDK
(AI calls go through Portkey) and PyMuPDF (reads PDF reports).

```bash
git clone <this repo>
cd hearing-hub
python -m venv .venv                     # optional but recommended
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open **http://localhost:8000**.

**API keys:** the app reads `PORTKEY_API_KEY` (course key, model `gpt-6-luna`) or `OPENAI_API_KEY`.
It looks for `.env` **in this folder first, then in the folder one level above it**, so a shared `.env` next to the
cloned repo works with no copying. To use your own, `cp .env.example .env` and paste a key.
`.env.example` lists every setting; all except the key are optional. Without a key the audiogram chart and the
news list still work, and the AI tools show a clear message.

Tested on 2026-10-25: fresh clone into a folder with the `.env` one level up, new virtual environment,
`pip install`, all 79 tests and every tool live on `gpt-6-luna`. Setup took under 3 minutes.

**Quick demo path (for graders, about 5 minutes):**
1. **My hearing.** In **Step 1** press **Fill in an example (Deep's real report)**. It loads the real audiogram of
   the app's author (bilateral atresia, a Bonebridge implant on the left). Or upload your own audiogram photo or PDF
   and press **Read my report**. Then press **Make my share card**.
2. **Phone calls.** Press **Fill in an example** (a real case: a Yale Pharmacy appointment by phone), then
   **Plan my call**. The "Where to reach them in writing" card shows checked contacts. Try naming any company, for
   example "Starkey, Signia and MED-EL": each gets its own verified card (about 15 seconds per new organization).
3. **Places.** Press **Load sample reviews (one has a hidden attack)**, then **Check this place**. The
   "Safety filter" banner shows the attack being removed before the AI sees it.
4. **The circle.** Press **Fill in an example**, then **Plan my seat** for a seating map of a family dinner.
5. **What's new.** Recent hearing tech with sources. **Search for newer news** runs a live web search.

Watch the campfire at the top: the "You" figure walks into the circle as each tool is used. Under every result,
**How this was made** shows the tool calls, tokens (input, cached, output) and time.

## How it works

```
                          +----------------+
 "I have to call     ---> |  Router agent  |  picks a tool (no tools itself, ~400 tokens)
  my pharmacy"            +-------+--------+
                                  |
     +-------------+--------------+--------------+---------------+
     v             v              v              v               v
 Report reader  Profile agent  Call agent     Place agent     Circle agent
 (vision:       tool:          tool:          tool:           tool:
  photos, PDF,  analyze_       find_call_     score_          plan_seating
  Word, PPT)    audiogram      alternatives   acoustics
     |                         + web search -> verify contacts on the official site
     v
 numbers checked in code -> the other tools use them        News: hand-checked list + verified live search
```

Design choices worth knowing:

* **Tools are plain code, not model guesses.** Audiogram analysis (including conductive vs sensorineural loss),
  the 1-10 acoustics score, the seating plan and the contact checks are deterministic Python.
* **The model cannot change your numbers.** Tools are bound to the validated input, so the model calls them with
  no arguments. Numbers read from a report are range-checked in code before they are used.
* **The score and the seats are never invented.** Only features the model backed with evidence count in
  *Places*, and *The circle* seats people in code; the AI only writes the messages.
* **Contacts are verified, not generated.** A web search proposes contacts; code keeps only those found on the
  organization's own site, and replaces any other email the model writes.
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

79 tests cover the audiogram math (including loss type), scoring, seating, report reading in every file
format, contact and news verification, injection defenses, budget and rate limits, and end-to-end runs of the real
server against a **fake model** (`tests/fake_llm.py`) that deliberately misbehaves (misread reports, invented
emails, hidden instructions) to prove the guards catch it.
The fake model is only for testing the plumbing. It produces canned text, not real answers.

## Limits (honest)

* Speech-sound levels are approximate teaching values, not clinical data. The app says so and never diagnoses.
* The call tool lists US relay services (711, IP CTS, VRS). Verify availability for your location.
* Photo analysis depends on whether the model accepts images; if not, the app says so and uses text only.
* The 1-10 score is a transparent heuristic, not a measurement. Validate it against places you know.
* Reading a phone photo of a paper audiogram is close but not exact (within 5-10 dB in testing), so the app always
  asks the person to compare the numbers with the original.
* A new organization lookup takes about 15 seconds and ~15k tokens; results are cached for 7 days.
* Older .doc and .ppt files give text only; save them as PDF or DOCX for pictures.

## Layout

```
server.py             HTTP server and API (standard library only)
hearing/tools.py      deterministic tools: audiogram, loss type, acoustics score, seating, call routes, contacts list
hearing/agents.py     router + report, profile, call, place and circle agents; tool loop; JSON and guard checks
hearing/documents.py  reads uploaded PDF, Word, PowerPoint, text and picture files (no AI)
hearing/contacts.py   finds an organization's official contacts by web search and verifies them on its own site
hearing/news.py       hand-checked hearing-tech news + verified live search
hearing/safety.py     injection defense, personal-data stripping, budget, rate limit
hearing/llm.py        API client (OpenAI SDK via Portkey), token accounting, .env loading
static/index.html     the whole UI
static/examples/      the example audiogram (personal details cropped out)
tests/                unit tests, a fake model, end-to-end tests
```
