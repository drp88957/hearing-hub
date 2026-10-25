"""Hearing technology news.

CURATED is a hand-checked list (sources read on the date in `checked`). It costs no tokens.
`search_newer()` asks the model to search the web for newer stories; code keeps only stories whose link
the search really returned, labels trusted outlets, and caches the result for a day.
"""
import json
import re
import threading
import time
import urllib.parse
from pathlib import Path

from . import llm, safety

CHECKED = "2026-10-25"

CURATED = [
    {"id": "tici-unico", "tag": "Cochlear implants", "date": "2026-09-29",
     "title": "MED-EL launches TICI UNICO, a fully implantable \"invisible\" cochlear implant",
     "summary": "Every part, including the microphone and a rechargeable battery, sits under the skin, so there is "
                "no external processor. MED-EL calls it the first commercially available totally implantable "
                "cochlear implant, and says users keep hearing while it charges.",
     "why": "Hearing while sleeping, swimming or showering, and no visible device.",
     "caveat": "CE-marked and rolling out in several European countries; not FDA-approved yet. The published "
               "feasibility study had six participants, so long-term data is still coming.",
     "for_devices": ["cochlear_implant"],
     "sources": [["Hearing Review", "https://hearingreview.com/hearing-products/implants-bone-conduction/cochlear-implants/med-el-launches-tici-unico-a-totally-invisible-cochlear-implant"],
                 ["IEEE Spectrum", "https://spectrum.ieee.org/fully-implantable-cochlear-implant"],
                 ["MED-EL: feasibility study (2025)", "https://www.medel.com/press-room/press-details/2025/02/06/the-future-of-cochlear-implants--promising-feasibility-study-results-for-med-el-s-totally-implantable-cochlear-implant-(tici)"]]},
    {"id": "otof-gene-therapy", "tag": "Gene therapy", "date": "2026-04",
     "title": "Gene therapy restores hearing in children and adults born deaf (OTOF)",
     "summary": "A multicentre trial published in Nature treated 42 people, aged under 1 to 32, who were born deaf "
                "because of changes in the OTOF gene. About 90% recovered hearing, and gains held up over "
                "2.5 years of follow-up. Results were strongest in people 18 and younger.",
     "why": "The first treatment that fixes the cause of a genetic deafness instead of working around it.",
     "caveat": "Only for OTOF-related deafness, a small share of genetic hearing loss. Trials are small and "
               "single-arm, and it is not an approved treatment yet.",
     "for_devices": [],
     "sources": [["Nature (2026)", "https://www.nature.com/articles/s41586-026-10393-y"],
                 ["Harvard Gazette", "https://news.harvard.edu/gazette/story/2026/04/hearing-breakthrough-holds-up/"],
                 ["Mass General Brigham", "https://www.massgeneralbrigham.org/en/about/newsroom/press-releases/otof-deafness-gene-therapy-trial-update"]]},
    {"id": "osia-3", "tag": "Bone conduction", "date": "2026-07",
     "title": "Cochlear's Osia 3: a fully rechargeable bone-conduction sound processor (FDA-cleared)",
     "summary": "Cochlear says Osia 3 is the first fully rechargeable active bone-conduction processor. It fits "
                "bone-conduction thresholds up to 65 dB HL and streams from Apple, Android and Bluetooth LE "
                "Audio devices. Existing Osia implant users can upgrade without surgery.",
     "why": "Direct phone streaming matters for people who struggle with calls.",
     "caveat": "Performance claims come from Cochlear. Osia and Bonebridge are different systems; "
               "a processor for one does not fit the other.",
     "for_devices": ["bone_conduction_implant", "bone_conduction_softband"],
     "sources": [["Cochlear media release", "https://www.cochlear.com/us/en/corporate/media-center/media-releases/2026/cochlear-introduces-the-new-osia-3-sound-processor"]]},
    {"id": "bonebridge-355", "tag": "Bone conduction", "date": "2026",
     "title": "Bonebridge results from 355 patients: the largest study of the implant so far",
     "summary": "A 2026 Clinical Otolaryngology paper reports safety and hearing outcomes for the MED-EL "
                "Bonebridge BCI 601 and 602 implants across 355 patients. SAMBA 2 remains MED-EL's current "
                "audio processor for Bonebridge.",
     "why": "Real-world evidence for people with conductive loss, atresia or single-sided deafness.",
     "caveat": "Read the paper for the details that apply to you; outcomes depend on bone-conduction thresholds.",
     "for_devices": ["bone_conduction_implant", "bone_conduction_softband"],
     "sources": [["PubMed", "https://pubmed.ncbi.nlm.nih.gov/41121568/"],
                 ["MED-EL: SAMBA 2", "https://www.medel.com/en-us/hearing-solutions/samba2"]]},
    {"id": "ai-hearing-aids", "tag": "Hearing aids", "date": "2026-08 to 2026-10",
     "title": "A wave of AI hearing aids that separate speech from noise",
     "summary": "Oticon Reveal runs two AI systems at once, one for speech and one for the listening "
                "situation. Widex Allure AI adds a dedicated AI co-processor and is rolling out in the US and "
                "Europe this autumn. Other major brands launched AI flagships in the same months.",
     "why": "Noisy restaurants and group talk, the hardest places to hear, are exactly what these target.",
     "caveat": "Launch dates vary by country, and most performance claims come from the makers. Try before you buy.",
     "for_devices": ["hearing_aids", "none"],
     "sources": [["Hearing Review: Oticon Reveal", "https://hearingreview.com/hearing-products/hearing-aids/oticon-launches-reveal-hearing-aid-powered-by-dual-ai-system"],
                 ["Hearing Health Matters: Widex Allure AI", "https://hearinghealthmatters.org/hearing-technologies/2026/widex-allure-ai-launch/"]]},
    {"id": "nuance-audio-plus", "tag": "Hearing glasses", "date": "2026-09",
     "title": "Nuance Audio Plus: second-generation hearing glasses",
     "summary": "EssilorLuxottica's hearing-aid glasses hide the speakers and microphones in the frame. The new "
                "version promises clearer speech in noise, about 10 hours of battery, and calls and music when "
                "paired with an iPhone. From $829 in the US.",
     "why": "Hearing help with nothing in the ears, for people who do not want a visible device.",
     "caveat": "FDA-cleared over the counter for adults with perceived mild to moderate loss only.",
     "for_devices": ["none", "hearing_aids"],
     "sources": [["Hearing Review", "https://hearingreview.com/hearing-products/hearing-aids/otc/essilorluxottica-launches-nuance-audio-plus-hearing-glasses"],
                 ["Healio", "https://www.healio.com/news/optometry/20261003/nuance-audio-plus-updated-version-of-hearing-assistance-glasses-launches-in-us"]]},
    {"id": "airpods-hearing-aid", "tag": "Earbuds", "date": "2024-09 (still current)",
     "title": "AirPods Pro as an over-the-counter hearing aid",
     "summary": "The FDA authorized Apple's Hearing Aid Feature, the first over-the-counter hearing aid "
                "software. It runs on AirPods Pro 2 and AirPods Pro 3 and is set up with a hearing test on the iPhone.",
     "why": "A low-cost first step that many people already own.",
     "caveat": "Adults with perceived mild to moderate loss only. Not for one-sided, sudden or severe loss.",
     "for_devices": ["none"],
     "sources": [["FDA", "https://www.fda.gov/news-events/press-announcements/fda-authorizes-first-over-counter-hearing-aid-software"],
                 ["Apple: feature availability", "https://www.apple.com/airpods-pro/feature-availability/"]]},
]

TRUSTED = {"hearingreview.com", "hearingtracker.com", "hearinghealthmatters.org", "nature.com", "nih.gov",
           "fda.gov", "ieee.org", "medel.com", "cochlear.com", "who.int", "asha.org", "audiology.org",
           "healio.com", "harvard.edu", "massgeneralbrigham.org", "reuters.com", "apnews.com", "bbc.co.uk",
           "nytimes.com", "theverge.com", "statnews.com", "sciencedaily.com", "hearingloss.org"}

CACHE_FILE = Path(__file__).resolve().parent.parent / "logs" / "news_cache.json"
CACHE_HOURS = 24
_lock = threading.Lock()

PROMPT = """Search the web for the most important hearing technology news from the last 90 days (today is {today}):
cochlear and bone-conduction implants, hearing aids, gene therapy for deafness, captioning and other access
technology. Prefer trade press (Hearing Review, HearingTracker), journals and official announcements.
{focus}
Return ONLY one JSON object, no other text:
{{"items": [{{"title": "<headline>", "date": "<YYYY-MM-DD or YYYY-MM>", "summary": "<2 plain sentences>",
            "why": "<who it helps, one sentence>", "caveat": "<limits, approval status, or empty>",
            "url": "<link to the article you read>", "source": "<outlet name>"}}]}}
Up to 5 items. Only include stories you found on a page you cite. Do not repeat these known stories: {known}."""


def _site(url):
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return ""
    parts = host.lower().split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 and parts[-2] in ("co", "ac", "gov", "org") and len(parts[-1]) == 2 \
        else ".".join(parts[-2:])


def curated():
    return {"checked": CHECKED, "items": CURATED}


def search(prompt):
    return llm.web_search(prompt)


SEARCH = search


def search_newer(device=None, budget=None):
    """Live search for newer stories. Returns (payload, usage or None)."""
    key = device or "any"
    with _lock:
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        hit = cache.get(key)
        if hit and time.time() - hit["at"] < CACHE_HOURS * 3600:
            return dict(hit["payload"], cached=True), None
    if budget:
        budget.check()
    focus = {"bone_conduction_implant": "Include bone-conduction implant news if there is any.",
             "cochlear_implant": "Include cochlear implant news if there is any.",
             "hearing_aids": "Include hearing aid news if there is any."}.get(device, "")
    text, cited, usage = SEARCH(PROMPT.format(today=time.strftime("%Y-%m-%d"), focus=focus,
                                              known="; ".join(i["title"] for i in CURATED)))
    if budget:
        budget.add(usage["prompt_tokens"] + usage["completion_tokens"])
    try:
        raw = safety.extract_json(text).get("items") or []
    except (ValueError, json.JSONDecodeError, AttributeError):
        raw = []
    cited_set = {u.split("#")[0].rstrip("/") for u in cited}
    clean = lambda s, n: safety.escape_tags(safety._normalize(str(s or "")))[:n]
    items, dropped = [], 0
    for it in raw[:5]:
        url = str((it or {}).get("url") or "")
        if not url.startswith("https://") or url.split("#")[0].rstrip("/") not in cited_set:
            dropped += 1  # the model named a link the search never returned
            continue
        if any(safety.scan(str(it.get(k) or "")) for k in ("title", "summary", "why", "caveat")):
            dropped += 1
            continue
        items.append({"id": "live-" + re.sub(r"[^a-z0-9]+", "-", clean(it.get("title"), 60).lower()),
                      "tag": "New", "date": clean(it.get("date"), 20), "title": clean(it.get("title"), 160),
                      "summary": clean(it.get("summary"), 400), "why": clean(it.get("why"), 200),
                      "caveat": clean(it.get("caveat"), 240), "for_devices": [],
                      "trusted": _site(url) in TRUSTED,
                      "sources": [[clean(it.get("source") or _site(url), 60), url]]})
    payload = {"checked": time.strftime("%Y-%m-%d"), "items": items, "dropped": dropped}
    with _lock:
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        cache[key] = {"at": time.time(), "payload": payload}
        try:
            CACHE_FILE.parent.mkdir(exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except OSError:
            pass
    return payload, usage
