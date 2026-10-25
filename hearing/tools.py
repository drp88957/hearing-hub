"""Deterministic tools the agents can call.

Nothing in this file uses an LLM. The agents call these functions so that the
numbers in the final answer come from code, not from a model's guess.

IMPORTANT: the speech-sound table below uses approximate teaching values
(based on the widely used "familiar sounds" audiogram idea). It is meant to
help people explain how they hear. It is not a clinical tool or a diagnosis.
"""
import math
import re

FREQS = [250, 500, 1000, 2000, 4000, 8000]
BC_FREQS = [250, 500, 1000, 2000, 4000]  # bone conduction is not tested at 8 kHz
DB_MIN, DB_MAX = -10, 120

# (symbol, example word, centre frequency Hz, typical level in dB HL at conversation)
PHONEMES = [
    ("oo", "moon", 300, 50),
    ("m", "map", 400, 45),
    ("b", "ball", 500, 40),
    ("ah", "father", 800, 50),
    ("d", "dog", 1000, 40),
    ("k", "cat", 2000, 30),
    ("sh", "ship", 3000, 40),
    ("t", "top", 3500, 30),
    ("f", "fan", 4000, 25),
    ("th", "thin", 5000, 25),
    ("s", "sun", 6000, 28),
]

# What tends to go wrong when a sound is faint or missed.
CONFUSIONS = {
    "s": "Word endings can vanish: 'cat' vs 'cats', 'he' vs 'he's'.",
    "th": "'thin', 'fin' and 'sin' can sound alike.",
    "f": "'fan', 'van' and 'than' can sound alike.",
    "t": "'fifteen' vs 'fifty' and 'sixteen' vs 'sixty' are easy to mix up.",
    "k": "'cat', 'tat' and 'pat' can blur together.",
    "sh": "'ship', 'sip' and 'chip' can sound alike.",
    "d": "'dad' and 'bad' can be hard to tell apart.",
    "b": "'bat' and 'mat' can be hard to tell apart.",
    "m": "'me' and 'be' can be hard to tell apart.",
}


class AudiogramError(ValueError):
    pass


def clean_ear(raw, freqs=FREQS):
    """Validate one ear's thresholds. Returns {freq:int -> dB:float}."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AudiogramError("Each ear must be a map of frequency to dB.")
    out = {}
    for k, v in raw.items():
        if v is None or v == "":
            continue
        try:
            f = int(k)
            db = float(v)
        except (TypeError, ValueError):
            raise AudiogramError("Frequencies and dB values must be numbers.")
        if f not in freqs:
            raise AudiogramError(f"Frequency {f} Hz is not one of {freqs}.")
        if not (DB_MIN <= db <= DB_MAX):
            raise AudiogramError(f"{db} dB is outside {DB_MIN} to {DB_MAX}.")
        out[f] = db
    return out


def degree(pta):
    if pta is None:
        return None
    if pta <= 25:
        return "normal"
    if pta <= 40:
        return "mild"
    if pta <= 55:
        return "moderate"
    if pta <= 70:
        return "moderately severe"
    if pta <= 90:
        return "severe"
    return "profound"


def _pta(ear):
    vals = [ear[f] for f in (500, 1000, 2000, 4000) if f in ear]
    if len(vals) < 3:
        return None
    return round(sum(vals) / len(vals), 1)


def _interp(ear, hz):
    """Threshold at hz, interpolating on a log-frequency scale."""
    pts = sorted(ear.items())
    if hz <= pts[0][0]:
        return pts[0][1]
    if hz >= pts[-1][0]:
        return pts[-1][1]
    for (f1, d1), (f2, d2) in zip(pts, pts[1:]):
        if f1 <= hz <= f2:
            t = (math.log(hz) - math.log(f1)) / (math.log(f2) - math.log(f1))
            return d1 + t * (d2 - d1)
    return pts[-1][1]


def _configuration(ear):
    lows = [ear[f] for f in (250, 500) if f in ear]
    highs = [ear[f] for f in (4000, 8000) if f in ear]
    if not lows or not highs:
        return "unknown"
    slope = sum(highs) / len(highs) - sum(lows) / len(lows)
    if slope >= 20:
        return "sloping (high sounds are harder than low sounds)"
    if slope <= -15:
        return "rising (low sounds are harder than high sounds)"
    return "flat (similar across low and high sounds)"


# Type of loss, from the gap between air- and bone-conduction thresholds.
LOSS_TYPE_NOTES = {
    "conductive": "Sound is blocked on its way to a working inner ear. Speech is too quiet rather than "
                  "distorted, so once it is loud enough (closer, louder, or through a bone-conduction "
                  "device) it is usually clear.",
    "sensorineural": "The inner ear or hearing nerve is affected. Even when speech is loud enough, some "
                     "sounds can stay blurred, especially in noise.",
    "mixed": "Part of the loss blocks sound on its way in, and part is in the inner ear. Making speech "
             "louder helps, but some sounds can still stay blurred.",
}


def _loss_type(ac, bc):
    """Classify one ear from air (ac) and bone (bc) thresholds. Returns (type, gap, bone_avg)."""
    common = [f for f in (500, 1000, 2000, 4000) if f in ac and f in bc]
    if len(common) < 2:
        return None, None, None
    gap = round(sum(ac[f] - bc[f] for f in common) / len(common), 1)
    bone = round(sum(bc[f] for f in common) / len(common), 1)
    air = sum(ac[f] for f in common) / len(common)
    if air <= 25:
        return "normal", gap, bone
    if gap >= 15:
        return ("conductive" if bone <= 25 else "mixed"), gap, bone
    return "sensorineural", gap, bone


DEVICES = {
    "none": "no device",
    "hearing_aids": "hearing aids",
    "cochlear_implant": "cochlear implant",
    "bone_conduction_implant": "bone-conduction implant (for example Bonebridge, Osia or BAHA)",
    "bone_conduction_softband": "bone-conduction device on a headband or adhesive",
    "cros": "CROS hearing aid",
}
SIDES = ("right", "left", "both")


def clean_device(device, side):
    device = device if device in DEVICES else "none"
    side = side if side in SIDES and device != "none" else None
    return device, side


def _device_notes(device, side, ear_info):
    notes = []
    if device == "none":
        return notes
    notes.append("The test numbers are without the device. With it, much more is audible, but noise, "
                 "distance and accents still make listening harder.")
    if device.startswith("bone_conduction"):
        sides = [side] if side in ("right", "left") else list(ear_info)
        if any(ear_info.get(e, {}).get("loss_type") == "conductive" for e in sides):
            notes.append("A bone-conduction device sends sound through the skull straight to the inner ear, "
                         "skipping the blocked outer and middle ear. Because the inner ear works well here, "
                         "it can make speech clear, not just louder.")
        else:
            notes.append("A bone-conduction device sends sound through the skull straight to the inner ear, "
                         "skipping the outer and middle ear.")
    if side in ("right", "left"):
        other = "left" if side == "right" else "right"
        notes.append(f"The device's microphone is on the {side}, so voices from the {other} are quieter: the head "
                     f"blocks some sound before it reaches the microphone. It helps when people sit or walk on "
                     f"the {side}.")
        if device.startswith("bone_conduction"):
            notes.append("Bone-conducted sound reaches both inner ears, but with one microphone it is hard to tell "
                         "where a sound is coming from, and background noise is harder to listen through.")
        else:
            notes.append("With a device on one side it is hard to tell where a sound is coming from, and "
                         "background noise is harder to listen through.")
    return notes


def _status(sensation_level):
    if sensation_level >= 10:
        return "clear"
    if sensation_level >= 0:
        return "faint"
    return "missed"


def analyze_audiogram(right=None, left=None, right_bc=None, left_bc=None, device="none", device_side=None):
    """Turn thresholds into a plain summary plus a per-sound audibility table.

    Bone-conduction numbers (optional) tell conductive from sensorineural loss. Device and side
    (optional) add fixed notes, and a one-sided device sets which side the person should sit on.
    """
    ears = {"right": clean_ear(right), "left": clean_ear(left)}
    bones = {"right": clean_ear(right_bc, BC_FREQS), "left": clean_ear(left_bc, BC_FREQS)}
    device, device_side = clean_device(device, device_side)
    usable = {n: e for n, e in ears.items() if len(e) >= 3}
    if not usable:
        raise AudiogramError("Enter at least 3 frequencies for one ear.")

    ear_info = {}
    for name, ear in usable.items():
        pta = _pta(ear)
        kind, gap, bone = _loss_type(ear, bones[name])
        ear_info[name] = {
            "pta_db": pta,
            "degree": degree(pta),
            "configuration": _configuration(ear),
            "loss_type": kind or "unknown (no bone-conduction numbers)",
            "air_bone_gap_db": gap,
            "bone_pta_db": bone,
        }

    sounds = []
    for sym, word, hz, level in PHONEMES:
        row = {"sound": sym, "example": word, "hz": hz, "speech_level_db": level}
        best = None
        for name, ear in usable.items():
            sl = level - _interp(ear, hz)
            row[name] = _status(sl)
            best = sl if best is None else max(best, sl)
        row["overall"] = _status(best)
        sounds.append(row)

    missed = [s["sound"] for s in sounds if s["overall"] == "missed"]
    faint = [s["sound"] for s in sounds if s["overall"] == "faint"]
    confusions = [CONFUSIONS[s] for s in missed + faint if s in CONFUSIONS]

    asym, better, diff = None, None, 0
    if len(ear_info) == 2:
        diff = abs(ear_info["right"]["pta_db"] - ear_info["left"]["pta_db"])
        if diff >= 15:
            worse = "right" if ear_info["right"]["pta_db"] > ear_info["left"]["pta_db"] else "left"
            better = "left" if worse == "right" else "right"
            asym = f"The {worse} ear is about {round(diff)} dB worse than the other."

    types = sorted({i["loss_type"] for i in ear_info.values() if i["loss_type"] in LOSS_TYPE_NOTES})
    seat_side = device_side if device_side in ("right", "left") else better

    return {
        "ears": ear_info,
        "sounds": sounds,
        "missed_sounds": missed,
        "faint_sounds": faint,
        "mishear_examples": confusions,
        "asymmetry": asym,
        "high_frequency_trouble": len([s for s in missed if s in ("s", "th", "f", "t", "sh", "k")]) >= 2,
        "loss_type_notes": [LOSS_TYPE_NOTES[t] for t in types],
        "device": {"type": device, "label": DEVICES[device], "side": device_side,
                   "notes": _device_notes(device, device_side, ear_info)},
        "seat_side": seat_side,
        "single_sided": device_side in ("right", "left") or diff >= 30,
        "note": "Speech sounds are based on unaided thresholds and approximate speech levels. "
                "Hearing aids and implants change what is audible. Not a diagnosis.",
    }


def profile_summary(analysis):
    """One short line other agents can use for context."""
    parts = []
    for name, info in analysis["ears"].items():
        kind = info.get("loss_type", "")
        kind = "" if kind.startswith("unknown") else f" {kind}"
        parts.append(f"{name} ear {info['degree']}{kind}, {info['configuration'].split(' (')[0]}")
    s = "; ".join(parts)
    dev = analysis.get("device") or {}
    if dev.get("type", "none") != "none":
        s += f". Uses a {dev['label'].split(' (')[0]}" + (f" on the {dev['side']} side" if dev.get("side") else "")
    if analysis.get("seat_side"):
        s += f". Hears best on the {analysis['seat_side']} side"
    if analysis["missed_sounds"]:
        s += f". Sounds likely missed: {', '.join(analysis['missed_sounds'])}"
    if analysis["asymmetry"]:
        s += f". {analysis['asymmetry']}"
    return s


# ---------------------------------------------------------------- the circle (group seating)

SHAPES = ("round", "long", "campfire", "sofa")
NOISES = {"tv": "TV", "kitchen": "kitchen", "music": "music or speakers", "street": "window or street",
          "crowd": "other tables or a crowd"}
TAG_WORDS = [  # (tag, keywords in the person's note), checked in this order
    ("most", ("most", "important", "main", "host", "boss", "partner", "best friend", "want to hear")),
    ("quiet", ("soft", "quiet", "low voice", "whisper")),
    ("hard", ("fast", "mumble", "accent", "beard", "mustache", "covers mouth")),
    ("loud", ("loud",)),
]
TAG_RANK = {"most": 0, "quiet": 1, "hard": 2, "": 3, "loud": 4}
TAG_LABEL = {"most": "you most want to hear", "quiet": "soft voice", "hard": "harder to understand",
             "loud": "loud voice", "": ""}
MAX_PEOPLE = 12


def parse_people(lines):
    """'Mom - soft voice' -> {'name': 'Mom', 'note': 'soft voice', 'tag': 'quiet'}. No AI."""
    people = []
    for line in lines[:MAX_PEOPLE]:
        name, note = line.strip(), ""
        m = re.match(r"^(.*?)\s*(?:[-:,(]\s*)(.*?)\)?$", line.strip())
        if m and m.group(1):
            name, note = m.group(1).strip(), m.group(2).strip()
        if not name:
            continue
        low = note.lower()
        tag = next((t for t, words in TAG_WORDS if any(w in low for w in words)), "")
        people.append({"name": name[:24], "note": note[:60], "tag": tag})
    return people


def _seat_points(shape, n):
    """n seats (seat 0 = you). Returns [(x, y)] in a 0-1 box; you sit at the bottom, facing up."""
    if shape in ("long",):
        pts = [(0.5, 0.93)]
        side = [p for p in range(1, n)]
        rows = (len(side) + 1) // 2
        for i, k in enumerate(side):
            col = 0.27 if i % 2 == 0 else 0.73        # your left side of the table is on the screen left
            row = i // 2
            y = 0.76 - row * (0.62 / max(1, rows - 1 if rows > 1 else 1))
            pts.append((col, round(y, 3)))
        return pts
    pts = []
    for k in range(n):
        a = math.radians(90 + k * 360 / n)          # k=1 is on your left (screen left), going round
        pts.append((round(0.5 + 0.38 * math.cos(a), 3), round(0.5 + 0.38 * math.sin(a), 3)))
    return pts


def _seat_order(shape, n, side):
    """Best seats first for the people you most need to hear (seat numbers, never 0)."""
    others = list(range(1, n))
    if shape == "long":
        left = [k for i, k in enumerate(others) if i % 2 == 0]
        right = [k for i, k in enumerate(others) if i % 2 == 1]
        near_first = []
        hear, other = (right, left) if side == "right" else (left, right)
        for a, b in zip(hear + [None] * len(other), other + [None] * len(hear)):
            near_first += [x for x in (a, b) if x is not None]
        return near_first
    across = n / 2
    if side in ("left", "right"):
        hear_side = [k for k in others if (k < across) == (side == "left")]
        other_side = [k for k in others if k not in hear_side]
        hear_side.sort(key=lambda k: min(k, n - k))                 # nearest on the hearing side first
        far_other = sorted(other_side, key=lambda k: -min(k, n - k))  # then across, then near on the deaf side
        return hear_side[:1] + far_other[:1] + hear_side[1:] + far_other[1:]
    return sorted(others, key=lambda k: abs(k - across))           # faces across the table first


def plan_seating(shape, people, side=None, noise=None, lipreading=False):
    """Who sits where so the person can follow the group. Deterministic; the AI never picks seats."""
    shape = shape if shape in SHAPES else "round"
    people = [p for p in people if p.get("name")][:MAX_PEOPLE]
    n = len(people) + 1
    side = side if side in ("left", "right") else None
    noise = [x for x in (noise or []) if x in NOISES]
    pts = _seat_points(shape, n)
    order = _seat_order(shape, n, side)
    ranked = sorted(people, key=lambda p: TAG_RANK.get(p.get("tag", ""), 3))
    seats = [{"seat": 0, "x": pts[0][0], "y": pts[0][1], "name": "You", "you": True}]
    for person, k in zip(ranked, order):
        seats.append({"seat": k, "x": pts[k][0], "y": pts[k][1], "name": person["name"],
                      "tag": person.get("tag", ""), "why": TAG_LABEL.get(person.get("tag", ""), "")})
    tips = []
    if side:
        other = "right" if side == "left" else "left"
        tips.append(f"You hear best on your {side}, so the person you most want to hear sits right there, "
                    f"and the next one sits across where you can see their face.")
        tips.append(f"Keep side conversations and noise on your {other}.")
    else:
        tips.append("The people you most need to follow sit across from you, where you can see their faces.")
    if noise:
        names = ", ".join(NOISES[x] for x in noise)
        tips.append(f"Sit with your back to the noise ({names}), and ask if it can be turned down or off.")
    if shape == "long":
        tips.append("At a long table, the end seat lets you see most faces. The middle of the long side is the hardest.")
    if shape == "campfire":
        tips.append("Outdoors, sound drifts away. Sit close, and keep the fire's light on faces, not behind them.")
    if lipreading:
        tips.append("Make sure faces are lit from the front. Avoid sitting facing a bright window or lamp.")
    if len(people) >= 6:
        tips.append(f"With {len(people) + 1} people, side conversations start. Ask for one conversation at a time "
                    "for the parts you care about.")
    return {"shape": shape, "people": len(people), "your_hearing_side": side, "seats": seats,
            "noise_behind_you": [NOISES[x] for x in noise], "tips": tips}


# ---------------------------------------------------------------- report numbers

def clean_report_ear(raw, freqs=FREQS):
    """Check thresholds the model read from a report. Keeps only valid frequencies and dB values,
    snapped to the 5 dB steps audiograms use. Never raises: bad values are dropped and counted."""
    out, dropped = {}, 0
    if not isinstance(raw, dict):
        return out, (1 if raw else 0)
    for k, v in raw.items():
        try:
            f, db = int(float(k)), float(v)
        except (TypeError, ValueError):
            dropped += v not in (None, "")
            continue
        if f not in freqs or not (DB_MIN <= db <= DB_MAX):
            dropped += 1
            continue
        out[f] = int(round(db / 5) * 5)
    return out, dropped


# ---------------------------------------------------------------- acoustics

# feature -> (weight, group, plain explanation). Positive weight = harder.
FEATURES = {
    "hard_surfaces": (2.0, "noise", "Hard floors, walls or glass bounce sound and blur consonants."),
    "high_ceiling": (1.0, "noise", "High ceilings add echo."),
    "open_kitchen": (1.5, "noise", "An open kitchen adds clatter close to the tables."),
    "background_music": (2.0, "noise", "Music competes with speech."),
    "crowded": (2.0, "noise", "Many voices at once create a wall of noise."),
    "large_table": (1.0, "noise", "Large groups make it hard to follow who is talking."),
    "street_noise": (1.0, "noise", "Traffic or street noise leaks in."),
    "dim_lighting": (1.5, "light", "Low light makes it hard to see faces and lips."),
    "backlit_seating": (1.0, "light", "Bright windows behind the speaker put their face in shadow."),
    "carpet_or_soft_furnishings": (-1.5, "noise", "Soft materials absorb echo."),
    "booth_or_corner_seating": (-1.0, "noise", "A corner or booth cuts noise from behind you."),
    "quiet_room": (-2.0, "noise", "A quiet room helps a lot."),
    "small_group": (-1.0, "noise", "Small groups are easier to follow."),
    "hearing_loop": (-2.0, "access", "A hearing loop sends sound straight to hearing aids and implants that have a telecoil (T-setting)."),
    "captioning_available": (-2.0, "access", "Captions are available."),
    "good_face_lighting": (-1.0, "light", "Faces are well lit, which helps with lipreading."),
}
BASE_SCORE = 3.0


def score_acoustics(features, high_freq_trouble=False, relies_on_lipreading=False, single_sided=False):
    """Score how hard a place is to hear in, from 1 (easy) to 10 (very hard)."""
    if not isinstance(features, list):
        raise ValueError("features must be a list")
    seen, parts, total = set(), [], BASE_SCORE
    unknown = []
    for f in features:
        if f in seen:
            continue
        seen.add(f)
        if f not in FEATURES:
            unknown.append(f)
            continue
        w, group, why = FEATURES[f]
        mult = 1.0
        if group == "noise" and w > 0 and (high_freq_trouble or single_sided):
            mult = 1.25  # noise hurts more with high-pitch loss or hearing on one side
        if group == "light" and w > 0 and relies_on_lipreading:
            mult = 1.5
        if group == "light" and w < 0 and relies_on_lipreading:
            mult = 1.5
        total += w * mult
        parts.append({"feature": f, "points": round(w * mult, 2), "why": why})
    score = max(1.0, min(10.0, round(total, 1)))
    if score <= 3.5:
        verdict = "Easy"
    elif score <= 6.5:
        verdict = "Manageable with a good seat"
    else:
        verdict = "Hard: consider another place or ask for changes"
    result = {"score": score, "verdict": verdict, "contributions": parts}
    if unknown:
        result["ignored_unknown_features"] = unknown
    return result


# ------------------------------------------------------------ call alternatives

RELAY = [
    {"name": "711 / Telecommunications Relay Service (TRS)",
     "how": "Dial 711 in the US to reach a relay operator who connects you to the business. "
            "Businesses are expected to accept relay calls."},
    {"name": "IP Captioned Telephone (IP CTS)",
     "how": "Shows captions of what the other person says during a call. "
            "It is free for people with hearing loss in the US; you register with a provider. Check current rules."},
    {"name": "Video Relay Service (VRS)",
     "how": "For ASL users: a sign language interpreter on video relays the call."},
]

ORG_TYPES = {
    "medical": {
        "routes": ["Patient portal secure message", "Online appointment booking",
                   "Email or text the clinic's scheduling team"],
        "tips": ["Say you are hard of hearing as the first thing in any call or visit.",
                 "Ask them to confirm dates, times and prices in writing."],
    },
    "bank_or_financial": {
        "routes": ["Secure message inside the bank's app or website", "Live chat",
                   "Visit a branch with a written summary of your issue"],
        "tips": ["Never share full account numbers by email; use the bank's secure messaging."],
    },
    "utility_or_telecom": {
        "routes": ["Account chat or support ticket", "Social media support team (public message, then move to private)",
                   "Online account tools for billing and service changes"],
        "tips": ["Keep the ticket number and ask for a written confirmation."],
    },
    "government": {
        "routes": ["Online forms or account portal", "Email the agency's contact address",
                   "Request an in-person appointment with a written note"],
        "tips": ["Government offices must provide effective communication on request. Ask for it by name."],
    },
    "insurance": {
        "routes": ["Member portal secure message", "Live chat", "Email your agent directly"],
        "tips": ["Ask for claim numbers and decisions in writing."],
    },
    "school_or_university": {
        "routes": ["Email the office or instructor", "Disability services office request form",
                   "Student portal ticket"],
        "tips": ["Put accommodation requests in writing so there is a record."],
    },
    "retail_or_delivery": {
        "routes": ["Order page chat", "Email support", "Text updates from the delivery app"],
        "tips": ["Add a delivery note: 'Hard of hearing. Please text, do not call or rely on the doorbell.'"],
    },
    "other": {
        "routes": ["Email", "Website contact form", "Live chat"],
        "tips": ["Say clearly at the top that you are hard of hearing and prefer written replies."],
    },
}


# Where each kind of organization usually offers a written channel, for when no specific one is known.
WHERE_TO_LOOK = {
    "medical": "Your clinic's patient portal (many US clinics use MyChart): look for Messages or Ask a question. "
               "The clinic's website Contact page usually lists an email.",
    "bank_or_financial": "Log in to the bank's app or website and look for Secure messages or Chat. Don't email account details.",
    "utility_or_telecom": "Log in to your account online and look for Chat, Support or Create a ticket.",
    "government": "The agency website's Contact us page lists email addresses and online forms.",
    "insurance": "The member portal (link on your insurance card) usually has Secure message or Chat.",
    "school_or_university": "The office's page on the school website lists its email. Disability services has its own email.",
    "retail_or_delivery": "Your order page or the app's Help section usually has Chat or email support.",
    "other": "The organization's website: look for Contact us, Chat or a support email.",
}

# Organizations we have checked by hand. Only these contact details may appear in an answer.
# Each entry lists where the facts came from and when they were checked.
KNOWN_ORGS = [
    {"id": "yale_health_pharmacy",
     "keywords": ["yale pharmacy", "yale health pharmacy", "yale health's pharmacy", "pharmacy at yale"],
     "name": "Yale Health Pharmacy (55 Lock Street, New Haven)",
     "written_channels": [
         {"channel": "MyChart (Yale Health's patient portal)",
          "how": "Request refills and send non-urgent messages. Replies usually take 1-3 business days, "
                 "so it is not for urgent issues.",
          "link": "https://yalehealth.yale.edu/using-mychart"}],
     "emails": [
         {"address": "member.services@yale.edu", "for": "Yale Health Member Services: general questions and "
                                                         "requests about how Yale Health contacts you"},
         {"address": "mychartsupport@ynhhs.org", "for": "MyChart help: login or activation problems"}],
     "phone": "203-432-0033 (Mon-Fri 8 am-6 pm, Sat 8 am-3 pm). You can also call through relay by dialing 711.",
     "note": "No public email for the pharmacy itself was found. Use MyChart, or email Member Services.",
     "sources": ["https://yalehealth.yale.edu/department/pharmacy-prescriptions",
                 "https://yalehealth.yale.edu/using-mychart",
                 "https://yalehealth.yale.edu/department/member-services"],
     "checked": "2026-10-24"},
    {"id": "yale_health",
     "keywords": ["yale health", "student health", "yale health center", "yale internal medicine"],
     "name": "Yale Health (55 Lock Street, New Haven)",
     "written_channels": [
         {"channel": "MyChart (Yale Health's patient portal)",
          "how": "Message your primary care clinician about non-urgent issues; replies usually take 1-3 business days.",
          "link": "https://yalehealth.yale.edu/using-mychart"}],
     "emails": [
         {"address": "member.services@yale.edu", "for": "Yale Health Member Services"},
         {"address": "mychartsupport@ynhhs.org", "for": "MyChart help: login or activation problems"}],
     "phone": "Member Services 203-432-0246 (Mon-Fri 8:30 am-5 pm). Relay: dial 711.",
     "note": "For urgent symptoms Yale Health asks you to call rather than use MyChart.",
     "sources": ["https://yalehealth.yale.edu/department/member-services", "https://yalehealth.yale.edu/using-mychart"],
     "checked": "2026-10-24"},
    {"id": "yale_sas",
     "keywords": ["student accessibility", "accessibility services", "yale sas", "disability services at yale"],
     "name": "Yale Student Accessibility Services (SAS)",
     "written_channels": [
         {"channel": "Accommodations request form (Accommodate)", "how": "Request or update accommodations online.",
          "link": "https://sas.yale.edu/students"}],
     "emails": [
         {"address": "sas@yale.edu", "for": "Accommodation questions and documentation"},
         {"address": "sasAT@yale.edu", "for": "Assistive technology, including missing captions on course videos"}],
     "phone": "203-432-2324",
     "note": "",
     "sources": ["https://sas.yale.edu/students", "https://sas.yale.edu/assistive-technology-services/access-communication"],
     "checked": "2026-10-24"},
]


def match_known_orgs(*texts):
    """Find hand-checked organizations named in the person's request (simple keyword match, no AI)."""
    blob = " ".join(t for t in texts if t).lower()
    found = [o for o in KNOWN_ORGS if any(k in blob for k in o["keywords"])]
    # the pharmacy entry is more specific than general Yale Health; keep both only if both are named
    if any(o["id"] == "yale_health_pharmacy" for o in found):
        found = [o for o in found if o["id"] != "yale_health" or "yale health center" in blob]
    return [{k: v for k, v in o.items() if k != "keywords"} for o in found]


def allowed_emails(orgs):
    return {e["address"].lower() for o in orgs for e in o["emails"]}


def find_call_alternatives(organization_type, organization_name=None, request_text=None):
    info = ORG_TYPES.get(organization_type, ORG_TYPES["other"])
    known = match_known_orgs(organization_name, request_text)
    return {
        "organization_type": organization_type if organization_type in ORG_TYPES else "other",
        "known_organizations": known,
        "where_to_look": WHERE_TO_LOOK.get(organization_type, WHERE_TO_LOOK["other"]),
        "non_phone_routes": info["routes"],
        "tips": info["tips"],
        "phone_relay_services": RELAY,
        "note": "Only use email addresses listed in known_organizations. Relay services are US services; "
                "verify before relying on them.",
    }
