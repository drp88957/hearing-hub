"""Turn uploaded hearing reports into text and images the report agent can read.

Nothing in this file uses an LLM. Files are read in memory and never saved.
Supported: PNG, JPEG, WebP, GIF, PDF, DOCX, PPTX, TXT, plus best-effort text from
older DOC and PPT files.
"""
import base64
import binascii
import re
import zipfile
from io import BytesIO

try:
    import pymupdf  # reads PDFs and resizes images
except ImportError:  # the rest of the app still works without it
    pymupdf = None

MAX_FILES = 8
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_IMAGES = 6            # images sent to the model per upload
MAX_TEXT_CHARS = 12000    # text sent to the model per upload
MAX_PDF_PAGES = 8         # pages read for text
RENDER_PDF_PAGES = 3      # pages turned into images (audiograms are charts)
IMG_MAX_SIDE = 2000  # audiogram symbols need detail
ZIP_MAX_UNCOMPRESSED = 40 * 1024 * 1024  # zip-bomb guard for DOCX/PPTX
MAX_PIXELS = 40_000_000  # a small file can still decode to gigabytes (pictures inside DOCX/PPTX are not shrunk)

EXTENSIONS = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "pdf": "pdf", "docx": "docx", "pptx": "pptx", "doc": "doc", "ppt": "ppt", "txt": "txt",
}
ACCEPT_LABEL = "PNG, JPG, WebP, GIF, PDF, DOC, DOCX, PPT, PPTX or TXT"
IMAGE_MEDIA = re.compile(r"\.(png|jpe?g|gif|bmp)$", re.I)


class DocumentError(ValueError):
    """A problem with an upload. The message is safe to show to the user."""


# ------------------------------------------------------------------ decoding

def _decode(item):
    if not isinstance(item, dict):
        raise DocumentError("Each upload must have a name and data.")
    name = str(item.get("name") or "file")[:120]
    data = item.get("data")
    if not isinstance(data, str) or not data:
        raise DocumentError(f"{name}: the file was empty.")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    if len(data) > MAX_FILE_BYTES * 4 // 3 + 16:
        raise DocumentError(f"{name}: files must be under {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise DocumentError(f"{name}: the file could not be read.")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in EXTENSIONS:
        raise DocumentError(f"{name}: this file type is not supported. Use {ACCEPT_LABEL}.")
    return name, ext, raw


def _sniff(raw):
    """What the bytes really are, whatever the file name says."""
    if raw.startswith(b"\x89PNG"):
        return "png"
    if raw.startswith(b"\xff\xd8"):
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:4] == b"GIF8":
        return "gif"
    if raw.startswith(b"%PDF"):
        return "pdf"
    if raw.startswith(b"PK\x03\x04"):
        return "zip"
    if raw.startswith(b"\xd0\xcf\x11\xe0"):
        return "ole"
    return "other"


EXPECTED = {"image": {"png", "jpeg", "webp", "gif"}, "pdf": {"pdf"}, "docx": {"zip"}, "pptx": {"zip"},
            "doc": {"ole"}, "ppt": {"ole"}, "txt": {"other"}}


# ------------------------------------------------------------------ images

def _exif_rotation(raw):
    """Degrees to turn a JPEG clockwise so it is upright (phones store photos sideways + a flag)."""
    if not raw.startswith(b"\xff\xd8"):
        return 0
    i = 2
    while i + 4 <= len(raw) and raw[i] == 0xFF:
        marker, size = raw[i + 1], int.from_bytes(raw[i + 2:i + 4], "big")
        if marker == 0xE1 and raw[i + 4:i + 10] == b"Exif\x00\x00":
            t = raw[i + 10:i + 2 + size]
            order = "little" if t[:2] == b"II" else "big"
            ifd = int.from_bytes(t[4:8], order)
            for k in range(int.from_bytes(t[ifd:ifd + 2], order)):
                e = ifd + 2 + 12 * k
                if int.from_bytes(t[e:e + 2], order) == 0x0112:
                    return {3: 180, 6: 90, 8: 270}.get(int.from_bytes(t[e + 8:e + 10], order), 0)
            return 0
        if marker == 0xDA:  # start of image data: no EXIF before it
            return 0
        i += 2 + size
    return 0


def _pixels(raw):
    """Width x height from the image header without decoding it, or None if unknown."""
    try:
        if raw.startswith(b"\x89PNG"):
            return int.from_bytes(raw[16:20], "big") * int.from_bytes(raw[20:24], "big")
        if raw[:4] == b"GIF8":
            return int.from_bytes(raw[6:8], "little") * int.from_bytes(raw[8:10], "little")
        if raw[:4] == b"RIFF" and raw[12:16] == b"VP8X":
            return (int.from_bytes(raw[24:27], "little") + 1) * (int.from_bytes(raw[27:30], "little") + 1)
        if raw.startswith(b"\xff\xd8"):
            i = 2
            while i + 9 < len(raw) and raw[i] == 0xFF:
                marker, size = raw[i + 1], int.from_bytes(raw[i + 2:i + 4], "big")
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
                    return int.from_bytes(raw[i + 5:i + 7], "big") * int.from_bytes(raw[i + 7:i + 9], "big")
                i += 2 + size
    except (IndexError, ValueError):
        pass
    return None


def _image_url(raw):
    """Return an upright JPEG data URL no larger than IMG_MAX_SIDE on its long side, or None."""
    kind = _sniff(raw)
    if (_pixels(raw) or 0) > MAX_PIXELS:
        raise DocumentError("a picture is too large (over 40 megapixels). Take a smaller photo or screenshot.")
    if pymupdf is not None:
        try:
            pix = pymupdf.Pixmap(raw)
            w, h = pix.width, pix.height
            turn = _exif_rotation(raw)
            if turn in (90, 270):
                w, h = h, w
            scale = min(1.0, IMG_MAX_SIDE / max(w, h))
            doc = pymupdf.open()
            page = doc.new_page(width=w * scale, height=h * scale)
            page.insert_image(page.rect, stream=raw, rotate=(360 - turn) % 360)  # pymupdf turns anti-clockwise
            out = page.get_pixmap(dpi=72, alpha=False)
            return "data:image/jpeg;base64," + base64.b64encode(out.tobytes("jpeg", jpg_quality=85)).decode()
        except Exception:
            pass
    if kind in ("png", "jpeg", "webp", "gif") and len(raw) <= 4 * 1024 * 1024:
        mime = "jpeg" if kind == "jpeg" else kind
        return f"data:image/{mime};base64," + base64.b64encode(raw).decode()
    return None


def _render_page(page):
    zoom = min(2.0, IMG_MAX_SIDE / max(page.rect.width, page.rect.height, 1))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg", jpg_quality=82)).decode()


# ------------------------------------------------------------------ readers

def _read_pdf(raw):
    if pymupdf is None:
        raise DocumentError("PDF support needs the pymupdf package (pip install -r requirements.txt).")
    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception:
        raise DocumentError("the PDF could not be opened.")
    if doc.needs_pass:
        raise DocumentError("the PDF is password-protected.")
    pages = min(doc.page_count, MAX_PDF_PAGES)
    text = "\n".join(doc[i].get_text("text") for i in range(pages))
    images = [_render_page(doc[i]) for i in range(min(doc.page_count, RENDER_PDF_PAGES))]
    note = f"{doc.page_count} page(s)" + (f", first {pages} read" if doc.page_count > pages else "")
    return text, images, note


def _zip_parts(raw):
    try:
        z = zipfile.ZipFile(BytesIO(raw))
    except zipfile.BadZipFile:
        raise DocumentError("the file looks damaged.")
    if sum(i.file_size for i in z.infolist()) > ZIP_MAX_UNCOMPRESSED:
        raise DocumentError("the file is too large once unpacked.")
    return z


def _xml_text(xml, tag, para_end):
    xml = re.sub(para_end, "\n", xml)
    parts = re.findall(rf"<{tag}(?:\s[^>]*)?>([^<]*)</{tag}>|(\n)", xml)
    text = "".join(a or b for a, b in parts)
    return re.sub(r"&lt;", "<", re.sub(r"&gt;", ">", re.sub(r"&amp;", "&", text)))


def _media_images(z, prefix):
    out = []
    for info in z.infolist():
        if info.filename.startswith(prefix) and IMAGE_MEDIA.search(info.filename) and info.file_size > 2000:
            url = _image_url(z.read(info))
            if url:
                out.append(url)
    return out


def _read_docx(raw):
    z = _zip_parts(raw)
    if "word/document.xml" not in z.namelist():
        raise DocumentError("this is not a Word (.docx) document.")
    text = _xml_text(z.read("word/document.xml").decode("utf-8", "replace"), "w:t", r"</w:p>")
    images = _media_images(z, "word/media/")
    return text, images, f"{len(images)} picture(s)"


def _read_pptx(raw):
    z = _zip_parts(raw)
    slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)))
    if not slides:
        raise DocumentError("this is not a PowerPoint (.pptx) file.")
    text = "\n".join(f"[Slide {i + 1}]\n" + _xml_text(z.read(n).decode("utf-8", "replace"), "a:t", r"</a:p>")
                     for i, n in enumerate(slides))
    images = _media_images(z, "ppt/media/")
    return text, images, f"{len(slides)} slide(s), {len(images)} picture(s)"


def _read_legacy(raw):
    """Older .doc/.ppt: pull readable text runs out of the binary. Best effort, no pictures."""
    runs = re.findall(rb"(?:[\x20-\x7e]\x00){5,}", raw)
    text = [r.decode("utf-16-le", "ignore") for r in runs]
    text += [r.decode("latin-1") for r in re.findall(rb"[\x20-\x7e]{6,}", raw)]
    seen, lines = set(), []
    for t in text:
        t = t.strip()
        if len(t) >= 5 and sum(c.isalpha() for c in t) >= len(t) * 0.5 and t not in seen:
            seen.add(t)
            lines.append(t)
    return "\n".join(lines), [], "older format: text only (save as PDF or DOCX for pictures)"


# ------------------------------------------------------------------ entry point

def read_files(items):
    """Read a list of {"name", "data"} uploads.

    Returns {"texts": [{"name", "text"}], "images": [data_url], "files": [summary]}.
    """
    if not isinstance(items, list) or not items:
        raise DocumentError("Choose at least one file to upload.")
    if len(items) > MAX_FILES:
        raise DocumentError(f"Upload up to {MAX_FILES} files at a time.")
    texts, images, files, total = [], [], [], 0
    for item in items:
        name, ext, raw = _decode(item)
        total += len(raw)
        if len(raw) > MAX_FILE_BYTES:
            raise DocumentError(f"{name}: files must be under {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        if total > MAX_TOTAL_BYTES:
            raise DocumentError(f"All files together must be under {MAX_TOTAL_BYTES // (1024 * 1024)} MB.")
        kind = EXTENSIONS[ext]
        if _sniff(raw) not in EXPECTED[kind]:
            raise DocumentError(f"{name}: the contents do not match a .{ext} file.")
        try:
            if kind == "image":
                url = _image_url(raw)
                if not url:
                    raise DocumentError("the picture could not be read.")
                file_text, file_images, note = "", [url], "picture"
            elif kind == "pdf":
                file_text, file_images, note = _read_pdf(raw)
            elif kind == "docx":
                file_text, file_images, note = _read_docx(raw)
            elif kind == "pptx":
                file_text, file_images, note = _read_pptx(raw)
            elif kind in ("doc", "ppt"):
                file_text, file_images, note = _read_legacy(raw)
            else:
                file_text, file_images, note = raw.decode("utf-8", "replace"), [], "text"
        except DocumentError as e:
            raise DocumentError(f"{name}: {e}")
        except Exception:  # damaged pages, unusual zip compression: say so instead of a server error
            raise DocumentError(f"{name}: the file could not be read. Try saving it as PDF or a photo.")
        file_text = re.sub(r"\n{3,}", "\n\n", file_text).strip()
        if file_text:
            texts.append({"name": name, "text": file_text})
        room = MAX_IMAGES - len(images)
        images.extend(file_images[:max(0, room)])
        files.append({"name": name, "type": ext, "note": note, "characters": len(file_text),
                      "pictures_used": min(len(file_images), max(0, room))})
    # share the text allowance across files
    budget = MAX_TEXT_CHARS
    for t in texts:
        t["text"] = t["text"][:max(0, budget)]
        budget -= len(t["text"])
    texts = [t for t in texts if t["text"]]
    if not texts and not images:
        raise DocumentError("Nothing readable was found in these files.")
    return {"texts": texts, "images": images, "files": files}
