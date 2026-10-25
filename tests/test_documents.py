"""The upload reader: every supported format, size limits and spoofed files. No AI involved."""
import base64, io, os, pathlib, sys, unittest, zipfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from hearing import documents

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def item(name, raw):
    return {"name": name, "data": base64.b64encode(raw).decode()}


def office_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def big_png():
    # a real picture big enough to count as embedded media (> 2 KB)
    if documents.pymupdf is None:
        return PNG
    pix = documents.pymupdf.Pixmap(documents.pymupdf.csRGB, 120, 80, os.urandom(120 * 80 * 3), False)
    return pix.tobytes("png")


DOCX = office_zip({
    "word/document.xml": '<w:document><w:body><w:p><w:r><w:t>Audiogram: moderate conductive loss</w:t></w:r></w:p>'
                         '<w:p><w:r><w:t xml:space="preserve">Right 65 dB &amp; left 65 dB</w:t></w:r></w:p></w:body></w:document>',
    "word/media/image1.png": big_png()})
PPTX = office_zip({
    "ppt/slides/slide2.xml": "<p:sld><a:p><a:r><a:t>Second slide: Bonebridge left</a:t></a:r></a:p></p:sld>",
    "ppt/slides/slide1.xml": "<p:sld><a:p><a:r><a:t>First slide: my hearing</a:t></a:r></a:p></p:sld>",
    "ppt/slides/slide10.xml": "<p:sld><a:p><a:r><a:t>Tenth slide</a:t></a:r></a:p></p:sld>"})


class ReaderTests(unittest.TestCase):
    def test_picture(self):
        out = documents.read_files([item("chart.png", PNG)])
        self.assertEqual(len(out["images"]), 1)
        self.assertTrue(out["images"][0].startswith("data:image/"))

    def test_docx_text_and_pictures(self):
        out = documents.read_files([item("report.docx", DOCX)])
        text = out["texts"][0]["text"]
        self.assertIn("moderate conductive loss", text)
        self.assertIn("Right 65 dB & left 65 dB", text)
        if documents.pymupdf is not None:
            self.assertEqual(len(out["images"]), 1)

    def test_pptx_slides_in_order(self):
        text = documents.read_files([item("deck.pptx", PPTX)])["texts"][0]["text"]
        self.assertLess(text.index("First slide"), text.index("Second slide"))
        self.assertLess(text.index("Second slide"), text.index("Tenth slide"))

    @unittest.skipIf(documents.pymupdf is None, "pymupdf not installed")
    def test_pdf_text_and_page_image(self):
        doc = documents.pymupdf.open()
        doc.new_page().insert_text((72, 72), "Audiology report: bilateral conductive loss")
        out = documents.read_files([item("report.pdf", doc.tobytes())])
        self.assertIn("bilateral conductive loss", out["texts"][0]["text"])
        self.assertEqual(len(out["images"]), 1)  # the page is also sent as a picture

    def test_legacy_doc_best_effort_text(self):
        raw = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64 + "Moderate conductive hearing loss".encode("utf-16-le")
        out = documents.read_files([item("old.doc", raw)])
        self.assertIn("Moderate conductive hearing loss", out["texts"][0]["text"])
        self.assertIn("older format", out["files"][0]["note"])

    def test_txt_and_mixed_upload(self):
        out = documents.read_files([item("notes.txt", b"Uses a Bonebridge on the left."), item("chart.png", PNG),
                                    item("deck.pptx", PPTX)])
        self.assertEqual([f["type"] for f in out["files"]], ["txt", "png", "pptx"])
        self.assertEqual(len(out["texts"]), 2)

    def test_spoofed_extension_is_rejected(self):
        with self.assertRaisesRegex(documents.DocumentError, "do not match"):
            documents.read_files([item("report.pdf", PNG)])

    def test_unsupported_type_and_limits(self):
        with self.assertRaisesRegex(documents.DocumentError, "not supported"):
            documents.read_files([item("run.exe", b"MZ")])
        with self.assertRaisesRegex(documents.DocumentError, "up to 8"):
            documents.read_files([item(f"{i}.png", PNG) for i in range(9)])
        with self.assertRaises(documents.DocumentError):
            documents.read_files([])

    def test_zip_bomb_is_refused(self):
        bomb = office_zip({"word/document.xml": "<w:t>x</w:t>", "word/media/a.png": b"\x00" * (41 * 1024 * 1024)})
        with self.assertRaisesRegex(documents.DocumentError, "too large once unpacked"):
            documents.read_files([item("bomb.docx", bomb)])

    def test_image_cap(self):
        out = documents.read_files([item(f"{i}.png", PNG) for i in range(8)])
        self.assertEqual(len(out["images"]), documents.MAX_IMAGES)


if __name__ == "__main__":
    unittest.main()
