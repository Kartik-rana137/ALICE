"""Reading PDF / Word / PowerPoint files and handing them to ALICE (/read), with files built in the test."""
import datetime
import zipfile

import pytest

import alice.session as session_mod
from alice import documents, persona
from alice.agent import TurnResult
from alice.config import Profile
from alice.conversation import Conversation
from alice.tools.files import Workspace

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
P_NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
REL_NS = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'


def make_docx(path, paragraphs, title=""):
    """paragraphs: (style or None, text) or ("table", [[cells]]) or ("list", text)."""
    body = []
    for style, text in paragraphs:
        if style == "table":
            rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row)
                           + "</w:tr>" for row in text)
            body.append(f"<w:tbl>{rows}</w:tbl>")
        else:
            ppr = (f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style not in (None, "list")
                   else '<w:pPr><w:numPr><w:ilvl w:val="0"/></w:numPr></w:pPr>' if style == "list" else "")
            body.append(f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f'<w:document {W_NS}><w:body>{"".join(body)}</w:body></w:document>')
        if title:
            z.writestr("docProps/core.xml", '<cp:coreProperties xmlns:cp="x" xmlns:dc="http://purl.org/dc/'
                       f'elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>')
    return path


def make_pptx(path, slides):
    """slides: list of (title, [body lines], notes or None). File names are reversed vs. presentation order."""
    n = len(slides)
    with zipfile.ZipFile(path, "w") as z:
        ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(n))
        z.writestr("ppt/presentation.xml", f"<p:presentation {P_NS}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>")
        rels = "".join(f'<Relationship Id="rId{i + 1}" Type="http://x/slide" Target="slides/slide{n - i}.xml"/>'
                       for i in range(n))
        z.writestr("ppt/_rels/presentation.xml.rels", f"<Relationships {REL_NS}>{rels}</Relationships>")
        for i, (title, lines, notes) in enumerate(slides):
            file_no = n - i
            title_sp = (f'<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r>'
                        f'<a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>')
            body = "".join(f"<a:p><a:r><a:t>{line}</a:t></a:r></a:p>" for line in lines)
            z.writestr(f"ppt/slides/slide{file_no}.xml",
                       f"<p:sld {P_NS}><p:cSld><p:spTree>{title_sp}<p:sp><p:txBody>{body}</p:txBody></p:sp>"
                       f"</p:spTree></p:cSld></p:sld>")
            if notes:
                z.writestr(f"ppt/slides/_rels/slide{file_no}.xml.rels",
                           f'<Relationships {REL_NS}><Relationship Id="rId9" Type="http://x/notesSlide" '
                           f'Target="../notesSlides/notesSlide{file_no}.xml"/></Relationships>')
                z.writestr(f"ppt/notesSlides/notesSlide{file_no}.xml",
                           f"<p:notes {P_NS}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{notes}</a:t></a:r>"
                           f"</a:p><a:p><a:r><a:t>{file_no}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>"
                           "</p:notes>")
    return path


def make_pdf(path, pages):
    """A minimal text PDF (Helvetica), one string per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 14 Tf 20 100 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


# ----- reading
def test_docx_headings_lists_tables_and_title(tmp_path):
    doc = documents.read(make_docx(tmp_path / "notes.docx", [
        ("Heading1", "Sorting"), (None, "Quicksort picks a pivot."), ("list", "average O(n log n)"),
        ("table", [["algorithm", "worst case"], ["quicksort", "O(n^2)"]]), ("Heading2", "Searching"),
        (None, "Binary search halves the range.")], title="DSA notes"))
    assert doc.kind == "docx" and doc.title == "DSA notes"
    text = doc.text()
    assert "# Sorting" in text and "- average O(n log n)" in text and "| quicksort | O(n^2) |" in text
    assert "## Searching" in text and text.index("Sorting") < text.index("Binary search")


def test_pptx_slides_in_presentation_order_with_notes(tmp_path):
    doc = documents.read(make_pptx(tmp_path / "lecture.pptx", [
        ("Gradient descent", ["Move against the gradient", "Step size = learning rate"], "Mention momentum"),
        ("Overfitting", ["Train well, test badly"], None)]))
    assert doc.kind == "pptx" and [u.label for u in doc.units] == ["slide 1: Gradient descent", "slide 2: Overfitting"]
    assert "Speaker notes: Mention momentum" in doc.units[0].text and "1" not in doc.units[0].text.split(": ")[-1]
    assert doc.title == "Gradient descent" and "2 slides" in doc.describe()


def test_pdf_pages(tmp_path):
    doc = documents.read(make_pdf(tmp_path / "lab.pdf", ["Lab 1: Ohms law", "Results: R = 220 ohm"]))
    assert doc.kind == "pdf" and [u.label for u in doc.units] == ["page 1", "page 2"]
    assert "220" in doc.units[1].text


def test_unreadable_files_get_clear_messages(tmp_path):
    (tmp_path / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(documents.DocumentError, match="save it as .docx or PDF"):
        documents.read(tmp_path / "old.doc")
    (tmp_path / "broken.pptx").write_bytes(b"not a zip")
    with pytest.raises(documents.DocumentError, match="damaged"):
        documents.read(tmp_path / "broken.pptx")
    with pytest.raises(documents.DocumentError, match="no such file"):
        documents.read(tmp_path / "missing.pdf")


def test_excerpts_find_the_relevant_part(tmp_path):
    slides = [(f"Topic {i}", [f"filler text about subject number {i}"] * 20, None) for i in range(30)]
    slides[17] = ("Backpropagation", ["The chain rule computes gradients layer by layer"], None)
    doc = documents.read(make_pptx(tmp_path / "big.pptx", slides))
    best = documents.Excerpts(doc).top("how does backpropagation use the chain rule?", k=2)
    assert best[0][0] == "slide 18: Backpropagation"


# ----- the library and the tool
def test_library_extract_handles_word_and_powerpoint(tmp_path):
    from alice.library import extract
    title, segments, n = extract(make_pptx(tmp_path / "l.pptx", [("Intro", ["hello"], None), ("End", ["bye"], None)]))
    assert n == 2 and [s for _, s, _ in segments] == ["Slide 1: Intro", "Slide 2: End"]
    title, segments, _ = extract(make_docx(tmp_path / "n.docx", [("Heading1", "Sorting"), (None, "Pivot.")]))
    assert segments[0][1] == "Sorting" and "Pivot." in segments[0][2]


def test_read_document_tool_ranges_and_workspace_boundary(tmp_path):
    make_pptx(tmp_path / "deck.pptx", [(f"S{i}", [f"content {i}"], None) for i in range(1, 6)])
    ws = Workspace(tmp_path)
    out = ws.read_document("deck.pptx", start=2, end=3)
    assert out.startswith("S1 (5 slides, PPTX), slides 2-3") and "content 2" in out and "content 4" not in out
    assert "more slides; read again with start=4" in out
    assert ws.read_document("../elsewhere.pdf").startswith("error:")


# ----- in the chat session
@pytest.fixture
def chat(tmp_path, monkeypatch):
    s = session_mod.ChatSession(Profile("default", "fake.gguf"), workspace=str(tmp_path), tools=False,
                                transcript=False, memory_notes=False, claude_escalation=False)
    s.convo, s._today = Conversation("SYS", ctx=16384, reply_budget=1000), datetime.date.today()
    seen = []

    def fake_run_turn(messages, profile, toolbox, **kw):
        seen.append(messages[-1]["content"])
        return TurnResult(reply="ok", messages=[{"role": "assistant", "content": "ok"}], finish="stop")
    monkeypatch.setattr(session_mod, "run_turn", fake_run_turn)
    return s, seen


def test_a_short_document_goes_along_whole_with_the_next_question(chat, tmp_path):
    s, seen = chat
    path = make_docx(tmp_path / "essay.docx", [("Heading1", "Thesis"), (None, "Entropy always increases.")])
    cmd = s.command(f'/read "{path}" what is the thesis?')
    assert cmd.ask == "what is the thesis?" and "whole text goes along" in cmd.message
    s.ask(cmd.ask)
    assert "[Document shared by the student" in seen[0] and "Entropy always increases." in seen[0]
    assert persona.strip_note(s.convo.history[0]["content"]) == "what is the thesis?"       # memory sees only this
    assert "Entropy" in persona.strip_note(s.convo.history[0]["content"], keep_documents=True)  # Claude sees the doc
    s.ask("and something else")
    assert "[Document" not in seen[1]                                                          # sent once


def test_a_long_document_stays_open_and_each_question_gets_excerpts(chat, tmp_path):
    s, seen = chat
    slides = [(f"Topic {i}", [f"filler text about subject number {i}"] * 40, None) for i in range(40)]
    slides[25] = ("Backpropagation", ["The chain rule computes gradients layer by layer"], None)
    cmd = s.command(f"/read {make_pptx(tmp_path / 'course.pptx', slides)}")
    assert cmd.ask == session_mod.DEFAULT_DOC_QUESTION and "too long to send whole" in cmd.message
    s.ask(cmd.ask)
    assert "Outline: slide 1: Topic 0" in seen[0]                                   # first question: outline
    s.ask("How does backpropagation use the chain rule?")
    assert "slide 26: Backpropagation" in seen[1] and "Outline" not in seen[1]
    s.ask("thanks!")                                                                # casual: no excerpts
    assert "[Document" not in seen[2]
    assert "closed" in s.command("/read off").message
    s.ask("How does backpropagation use the chain rule?")
    assert "[Document" not in seen[3]


def test_dragging_a_file_into_the_chat_reads_it(chat, tmp_path):
    s, _ = chat
    path = make_pdf(tmp_path / "lab.pdf", ["Lab 1: Ohms law"])
    cmd = s.command(f'"{path}"')                                # Windows Terminal pastes dropped files quoted
    assert cmd is not None and cmd.ask == session_mod.DEFAULT_DOC_QUESTION
    assert s.command("C:\\nonexistent\\file.pdf what is this") is None   # not a real file: just a message
    assert "can't read that" in s.command(f"/read {tmp_path / 'nope.pptx'}").message
