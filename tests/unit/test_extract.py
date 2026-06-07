import fitz  # type: ignore[import-untyped]
import pytest
from docx import Document as DocxDocument

from src.core.exceptions import UnsupportedFileTypeError
from src.core.ingestion.extract import extract_from_file


async def test_extract_txt(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    text = await extract_from_file(str(path), "note.txt")
    assert text == "hello world"


async def test_extract_docx(tmp_path):
    path = tmp_path / "doc.docx"
    doc = DocxDocument()
    doc.add_paragraph("first line")
    doc.add_paragraph("second line")
    doc.save(str(path))
    text = await extract_from_file(str(path), "doc.docx")
    assert text == "first line\nsecond line"


async def test_extract_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "pdf body text")
    pdf.save(str(path))
    pdf.close()
    text = await extract_from_file(str(path), "doc.pdf")
    assert "pdf body text" in text


async def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        await extract_from_file(str(path), "data.csv")


async def test_extension_matched_case_insensitively(tmp_path):
    path = tmp_path / "NOTE.TXT"
    path.write_text("upper ext", encoding="utf-8")
    text = await extract_from_file(str(path), "NOTE.TXT")
    assert text == "upper ext"
