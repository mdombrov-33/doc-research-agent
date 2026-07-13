import asyncio
from pathlib import Path
from typing import cast

import aiofiles
import fitz  # type: ignore[import-untyped]  # pymupdf
from docx import Document as DocxDocument

from src.core.exceptions import DocumentLimitError, UnsupportedFileTypeError
from src.utils.logger import logger


async def extract_from_file(
    file_path: str,
    filename: str,
    *,
    max_pdf_pages: int,
    max_extracted_characters: int,
) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return await asyncio.to_thread(
            _extract_pdf,
            file_path,
            max_pdf_pages,
            max_extracted_characters,
        )
    elif ext == ".docx":
        return await asyncio.to_thread(_extract_docx, file_path, max_extracted_characters)
    elif ext == ".txt":
        return await _extract_txt(file_path, max_extracted_characters)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")


def _extract_pdf(file_path: str, max_pages: int, max_characters: int) -> str:
    pages: list[str] = []
    with fitz.open(file_path) as pdf:
        if pdf.page_count > max_pages:
            raise DocumentLimitError("PDF page limit exceeded")

        characters = 0
        for page in pdf:
            page_text = cast(str, page.get_text("text"))
            if page_text.strip():
                characters += len(page_text) + (2 if pages else 0)
                if characters > max_characters:
                    raise DocumentLimitError("Extracted text limit exceeded")
                pages.append(page_text)
    text = "\n\n".join(pages)
    logger.info("text_extracted", format="pdf", chars=len(text))
    return text


def _extract_docx(file_path: str, max_characters: int) -> str:
    doc = DocxDocument(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    _validate_text_length(text, max_characters)
    logger.info("text_extracted", format="docx", chars=len(text))
    return text


async def _extract_txt(file_path: str, max_characters: int) -> str:
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        text = await f.read()
    _validate_text_length(text, max_characters)
    logger.info("text_extracted", format="txt", chars=len(text))
    return text


def _validate_text_length(text: str, max_characters: int) -> None:
    if len(text) > max_characters:
        raise DocumentLimitError("Extracted text limit exceeded")
