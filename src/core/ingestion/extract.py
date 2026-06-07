import asyncio
from pathlib import Path
from typing import cast

import aiofiles
import fitz  # type: ignore[import-untyped]  # pymupdf
from docx import Document as DocxDocument

from src.core.exceptions import UnsupportedFileTypeError
from src.utils.logger import logger


async def extract_from_file(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return await asyncio.to_thread(_extract_pdf, file_path)
    elif ext == ".docx":
        return await asyncio.to_thread(_extract_docx, file_path)
    elif ext == ".txt":
        return await _extract_txt(file_path)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")


def _extract_pdf(file_path: str) -> str:
    pages = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            page_text = cast(str, page.get_text("text"))
            if page_text.strip():
                pages.append(page_text)
    text = "\n\n".join(pages)
    logger.info("text_extracted", format="pdf", chars=len(text))
    return text


def _extract_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    logger.info("text_extracted", format="docx", chars=len(text))
    return text


async def _extract_txt(file_path: str) -> str:
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        text = await f.read()
    logger.info("text_extracted", format="txt", chars=len(text))
    return text
