import asyncio
from functools import lru_cache
from pathlib import Path

import aiofiles
import fitz  # type: ignore[import-untyped]  # pymupdf
import spacy
from docx import Document as DocxDocument

from src.core.exceptions import ModelLoadError, UnsupportedFileTypeError
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spacy_model_loaded", model="en_core_web_sm")
        return nlp
    except OSError as exc:
        logger.error("spacy_model_not_found", model="en_core_web_sm")
        raise ModelLoadError("spaCy model 'en_core_web_sm' not found") from exc


class TextExtractor:
    @staticmethod
    async def extract_from_file(file_path: str, filename: str) -> str:
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return await asyncio.to_thread(TextExtractor._extract_pdf, file_path)
        elif ext == ".docx":
            return await asyncio.to_thread(TextExtractor._extract_docx, file_path)
        elif ext == ".txt":
            return await TextExtractor._extract_txt(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        pages = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text: str = page.get_text("text")
                if page_text.strip():
                    pages.append(page_text)
        text = "\n\n".join(pages)
        logger.info("text_extracted", format="pdf", chars=len(text))
        return text

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        doc = DocxDocument(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        logger.info("text_extracted", format="docx", chars=len(text))
        return text

    @staticmethod
    async def _extract_txt(file_path: str) -> str:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()
        logger.info("text_extracted", format="txt", chars=len(text))
        return text
