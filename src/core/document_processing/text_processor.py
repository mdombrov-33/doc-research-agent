from functools import lru_cache
from pathlib import Path

import aiofiles
import pdfplumber
import spacy
from docx import Document as DocxDocument

from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
        return nlp
    except OSError:
        logger.error("spaCy model not found.")
        raise


class TextExtractor:
    @staticmethod
    async def extract_from_file(file_path: str, filename: str) -> str:
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return TextExtractor._extract_pdf(file_path)
        elif ext == ".docx":
            return TextExtractor._extract_docx(file_path)
        elif ext == ".txt":
            return await TextExtractor._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Extracted {len(text)} chars from PDF")
        return text

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        doc = DocxDocument(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        logger.info(f"Extracted {len(text)} chars from DOCX")
        return text

    @staticmethod
    async def _extract_txt(file_path: str) -> str:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()
        logger.info(f"Extracted {len(text)} chars from TXT")
        return text
