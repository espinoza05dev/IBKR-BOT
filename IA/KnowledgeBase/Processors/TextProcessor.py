"""
TextProcessor.py
Procesa: .txt, .pdf, .epub, .docx, paginas web, markdown.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Union


class TextProcessor:
    """
    Extrae texto limpio de multiples formatos de documento.

    Uso:
        tp = TextProcessor()
        text = tp.process("mi_libro.pdf")
        text = tp.process("https://example.com/articulo")
        text = tp.process("notas.txt")
    """

    SUPPORTED = {".txt", ".pdf", ".epub", ".docx", ".md", ".html"}

    def process(self, source: Union[str, Path]) -> str:
        """
        Detecta el tipo de fuente y extrae el texto.
        Acepta rutas de archivo o URLs.
        """
        source = str(source)

        if source.startswith("http://") or source.startswith("https://"):
            return self._from_url(source)

        path = Path(source)
        ext  = path.suffix.lower()

        handlers = {
            ".pdf":  self._from_pdf,
            ".docx": self._from_docx,
            ".epub": self._from_epub,
            ".html": self._from_html_file,
            ".md":   self._from_plain,
            ".txt":  self._from_plain,
        }
        handler = handlers.get(ext)
        if not handler:
            raise ValueError(f"[TextProcessor] Formato no soportado: {ext}")

        print(f"[TextProcessor] Procesando {path.name}...")
        text = handler(path)
        return self._clean(text)

    # ── Formatos ──────────────────────────────────────────────────────────────

    def _from_plain(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def _from_pdf(self, path: Path) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n".join(pages)
        except ImportError:
            # Fallback a PyPDF2
            import PyPDF2
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)

    def _from_docx(self, path: Path) -> str:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _from_epub(self, path: Path) -> str:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book   = epub.read_epub(str(path))
        chunks = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            chunks.append(soup.get_text(separator=" "))
        return "\n".join(chunks)

    def _from_html_file(self, path: Path) -> str:
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ")

    def _from_url(self, url: str) -> str:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return self._clean(soup.get_text(separator=" "))

    # ── Limpieza ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)          # Espacios multiples
        text = re.sub(r"\n{3,}", "\n\n", text)    # Saltos excesivos
        text = text.strip()
        return text