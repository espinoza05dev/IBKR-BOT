from __future__ import annotations
"""
KnowledgeIngestor.py
Orquestador principal de la KnowledgeBase.
Detecta automaticamente el tipo de contenido y lo enruta
al procesador correcto, luego almacena el texto en el VectorStore.
"""

import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Union


class KnowledgeIngestor:
    """
    Punto de entrada unico para ingerir cualquier tipo de contenido.

    Tipos soportados:
        - Texto:  .txt .pdf .docx .epub .md .html + URLs web
        - Audio:  .mp3 .wav .m4a .ogg .flac .aac
        - Video:  .mp4 .avi .mkv .mov .webm + URLs de YouTube
        - Imagen: .jpg .jpeg .png .bmp .tiff .webp
    """

    def __init__(self, collection: str = "trading_knowledge"):
        from IA.KnowledgeBase.VectorStore import VectorStore
        from IA.KnowledgeBase.Processors.TextProcessor  import TextProcessor
        from IA.KnowledgeBase.Processors.AudioProcessor import AudioProcessor
        from IA.KnowledgeBase.Processors.Videoprocessor import VideoProcessor
        from IA.KnowledgeBase.Processors.Imageprocessor import ImageProcessor

        self.store   = VectorStore(collection=collection)
        self._text   = TextProcessor()
        self._audio  = AudioProcessor()
        self._video  = VideoProcessor()
        self._image  = ImageProcessor(mode="ocr")
        self._log    = []

    # ── API publica ───────────────────────────────────────────────────────────

    def ingest(self, source: Union[str, Path], metadata: dict | None = None) -> int:
        """
        Ingesta automatica: detecta tipo y procesa.
        Retorna numero de chunks almacenados.
        """
        source = str(source)
        ctype  = self._detect_type(source)
        print(f"[Ingestor] Tipo detectado: {ctype.upper()} | {source}")

        try:
            if ctype == "text":
                text = self._text.process(source)
            elif ctype == "audio":
                text = self._audio.process(source)
            elif ctype == "video":
                text = self._video.process(source)
            elif ctype == "image":
                text = self._image.process(source)
            else:
                raise ValueError(f"Tipo desconocido: {ctype}")

            if not text.strip():
                print(f"[Ingestor] Advertencia: contenido vacio en {source}")
                return 0

            chunks = self.store.add(
                text         = text,
                source       = source,
                content_type = ctype,
            )

            self._log_entry(source, ctype, chunks, success=True)
            return chunks

        except Exception as e:
            print(f"[Ingestor] ERROR procesando {source}: {e}")
            self._log_entry(source, ctype, 0, success=False, error=str(e))
            return 0

    def ingest_text(self, text: str, source: str = "manual", **kwargs) -> int:
        """Ingesta texto en bruto directamente."""
        chunks = self.store.add(text=text, source=source, content_type="text")
        self._log_entry(source, "text", chunks, success=True)
        return chunks

    def ingest_folder(
        self,
        folder: Union[str, Path],
        recursive: bool = True,
        extensions: set | None = None,
    ) -> dict:
        """
        Ingesta todos los archivos de una carpeta.

        Args:
            folder:     Ruta a la carpeta.
            recursive:  Buscar en subcarpetas.
            extensions: Filtro de extensiones. Si None, procesa todo lo soportado.

        Returns:
            {"processed": int, "failed": int, "total_chunks": int}
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"[Ingestor] No es una carpeta: {folder}")

        pattern  = "**/*" if recursive else "*"
        files    = list(folder.glob(pattern))
        stats    = {"processed": 0, "failed": 0, "total_chunks": 0}

        all_exts = (
            self._text.SUPPORTED
            | self._audio.SUPPORTED
            | self._video.SUPPORTED
            | self._image.SUPPORTED
            if not extensions else extensions
        )

        for f in files:
            if not f.is_file():
                continue
            if extensions and f.suffix.lower() not in extensions:
                continue
            if f.suffix.lower() not in {
                ".txt",".pdf",".docx",".epub",".md",".html",
                ".mp3",".wav",".m4a",".ogg",".flac",".aac",
                ".mp4",".avi",".mkv",".mov",".webm",
                ".jpg",".jpeg",".png",".bmp",".tiff",".webp",
            }:
                continue

            chunks = self.ingest(f)
            if chunks > 0:
                stats["processed"]    += 1
                stats["total_chunks"] += chunks
            else:
                stats["failed"] += 1

        print(f"[Ingestor] Carpeta completada: {stats}")
        return stats

    def query(self, text: str, k: int = 5, content_type: str | None = None) -> list[dict]:
        """Busqueda semantica en la KnowledgeBase."""
        return self.store.search(query=text, k=k, content_type=content_type)

    def query_for_trading(self, market_context: str) -> str:
        """
        Retorna contexto relevante como string para el agente de trading.
        Ideal para enriquecer decisiones con conocimiento almacenado.
        """
        return self.store.search_trading_context(market_context, k=3)

    def status(self) -> dict:
        return {
            "total_chunks": self.store.count(),
            "sources":      self.store.list_sources(),
            "log_entries":  len(self._log),
        }

    def export_log(self, path: str = "IA_BackTestsKnowledgeBase/ingestion_log.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._log, f, indent=2, ensure_ascii=False)
        print(f"[Ingestor] Log exportado → {path}")

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_type(source: str) -> str:
        """Detecta el tipo de contenido por extension o URL."""
        source_lower = source.lower()

        # YouTube
        if "youtube.com" in source_lower or "youtu.be" in source_lower:
            return "video"
        # URLs genericas
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            return "text"

        ext = Path(source).suffix.lower()

        TEXT_EXTS  = {".txt", ".pdf", ".docx", ".epub", ".md", ".html", ".htm", ".csv"}
        AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
        VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

        if ext in TEXT_EXTS:  return "text"
        if ext in AUDIO_EXTS: return "audio"
        if ext in VIDEO_EXTS: return "video"
        if ext in IMAGE_EXTS: return "image"

        # Fallback: intentar detectar por mime type
        mime, _ = mimetypes.guess_type(source)
        if mime:
            if mime.startswith("text"):  return "text"
            if mime.startswith("audio"): return "audio"
            if mime.startswith("video"): return "video"
            if mime.startswith("image"): return "image"

        return "text"   # Default

    def _log_entry(self, source, ctype, chunks, success, error=""):
        self._log.append({
            "timestamp":    datetime.now().isoformat(),
            "source":       source,
            "content_type": ctype,
            "chunks":       chunks,
            "success":      success,
            "error":        error,
        })
"""
Uso:
        ingresar_info = KnowledgeIngestor()

        # Archivo individual
        ingresar_info.ingest("trading_for_dummies.pdf")
        ingresar_info.ingest("podcast_mercados.mp3")
        ingresar_info.ingest("grafico_btc.png")
        ingresar_info.ingest("https://youtube.com/watch?v=XYZ")
        ingresar_info.ingest("https://bloomberg.com/article/...")

        # Carpeta completa
        ingresar_info.ingest_folder("mis_recursos/")

        # Texto directo
        ingresar_info.ingest_text("El RSI sobre 70 indica sobrecompra.", source="nota_manual")

        # Consulta al conocimiento
        resultados = ki.query("estrategia de ruptura de resistencia")
        
        yt-dlp --flat-playlist -i "https://www.youtube.com/playlist?list=PLzjPEWJnnJNqFVLqpsnVAZoLuwpbq5_E5" --print "https://youtu.be/%(id)s"
        ejecutar en terminal para ver todos los link de una lista de videos de youtube

"""