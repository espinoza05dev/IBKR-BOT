"""
ImageProcessor.py
Extrae texto e informacion de imagenes:
  - OCR para capturas, graficos con texto, escaneados.
  - Descripcion semantica via modelo de vision (opcional).
Soporta: .jpg .jpeg .png .bmp .tiff .webp
"""

from __future__ import annotations
from pathlib import Path
from typing import Union


class ImageProcessor:
    """
    Procesa imagenes para extraer texto util para la KnowledgeBase.

    Modos:
        "ocr"     → Tesseract OCR (texto en la imagen)
        "vision"  → Modelo de vision para descripcion semantica (requiere GPU/API)
        "both"    → OCR + descripcion

    Uso:
        ip = ImageProcessor(mode="ocr")
        text = ip.process("captura_grafico.png")
    """

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

    def __init__(self, mode: str = "ocr", lang: str = "spa+eng"):
        self.mode = mode
        self.lang = lang

    def process(self, source: Union[str, Path]) -> str:
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"[ImageProcessor] Imagen no encontrada: {path}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"[ImageProcessor] Formato no soportado: {path.suffix}")

        print(f"[ImageProcessor] Procesando {path.name} | modo={self.mode}...")

        parts = []
        if self.mode in ("ocr", "both"):
            parts.append(self._run_ocr(path))
        if self.mode in ("vision", "both"):
            parts.append(self._run_vision(path))

        return "\n\n".join(p for p in parts if p.strip())

    def process_chart(self, source: Union[str, Path]) -> dict:
        """
        Procesamiento especializado para graficos de precios.
        Extrae: texto del eje, titulos, leyendas, patrones visibles.
        """
        path = Path(source)
        ocr_text = self._run_ocr(path)

        return {
            "raw_text":   ocr_text,
            "source":     str(path),
            "type":       "chart",
            "summary":    self._extract_chart_summary(ocr_text),
        }

    # ── Motores ───────────────────────────────────────────────────────────────

    def _run_ocr(self, path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image, ImageFilter, ImageOps

            img = Image.open(path).convert("RGB")

            # Preprocesamiento para mejor OCR
            img = img.resize(
                (img.width * 2, img.height * 2),
                Image.LANCZOS,
            )
            img = ImageOps.grayscale(img)
            img = img.filter(ImageFilter.SHARPEN)

            text = pytesseract.image_to_string(img, lang=self.lang, config="--psm 6")
            return text.strip()

        except ImportError as e:
            return f"[OCR no disponible: {e}]"
        except Exception as e:
            return f"[Error OCR: {e}]"

    def _run_vision(self, path: Path) -> str:
        """
        Descripcion semantica usando transformers de vision (ViT/BLIP).
        Requiere: pip install transformers torch
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image
            import torch

            print("[ImageProcessor] Cargando modelo de vision BLIP...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            img     = Image.open(path).convert("RGB")
            inputs  = processor(img, return_tensors="pt")
            out     = model.generate(**inputs, max_new_tokens=150)
            caption = processor.decode(out[0], skip_special_tokens=True)
            return f"Descripcion de imagen: {caption}"

        except ImportError:
            return "[Vision: instala transformers para descripcion semantica]"
        except Exception as e:
            return f"[Error vision: {e}]"

    @staticmethod
    def _extract_chart_summary(ocr_text: str) -> str:
        """
        Heuristica simple para detectar si el texto viene de un grafico
        financiero y extraer datos clave.
        """
        import re
        numbers = re.findall(r"\d+[\.,]?\d*", ocr_text)
        keywords = [w for w in ["open", "close", "high", "low", "volume",
                                  "sma", "ema", "rsi", "macd", "buy", "sell",
                                  "support", "resistance"] if w in ocr_text.lower()]
        summary = ""
        if numbers:
            summary += f"Valores numericos detectados: {', '.join(numbers[:10])}. "
        if keywords:
            summary += f"Indicadores/terminos: {', '.join(keywords)}."
        return summary or "Sin datos estructurados detectados."