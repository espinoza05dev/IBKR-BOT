"""
AudioProcessor.py
Transcribe audio a texto usando OpenAI Whisper (local).
Soporta: .mp3 .wav .m4a .ogg .flac .aac
"""

from __future__ import annotations
from pathlib import Path
from typing import Union


class AudioProcessor:
    """
    Transcribe archivos de audio a texto usando Whisper.

    Modelos disponibles (menor a mayor calidad/peso):
        tiny | base | small | medium | large

    Uso:
        ap = AudioProcessor(model_size="base")
        text = ap.process("podcast_trading.mp3")
    """

    SUPPORTED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

    def __init__(self, model_size: str = "base", language: str = "es"):
        self.model_size = model_size
        self.language   = language
        self._model     = None    # Carga lazy

    def process(self, source: Union[str, Path]) -> str:
        """Transcribe el archivo de audio y retorna el texto."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"[AudioProcessor] Archivo no encontrado: {path}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"[AudioProcessor] Formato no soportado: {path.suffix}")

        print(f"[AudioProcessor] Transcribiendo {path.name} con Whisper-{self.model_size}...")
        self._load_model()

        result = self._model.transcribe(
            str(path),
            language    = self.language,
            verbose     = False,
            fp16        = False,     # CPU-safe
            task        = "transcribe",
        )
        text = result["text"].strip()
        print(f"[AudioProcessor] OK — {len(text.split())} palabras transcritas")
        return text

    def process_with_timestamps(self, source: Union[str, Path]) -> list[dict]:
        """
        Transcribe y retorna segmentos con timestamps.
        Util para sincronizar con video.

        Returns:
            [{"start": float, "end": float, "text": str}, ...]
        """
        self._load_model()
        result   = self._model.transcribe(str(source), language=self.language, fp16=False)
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return segments

    def _load_model(self):
        if self._model is None:
            import whisper
            print(f"[AudioProcessor] Cargando modelo Whisper '{self.model_size}'...")
            self._model = whisper.load_model(self.model_size)