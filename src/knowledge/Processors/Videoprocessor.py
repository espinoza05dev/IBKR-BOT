"""
VideoProcessor.py
Extrae conocimiento de videos:
  1. Descarga (si es URL de YouTube/web) via yt-dlp
  2. Extrae audio y transcribe con Whisper
  3. Extrae frames clave y aplica OCR
Soporta: .mp4 .avi .mkv .mov .webm + URLs de YouTube
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Union

from src.knowledge.Processors.AudioProcessor import AudioProcessor
from src.knowledge.Processors.Imageprocessor import ImageProcessor


class VideoProcessor:
    """
    Pipeline completo video → texto.

    Estrategia:
        - Audio: Whisper transcription
        - Visual: OCR cada N frames (capturas con texto visible)
        - Combina ambos en un texto enriquecido

    Uso:
        vp = VideoProcessor()
        text = vp.process("https://www.youtube.com/watch?v=XXXX")
        text = vp.process("webinar_trading.mp4")
    """

    SUPPORTED = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
    FRAME_INTERVAL = 60    # Capturar frame cada N segundos

    def __init__(self, whisper_model: str = "base", extract_frames: bool = True):
        self.audio_proc  = AudioProcessor(model_size=whisper_model)
        self.image_proc  = ImageProcessor(mode="ocr")
        self.extract_frames = extract_frames

    def process(self, source: Union[str, Path]) -> str:
        """
        Procesa el video y retorna texto combinado (audio + visual).
        Acepta rutas locales o URLs de YouTube.
        """
        source = str(source)

        if "youtube.com" in source or "youtu.be" in source:
            return self._process_youtube(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"[VideoProcessor] Archivo no encontrado: {path}")

        return self._process_file(path)

    # ── Procesamiento ─────────────────────────────────────────────────────────

    def _process_file(self, path: Path) -> str:
        parts = []

        with tempfile.TemporaryDirectory() as tmp:
            # --- Audio ---
            audio_path = Path(tmp) / "audio.wav"
            self._extract_audio(path, audio_path)
            if audio_path.exists():
                transcript = self.audio_proc.process(audio_path)
                if transcript:
                    parts.append(f"=== TRANSCRIPCION DE AUDIO ===\n{transcript}")

            # --- Frames clave ---
            if self.extract_frames:
                frames_text = self._extract_and_ocr_frames(path, tmp)
                if frames_text:
                    parts.append(f"=== TEXTO EXTRAIDO DE FRAMES ===\n{frames_text}")

        combined = "\n\n".join(parts)
        print(f"[VideoProcessor] Procesamiento completado | {len(combined.split())} palabras")
        return combined

    def _process_youtube(self, url: str) -> str:
        """Descarga audio de YouTube y transcribe."""
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("[VideoProcessor] Instala yt-dlp: pip install yt-dlp")

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "yt_audio.%(ext)s"
            ydl_opts = {
                "format":     "bestaudio/best",
                "outtmpl":    str(audio_path),
                "quiet":      True,
                "postprocessors": [{
                    "key":            "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                }],
            }
            print(f"[VideoProcessor] Descargando audio de YouTube: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "YouTube Video")

            # Buscar el archivo descargado
            mp3_files = list(Path(tmp).glob("*.mp3"))
            if not mp3_files:
                return f"[VideoProcessor] No se pudo descargar el audio de: {url}"

            transcript = self.audio_proc.process(mp3_files[0])
            return f"=== VIDEO: {title} ===\n{transcript}"

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_audio(video_path: Path, output_path: Path):
        """Extrae pista de audio del video usando ffmpeg."""
        cmd = (
            f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le '
            f'-ar 16000 -ac 1 "{output_path}" -y -loglevel quiet'
        )
        ret = os.system(cmd)
        if ret != 0:
            print("[VideoProcessor] Advertencia: ffmpeg no disponible o error al extraer audio")

    def _extract_and_ocr_frames(self, video_path: Path, tmp_dir: str) -> str:
        """Captura frames cada N segundos y aplica OCR."""
        try:
            import cv2
        except ImportError:
            print("[VideoProcessor] opencv-python no instalado, omitiendo frames")
            return ""

        cap      = cv2.VideoCapture(str(video_path))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = int(fps * self.FRAME_INTERVAL)
        texts    = []
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_no % interval == 0:
                frame_path = Path(tmp_dir) / f"frame_{frame_no:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                ocr_text = self.image_proc.process(frame_path)
                if ocr_text.strip():
                    texts.append(ocr_text)
            frame_no += 1

        cap.release()
        return "\n---\n".join(texts)