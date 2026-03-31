"""
VectorStore.py
Almacen de vectores para la KnowledgeBase.
Guarda embeddings de texto usando sentence-transformers + ChromaDB.
Permite busqueda semantica para enriquecer las decisiones del agente.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# El modelo all-MiniLM-L6-v2 tiene la clave "embeddings.position_ids" en su
# checkpoint, que versiones recientes de Transformers ya no esperan.
# Es inofensivo: el modelo carga y funciona correctamente.
# Suprimimos el aviso para mantener la salida limpia.
warnings.filterwarnings(
    "ignore",
    message=".*position_ids.*",
    category=UserWarning,
)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


DB_PATH    = Path("../../IA/KnowledgeBase/db")
MODEL_NAME = "all-MiniLM-L6-v2"    # Ligero y rapido, 384 dims


class VectorStore:
    """
    Interfaz unificada para almacenar y consultar conocimiento
    extraido de cualquier tipo de contenido (texto, audio, video, imagen).

    Uso:
        vs = VectorStore()
        vs.add("El mercado tiende a subir en enero", source="libro_trading.pdf")
        resultados = vs.search("comportamiento mercado enero", k=3)
    """

    def __init__(self, collection: str = "trading_knowledge"):
        DB_PATH.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._encoder = SentenceTransformer(MODEL_NAME)
        print(f"[VectorStore] Coleccion '{collection}' | {self.count()} documentos")

    # ── Escritura ─────────────────────────────────────────────────────────────

    def add(
        self,
        text: str,
        source: str = "unknown",
        content_type: str = "text",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> int:
        """
        Divide el texto en chunks, genera embeddings y los almacena.
        Retorna el numero de chunks almacenados.
        """
        chunks = self._chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        embeddings = self._encoder.encode(chunks, show_progress_bar=False).tolist()
        ids        = [self._make_id(chunk, source, i) for i, chunk in enumerate(chunks)]
        metadata   = [
            {
                "source":       source,
                "content_type": content_type,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "ingested_at":  datetime.now().isoformat(),
            }
            for i in range(len(chunks))
        ]

        # Evitar duplicados: upsert por ID
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
        )
        print(f"[VectorStore] +{len(chunks)} chunks desde '{source}' ({content_type})")
        return len(chunks)

    def add_batch(self, entries: list[dict]) -> int:
        """
        Agrega multiples entradas.
        Cada dict: {"text": str, "source": str, "content_type": str}
        """
        total = 0
        for entry in entries:
            total += self.add(
                text         = entry["text"],
                source       = entry.get("source", "batch"),
                content_type = entry.get("content_type", "text"),
            )
        return total

    # ── Lectura ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        content_type: Optional[str] = None,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        Busqueda semantica.

        Returns:
            Lista de {"text": str, "source": str, "score": float, "metadata": dict}
        """
        where_filter = {"content_type": content_type} if content_type else None
        query_embed  = self._encoder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings = query_embed,
            n_results        = min(k, max(self.count(), 1)),
            where            = where_filter,
            include          = ["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist    # distancia coseno → similitud
            if score >= min_score:
                output.append({
                    "text":     doc,
                    "source":   meta.get("source", "?"),
                    "score":    round(score, 4),
                    "metadata": meta,
                })

        return output

    def search_trading_context(self, query: str, k: int = 3) -> str:
        """
        Retorna contexto relevante formateado como string
        para inyectar en la toma de decisiones del agente.
        """
        results = self.search(query, k=k)
        if not results:
            return ""
        lines = [f"[{r['source']} | {r['score']:.2f}] {r['text']}" for r in results]
        return "\n".join(lines)

    # ── Administracion ────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Lista todas las fuentes almacenadas (sin repetir)."""
        if self.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta})

    def delete_source(self, source: str) -> int:
        """Elimina todos los chunks de una fuente especifica."""
        results = self._collection.get(where={"source": source}, include=["metadatas"])
        ids     = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        print(f"[VectorStore] Eliminados {len(ids)} chunks de '{source}'")
        return len(ids)

    def reset(self):
        """Elimina toda la coleccion (IRREVERSIBLE)."""
        self._client.delete_collection(self._collection.name)
        print("[VectorStore] Coleccion eliminada")

    # ── Internos ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
        """Divide texto en chunks con solapamiento."""
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
            start += size - overlap
        return chunks

    @staticmethod
    def _make_id(text: str, source: str, idx: int) -> str:
        raw = f"{source}_{idx}_{text[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()