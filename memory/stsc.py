"""
Short-Term Success Cache (STSC)

Stores recent successful task-solution pairs, organized by domain.
The Searcher Agent queries this before planning to reuse strategies
that have worked before on similar tasks.

Search uses semantic embeddings (all-MiniLM-L6-v2) with cosine similarity.
Persists to stsc_store.json, embeddings cached in stsc_embeddings.npy.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

_DIR = os.path.dirname(__file__)
_DEFAULT_PATH = os.path.join(_DIR, "stsc_store.json")
_EMBED_PATH = os.path.join(_DIR, "stsc_embeddings.npy")

# Lazy-load the model (only when first needed)
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


@dataclass
class SuccessEntry:
    task: str
    domain: str
    approach_used: str
    result_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class STSC:
    def __init__(self, max_entries: int = 100, filepath: str = _DEFAULT_PATH):
        self._store: list[SuccessEntry] = []
        self._embeddings: np.ndarray | None = None  # (N, 384)
        self.max_entries = max_entries
        self._filepath = filepath
        self._embed_path = _EMBED_PATH
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._filepath):
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                rows = json.load(f)
            for r in rows:
                self._store.append(SuccessEntry(
                    task=r["task"],
                    domain=r["domain"],
                    approach_used=r["approach_used"],
                    result_summary=r["result_summary"],
                    timestamp=r.get("timestamp", ""),
                ))
            # Load cached embeddings if they match store size
            if os.path.exists(self._embed_path):
                cached = np.load(self._embed_path)
                if cached.shape[0] == len(self._store):
                    self._embeddings = cached
                else:
                    self._rebuild_embeddings()
            else:
                self._rebuild_embeddings()
        except Exception as e:
            print(f"  [STSC] Warning: could not load {self._filepath}: {e}")

    def _save(self) -> None:
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "task": e.task,
                        "domain": e.domain,
                        "approach_used": e.approach_used,
                        "result_summary": e.result_summary,
                        "timestamp": e.timestamp,
                    }
                    for e in self._store
                ], f, indent=2, ensure_ascii=False)
            # Save embeddings
            if self._embeddings is not None:
                np.save(self._embed_path, self._embeddings)
        except Exception as e:
            print(f"  [STSC] Warning: could not save {self._filepath}: {e}")

    # ── Embeddings ─────────────────────────────────────────────────────────

    def _rebuild_embeddings(self) -> None:
        """Rebuild all embeddings from scratch."""
        if not self._store:
            self._embeddings = None
            return
        model = _get_model()
        tasks = [e.task for e in self._store]
        self._embeddings = model.encode(tasks, normalize_embeddings=True)
        np.save(self._embed_path, self._embeddings)

    def _embed_query(self, text: str) -> np.ndarray:
        model = _get_model()
        return model.encode(text, normalize_embeddings=True)

    def _recency_weight(self, entry: SuccessEntry) -> float:
        """Newer entries get a small boost (1.0 to 1.1)."""
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            days_ago = (datetime.now() - ts).days
            return 1.0 + max(0, 0.1 - days_ago * 0.005)
        except (ValueError, TypeError):
            return 1.0

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, task: str, domain: str, approach_used: str, result_summary: str) -> None:
        # Dedup: check semantic similarity with existing entries
        task_lower = task.lower().strip()
        for i, entry in enumerate(self._store):
            if entry.task.lower().strip() == task_lower:
                # Exact match — update existing
                entry.approach_used = approach_used
                entry.result_summary = result_summary
                entry.timestamp = datetime.now().isoformat()
                self._save()
                return

        entry = SuccessEntry(
            task=task,
            domain=domain,
            approach_used=approach_used,
            result_summary=result_summary,
        )
        self._store.append(entry)

        # Add embedding for new entry
        new_emb = self._embed_query(task)
        if self._embeddings is not None:
            self._embeddings = np.vstack([self._embeddings, new_emb.reshape(1, -1)])
        else:
            self._embeddings = new_emb.reshape(1, -1)

        # Enforce max entries
        if len(self._store) > self.max_entries:
            self._store.pop(0)
            self._embeddings = self._embeddings[1:]

        self._save()

    def search(self, task: str, domain: str = None, top_k: int = 3,
               min_score: float = 0.3) -> list[dict]:
        """
        Return top_k relevant past successes using semantic cosine similarity.
        Scores range from 0 (unrelated) to 1 (identical).
        Only returns results above min_score.
        """
        if not self._store or self._embeddings is None:
            return []

        query_emb = self._embed_query(task)

        scored = []
        for i, entry in enumerate(self._store):
            if domain and entry.domain != domain:
                continue
            sim = _cosine_similarity(query_emb, self._embeddings[i])
            sim *= self._recency_weight(entry)
            if sim >= min_score:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "task": e.task,
                "domain": e.domain,
                "approach_used": e.approach_used,
                "result_summary": e.result_summary,
                "timestamp": e.timestamp,
                "relevance_score": round(score, 3),
            }
            for score, e in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._store)
