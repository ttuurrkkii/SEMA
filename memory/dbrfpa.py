"""
Domain-Based Reflective Failure Public Archive (DBRFPA)

Stores failure records organized by domain.
The Searcher Agent queries this to surface known pitfalls
so the Planner can avoid repeating the same mistakes.

Search uses semantic embeddings (all-MiniLM-L6-v2) with cosine similarity.
Persists to dbrfpa_store.json, embeddings cached in dbrfpa_embeddings.npy.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

_DIR = os.path.dirname(__file__)
_DEFAULT_PATH = os.path.join(_DIR, "dbrfpa_store.json")
_EMBED_PATH = os.path.join(_DIR, "dbrfpa_embeddings.npy")

# Reuse the lazy-loaded model from stsc
from memory.stsc import _get_model, _cosine_similarity


@dataclass
class FailureEntry:
    task: str
    domain: str
    failure_reason: str
    approach_attempted: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DBRFPA:
    def __init__(self, filepath: str = _DEFAULT_PATH):
        self._store: list[FailureEntry] = []
        self._embeddings: np.ndarray | None = None
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
                self._store.append(FailureEntry(
                    task=r["task"],
                    domain=r["domain"],
                    failure_reason=r["failure_reason"],
                    approach_attempted=r["approach_attempted"],
                    timestamp=r.get("timestamp", ""),
                ))
            if os.path.exists(self._embed_path):
                cached = np.load(self._embed_path)
                if cached.shape[0] == len(self._store):
                    self._embeddings = cached
                else:
                    self._rebuild_embeddings()
            else:
                self._rebuild_embeddings()
        except Exception as e:
            print(f"  [DBRFPA] Warning: could not load {self._filepath}: {e}")

    def _save(self) -> None:
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "task": e.task,
                        "domain": e.domain,
                        "failure_reason": e.failure_reason,
                        "approach_attempted": e.approach_attempted,
                        "timestamp": e.timestamp,
                    }
                    for e in self._store
                ], f, indent=2, ensure_ascii=False)
            if self._embeddings is not None:
                np.save(self._embed_path, self._embeddings)
        except Exception as e:
            print(f"  [DBRFPA] Warning: could not save {self._filepath}: {e}")

    # ── Embeddings ─────────────────────────────────────────────────────────

    def _rebuild_embeddings(self) -> None:
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

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, task: str, domain: str, failure_reason: str, approach_attempted: str) -> None:
        # Dedup: same task + same reason
        task_lower = task.lower().strip()
        reason_lower = failure_reason.lower().strip()
        for entry in self._store:
            if (entry.task.lower().strip() == task_lower and
                    entry.failure_reason.lower().strip() == reason_lower):
                entry.timestamp = datetime.now().isoformat()
                self._save()
                return

        entry = FailureEntry(
            task=task,
            domain=domain,
            failure_reason=failure_reason,
            approach_attempted=approach_attempted,
        )
        self._store.append(entry)

        new_emb = self._embed_query(task)
        if self._embeddings is not None:
            self._embeddings = np.vstack([self._embeddings, new_emb.reshape(1, -1)])
        else:
            self._embeddings = new_emb.reshape(1, -1)

        self._save()

    def search(self, task: str, domain: str = None, top_k: int = 3,
               min_score: float = 0.3) -> list[dict]:
        """
        Return top_k relevant past failures using semantic cosine similarity.
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
            if sim >= min_score:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "task": e.task,
                "domain": e.domain,
                "failure_reason": e.failure_reason,
                "approach_attempted": e.approach_attempted,
                "timestamp": e.timestamp,
                "relevance_score": round(score, 3),
            }
            for score, e in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._store)
