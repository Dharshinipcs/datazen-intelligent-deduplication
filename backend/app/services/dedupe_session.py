from __future__ import annotations

from threading import Lock
from typing import Any

from app.services.dedupe_active_learning import ActiveLearningSession


class DedupeSessionStore:
    """
    In-memory store for active Dedupe sessions.

    This keeps the trained/active-learning linker alive across
    multiple API requests.

    Prototype limitation:
        Sessions disappear when the backend process restarts.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ActiveLearningSession] = {}
        self._lock = Lock()

    def save(
        self,
        dataset_id: str,
        session: ActiveLearningSession,
    ) -> None:
        with self._lock:
            self._sessions[dataset_id] = session

    def get(
        self,
        dataset_id: str,
    ) -> ActiveLearningSession:
        with self._lock:
            session = self._sessions.get(dataset_id)

        if session is None:
            raise KeyError(dataset_id)

        return session

    def delete(self, dataset_id: str) -> None:
        with self._lock:
            self._sessions.pop(dataset_id, None)

    def exists(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._sessions


dedupe_sessions = DedupeSessionStore()