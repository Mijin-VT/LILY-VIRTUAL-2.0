"""
SessionCache - Cache de sesion en disco para Lily AI.
Permite replay de conversacion si el navegador se desconecta y reconecta.
"""
import os
import json
import time
import threading
from typing import Any, Optional

_SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sessions"
)

# Duracion maxima de sesion en disco: 2 horas
_SESSION_TTL = 7200


class SessionCache:
    """Cache de sesion en memoria + disco.

    Cada sesion de usuario es una lista de eventos SSE en orden cronologico.
    Al reconectarse, el cliente puede recibir replay de todos los eventos
    que ocurrieron mientras estaba desconectado.
    """

    def __init__(self):
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._ram: dict[str, list] = {}  # user_id -> lista de eventos

    # ------------------------------------------------------------------
    # Eventos de sesion
    # ------------------------------------------------------------------
    def append_event(self, user_id: str, event: dict):
        """Agrega un evento SSE a la sesion del usuario."""
        event["_ts"] = time.time()
        with self._lock:
            if user_id not in self._ram:
                self._ram[user_id] = self._load_from_disk(user_id)
            self._ram[user_id].append(event)
            self._persist(user_id)

    def get_events_since(self, user_id: str, since_ts: float = 0.0) -> list:
        """Retorna eventos de la sesion ocurridos despues de since_ts."""
        with self._lock:
            if user_id not in self._ram:
                self._ram[user_id] = self._load_from_disk(user_id)
            events = self._ram.get(user_id, [])
        return [e for e in events if e.get("_ts", 0) > since_ts]

    def clear_session(self, user_id: str):
        """Limpia la sesion del usuario."""
        with self._lock:
            self._ram.pop(user_id, None)
            path = self._session_path(user_id)
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------
    # Persistencia en disco
    # ------------------------------------------------------------------
    def _session_path(self, user_id: str) -> str:
        # Sanitize user_id para nombre de archivo
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return os.path.join(_SESSIONS_DIR, f"{safe_id}.json")

    def _persist(self, user_id: str):
        """Guarda la sesion en disco (llamar con lock adquirido)."""
        try:
            path = self._session_path(user_id)
            events = self._ram.get(user_id, [])
            # Solo guardar eventos de las ultimas 2 horas
            cutoff = time.time() - _SESSION_TTL
            recent = [e for e in events if e.get("_ts", 0) > cutoff]
            self._ram[user_id] = recent
            with open(path, "w", encoding="utf-8") as f:
                json.dump(recent, f, ensure_ascii=False)
        except Exception as e:
            print(f"[SessionCache] Error al persistir sesion: {e}")

    def _load_from_disk(self, user_id: str) -> list:
        """Carga la sesion desde disco."""
        try:
            path = self._session_path(user_id)
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                events = json.load(f)
            # Filtrar eventos expirados
            cutoff = time.time() - _SESSION_TTL
            return [e for e in events if e.get("_ts", 0) > cutoff]
        except Exception:
            return []

    def cleanup_old_sessions(self):
        """Limpia archivos de sesion expirados del disco."""
        try:
            cutoff = time.time() - _SESSION_TTL
            for fname in os.listdir(_SESSIONS_DIR):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(_SESSIONS_DIR, fname)
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
        except Exception as e:
            print(f"[SessionCache] Error limpiando sesiones: {e}")


# Instancia global compartida
session_cache = SessionCache()
