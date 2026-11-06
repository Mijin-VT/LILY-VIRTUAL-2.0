"""
Cliente de Google Gemini para Lily AI (google-genai / Google AI Studio).

La clave API se lee de las preferencias del usuario (guardada desde
"Configuración y Conocimiento (RAG)"). Si no hay clave, o si Gemini falla,
el motor de lenguaje continúa usando Ollama como respaldo.
"""

import os
import time
from typing import Dict, Iterator, List, Optional


class GeminiError(Exception):
    """Error controlado del cliente de Gemini."""


class GeminiClient:
    """Envoltorio mínimo de google-genai adaptado al formato de mensajes de Lily."""

    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    FALLBACK_MODELS = ("gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash")
    FAILURE_COOLDOWN_SECONDS = 60
    REQUEST_TIMEOUT_MS = 45000
    MAX_CACHED_CLIENTS = 5

    def __init__(self, memory_system=None, default_model: Optional[str] = None):
        self.memory_system = memory_system
        self.default_model = (
            default_model
            or os.environ.get("LILY_GEMINI_MODEL")
            or self.DEFAULT_MODEL
        )
        self._clients: Dict[str, object] = {}
        self._cooldown_until: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Configuración / disponibilidad
    # ------------------------------------------------------------------
    def get_api_key(self, user_id: str) -> str:
        """Obtiene la clave API guardada para el usuario."""
        if not self.memory_system:
            return ""
        try:
            key = self.memory_system.get_preference(user_id, "gemini_api_key", "")
            return (key or "").strip()
        except Exception:
            return ""

    def get_model(self, user_id: str) -> str:
        """Modelo de Gemini a usar (preferencia > variable de entorno > por defecto)."""
        model = ""
        if self.memory_system:
            try:
                model = (self.memory_system.get_preference(user_id, "gemini_model", "") or "").strip()
            except Exception:
                model = ""
        model = model or self.default_model
        if model.startswith("models/"):
            model = model[len("models/"):]
        return model

    def is_available(self, user_id: str) -> bool:
        """True si hay clave configurada y no estamos en enfriamiento por error."""
        api_key = self.get_api_key(user_id)
        if not api_key:
            return False
        return self._cooldown_until.get(api_key, 0.0) <= time.time()

    def has_any_configured_key(self) -> bool:
        """True si algún usuario tiene una clave de Gemini guardada."""
        if not self.memory_system:
            return False
        try:
            if hasattr(self.memory_system, "has_any_preference_value"):
                return self.memory_system.has_any_preference_value("gemini_api_key")
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _get_client(self, api_key: str):
        client = self._clients.get(api_key)
        if client is not None:
            return client

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiError(
                "El paquete google-genai no está instalado. Ejecuta: pip install google-genai"
            ) from exc

        http_options = None
        try:
            http_options = types.HttpOptions(timeout=self.REQUEST_TIMEOUT_MS)
        except Exception:
            http_options = None

        try:
            client = genai.Client(api_key=api_key, http_options=http_options)
        except TypeError:
            client = genai.Client(api_key=api_key)

        self._clients[api_key] = client
        # Evitar acumular clientes si el usuario cambia de clave muchas veces
        if len(self._clients) > self.MAX_CACHED_CLIENTS:
            for old_key in list(self._clients.keys())[:-self.MAX_CACHED_CLIENTS]:
                self._clients.pop(old_key, None)
        return client

    def _mark_failure(self, api_key: str):
        self._cooldown_until[api_key] = time.time() + self.FAILURE_COOLDOWN_SECONDS

    def _mark_success(self, api_key: str):
        self._cooldown_until.pop(api_key, None)

    @staticmethod
    def _build_request(messages: List[dict]):
        """Convierte mensajes estilo OpenAI (system/user/assistant) a Gemini."""
        system_parts: List[str] = []
        turns: List[tuple] = []

        for msg in messages or []:
            role = (msg.get("role") or "user").lower()
            content = msg.get("content") or ""
            if not str(content).strip():
                continue
            if role == "system":
                system_parts.append(str(content))
            else:
                turns.append(("model" if role in ("assistant", "model") else "user", str(content)))

        if not turns:
            raise GeminiError("No hay mensajes para enviar a Gemini")

        contents: List[dict] = []
        for role, text in turns:
            # Gemini requiere alternancia de roles: fusionamos mensajes consecutivos
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append({"text": text})
            else:
                contents.append({"role": role, "parts": [{"text": text}]})

        # Gemini exige que el primer contenido sea del usuario
        if contents and contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "Hola"}]})

        return "\n\n".join(system_parts).strip(), contents

    @staticmethod
    def _build_config(model: str, system_instruction: str):
        from google.genai import types

        kwargs = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        # Controlar "thinking" para respuestas rápidas y baratas.
        if hasattr(types, "ThinkingConfig"):
            try:
                if "2.5" in model and "flash" in model:
                    # Modelos 2.5: desactivar thinking por completo.
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                elif model.startswith(("gemini-3.8", "gemini-4")):
                    # 3.8 Flash / 4.x usan "medium" por defecto; bajar a "low".
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                # 3.5 Flash-Lite y 3.6 Flash ya usan "minimal" por defecto.
            except Exception:
                pass

        return types.GenerateContentConfig(**kwargs)

    def _model_candidates(self, model: str) -> List[str]:
        """Modelo solicitado + alternativas vigentes si el modelo no existe."""
        candidates = [model]
        for fallback in self.FALLBACK_MODELS:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    @staticmethod
    def _is_model_unavailable_error(exc: Exception) -> bool:
        """True si el modelo no existe, está saturado o expiró el tiempo."""
        text = str(exc).lower()
        markers = (
            "not found",
            "no longer available",
            "does not exist",
            "model_not_found",
            "is not supported",
            "unsupported model",
            "deadline",
            "timeout",
            "timed out",
            "504",
            "503",
            "unavailable",
            "overloaded",
        )
        return any(marker in text for marker in markers)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def generate(self, messages: List[dict], user_id: str, model: Optional[str] = None) -> str:
        """Genera una respuesta completa (bloqueante)."""
        api_key = self.get_api_key(user_id)
        if not api_key:
            raise GeminiError("No hay clave API de Gemini configurada")

        model = model or self.get_model(user_id)
        system_instruction, contents = self._build_request(messages)
        client = self._get_client(api_key)

        last_error = None
        for candidate in self._model_candidates(model):
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=contents,
                    config=self._build_config(candidate, system_instruction),
                )
                text = getattr(response, "text", None)
                if not text or not str(text).strip():
                    raise GeminiError("Gemini devolvió una respuesta vacía")
                self._mark_success(api_key)
                return str(text).strip()
            except Exception as exc:
                last_error = exc
                if self._is_model_unavailable_error(exc):
                    continue
                self._mark_failure(api_key)
                raise GeminiError(str(exc)) from exc

        self._mark_failure(api_key)
        raise GeminiError(str(last_error)) from last_error

    def stream(self, messages: List[dict], user_id: str, model: Optional[str] = None) -> Iterator[str]:
        """Genera una respuesta en streaming, produciendo fragmentos de texto."""
        api_key = self.get_api_key(user_id)
        if not api_key:
            raise GeminiError("No hay clave API de Gemini configurada")

        model = model or self.get_model(user_id)
        system_instruction, contents = self._build_request(messages)
        client = self._get_client(api_key)

        last_error = None
        for candidate in self._model_candidates(model):
            yielded = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=candidate,
                    contents=contents,
                    config=self._build_config(candidate, system_instruction),
                ):
                    text = getattr(chunk, "text", None)
                    if text:
                        yielded = True
                        yield text

                if not yielded:
                    raise GeminiError("Gemini devolvió una respuesta vacía")
                self._mark_success(api_key)
                return
            except Exception as exc:
                last_error = exc
                if not yielded and self._is_model_unavailable_error(exc):
                    continue
                self._mark_failure(api_key)
                raise GeminiError(str(exc)) from exc

        self._mark_failure(api_key)
        raise GeminiError(str(last_error)) from last_error
