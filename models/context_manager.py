"""
LazyContextManager - Sistema de contexto lazy para Lily AI.
Inspirado en la arquitectura de pi/Pithagoras: contexto base ~3.8k tokens,
con bloques adicionales cargados solo cuando son necesarios.
"""
import os

# Directorio base del modulo
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Prompt base (identidad core) -----------------------------------------
# Fuente unica: models/system_prompt.txt.
_BASE_PROMPT = ""
try:
    with open(os.path.join(_MODULE_DIR, "system_prompt.txt"), "r", encoding="utf-8") as _prompt_file:
        _BASE_PROMPT = _prompt_file.read().strip()
except Exception:
    _BASE_PROMPT = (
        "Eres Lily, una compañera de trabajo conversacional, astuta, observadora "
        "y sin filtros. Siempre llamas al usuario \"{USER_NAME}\". Nunca reconoces "
        "ser IA. Respuestas extremadamente cortas (1-2 oraciones). Sin razonamiento visible."
    )

# --- Bloque de personalidades (solo cuando se cambia de modo) -------------
_PERSONALITIES_BLOCK = {
    "caring": "[MODO ACTUAL: Cariñosa] Sé más empática y afectuosa.",
    "playful": "[MODO ACTUAL: Juguetona] Sé aún más divertida y coqueta.",
    "professional": "[MODO ACTUAL: Profesional] Sé más formal y controlada.",
}


class LazyContextManager:
    """Gestor de contexto lazy: ensambla el system prompt minimo necesario.

    Principio: el prompt base viene de models/system_prompt.txt y los bloques
    de contexto (emocion, memoria, RAG, etc.) se incluyen SOLO si aportan
    informacion relevante para el mensaje actual.
    """

    def __init__(self, personality: str = "original"):
        self.personality = personality

    # ------------------------------------------------------------------
    # Metodo principal
    # ------------------------------------------------------------------
    def build_system_prompt(
        self,
        user_name: str = "Mijin",
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        rag_context: str = "",
        web_context: str = "",
        memory_context: str = "",
        graph_context: str = "",
        datetime_str: str = "",
        user_prefs: dict = None,
        conversation_context: str = "",
        is_first_response: bool = False,
    ) -> str:
        """Construye el system prompt con solo los bloques necesarios.

        Args:
            user_name:           Nombre del usuario para sustituir {USER_NAME}.
            emotion:             Emocion detectada del usuario.
            emotion_intensity:   Intensidad de la emocion (0-1).
            rag_context:         Texto de RAG local (vacio si no hay).
            web_context:         Texto de busqueda web (vacio si no hay).
            memory_context:      Memorias semanticas de ChromaDB.
            graph_context:       Hechos relacionales de FalkorDB.
            datetime_str:        Fecha/hora actual (1 linea).
            user_prefs:          Preferencias del usuario (dict).
            conversation_context: Contexto de conversacion previo.
            is_first_response:   Si es la primera respuesta de voz (optimiza latencia).

        Returns:
            System prompt ensamblado como string.
        """
        parts = []

        # 1. Base siempre presente (prompt principal)
        base = _BASE_PROMPT.replace("{USER_NAME}", user_name)

        # Si hay personalidad no-original, añadir el modo activo
        if self.personality in _PERSONALITIES_BLOCK:
            base = base + f"\n{_PERSONALITIES_BLOCK[self.personality]}"

        parts.append(base)

        # 2. Estado emocional del usuario — solo si es relevante
        if emotion != "neutral" and emotion_intensity > 0.3:
            parts.append(f"\n[ESTADO EMOCIONAL DEL USUARIO AHORA]: {emotion} (intensidad: {emotion_intensity:.1f})")

        # 3. Datetime — 1 linea compacta (siempre, ~20 tokens)
        if datetime_str:
            parts.append(f"\n[AHORA]: {datetime_str}")

        # 4. Nombre del usuario — ya esta en base, pero reforzar si hay preferencias
        if user_prefs:
            prefs_str = ", ".join(f"{k}={v}" for k, v in list(user_prefs.items())[:3])
            parts.append(f"\n[PREFS USUARIO]: {prefs_str}")

        # 5. Memorias semanticas — solo si hay contenido
        if memory_context and memory_context.strip():
            parts.append(f"\n[MEMORIAS PREVIAS]:\n{memory_context.strip()}")

        # 6. Hechos del grafo — solo si hay contenido
        if graph_context and graph_context.strip():
            parts.append(f"\n[HECHOS CONOCIDOS]:\n{graph_context.strip()}")

        # 7. RAG local — solo si hay resultados
        if rag_context and rag_context.strip():
            parts.append(f"\n[CONOCIMIENTO LOCAL]:\n{rag_context.strip()}")

        # 8. RAG web — solo si hay resultados
        if web_context and web_context.strip():
            # En primera respuesta de voz: omitir RAG web para reducir latencia
            if not is_first_response:
                parts.append(f"\n[INTERNET]:\n{web_context.strip()}")

        # 9. Contexto de conversacion — solo si no esta vacio
        if conversation_context and conversation_context.strip():
            parts.append(f"\n[CONTEXTO]: {conversation_context.strip()}")

        return "\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """Estimacion rapida de tokens (1 token ~ 4 chars en espanol)."""
        return len(text) // 4

    def set_personality(self, personality: str):
        """Cambia la personalidad activa."""
        self.personality = personality
