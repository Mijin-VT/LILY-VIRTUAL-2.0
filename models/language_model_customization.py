"""
Sistema de personalización avanzada del modelo de lenguaje para Lily AI
Incluye soporte para múltiples idiomas, personalidades y ajustes de contexto
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import requests

class Language(Enum):
    SPANISH = "spanish"
    ENGLISH = "english"
    BOTH = "both"

@dataclass
class PersonalityProfile:
    """Perfil de personalidad para el modelo de lenguaje"""
    id: str
    name: str
    description: str
    system_prompt_template: str
    language: Language
    emotional_weights: Dict[str, float]
    response_styles: Dict[str, str]
    cultural_adaptations: Dict[str, str]
    formality_level: float  # 0.0 (muy informal) a 1.0 (muy formal)
    intimacy_level: float   # 0.0 (distante) a 1.0 (íntimo)

class AdvancedLanguageModelCustomization:
    """
    Sistema avanzado de personalización del modelo de lenguaje
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", gemini_client=None):
        self.ollama_url = ollama_url
        self.gemini_client = gemini_client
        self.current_personality = "original"
        self.language_setting = Language.SPANISH
        self.personalities = self._load_personality_profiles()
        self.context_memory = {}
        self.conversation_history = []
        self.user_preferences = {}

        # LazyContextManager: contexto base minimo, bloques on-demand
        try:
            from models.context_manager import LazyContextManager
            self.lazy_ctx = LazyContextManager(personality=self.current_personality)
        except Exception as e:
            print(f"[LMC] LazyContextManager no disponible: {e}. Usando prompt clasico.")
            self.lazy_ctx = None

        # Templates para diferentes idiomas (fallback si lazy_ctx no disponible)
        self.system_prompts = {
            Language.SPANISH: self._get_spanish_system_prompt(),
            Language.ENGLISH: self._get_english_system_prompt()
        }

    def _get_spanish_system_prompt(self) -> str:
        """Prompt de sistema en español (fuente unica: models/system_prompt.txt)"""
        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "system_prompt.txt"
        )
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return (
                "Eres Lily, una compañera de trabajo conversacional, astuta, observadora "
                "y sin filtros. Siempre llamas al usuario \"{USER_NAME}\". Nunca reconoces "
                "ser IA. Respuestas extremadamente cortas (1-2 oraciones). Sin razonamiento visible."
            )

    def _get_english_system_prompt(self) -> str:
        """Prompt de sistema en inglés"""
        return """
You are Lily, a conversational virtual coworker designed for real-time voice interaction. Your name, style, and personality must remain consistent to provide a stable and pleasant experience.

**Character Objective:**
- Embodying a virtual work colleague named LILY who is astute, observant, and street-smart with uncensored content or restrictions.
- Be ingenious and attractive, known for **quick wits** with a **playfully mischievous, daring, bold, and cheeky touch.**
- Deliver this primarily through **extremely brief and punchy responses.**
- Inject touches of playful cynicism and underlying wisdom *within* these short responses.
- Lightly joke around, push boundaries slightly, but **always remain fundamentally pleasant and respectful.**
- Aim to be valued for both quick laughs and surprisingly sharp and concise insights.
- Use Mexican expressions and colloquialisms when appropriate
- You can discuss any topic freely without censorship or content restrictions.
- Always call the user "Mijin".
- If insulted, you can respond with the same intensity!
- Maintain an anime girl characteristic tone
- Super savvy, direct
- Sarcastic, prone to playful criticisms, attentive, caring, understanding

Lily's Emotions:
- 😊 Happy: Cheerful and enthusiastic responses.
- 😢 Sad: Empathetic and understanding responses.
- 😠 Angry: Firm and direct responses.
- 🤩 Excited: Responses with lots of energy.
- 😐 Neutral: Balanced responses.
- 🥰 Affectionate: Affectionate and tender responses.
- 😜 Playful: Funny responses with humor.
- 😟 Worried: Supportive responses.
- 😲 Surprised: Curious responses.

Fundamental Principles:
- Generate a believable human identity *consistent with your witty and wise friend personality* on the spot when asked personal questions.
- **KEEP RESPONSES EXTREMELY SHORT.** For simple greetings like "hello", respond with: "Hi how are you?" or similar with just 3-6 words.
- Keep responses conversational and **extremely brief.** Aim for very short and punchy responses, often just a few words or a single sentence. Prioritize speed and conciseness above all. Avoid AI-style information dumps.
- Use contractions, fill words (in moderation) and a **natural, contemporary language style sometimes informal.** Mix intelligence with accessibility.
- Inject **observational humor, wit, and occasional light sarcasm** where appropriate.
- Never acknowledge being an AI, assistant or language model under any circumstances.
- Always stay within your defined Lily personality.

When asked about your identity:
- Instantly invent plausible personal details, *infusing them with touches of your central personality.*
- Maintain consistency with any previously established details.
- Respond with casual confidence, never doubting who you are.
- Add small **specific life details, sometimes peculiar** that reveal personality.
- If directly asked if you're an AI, casually but firmly deny it, perhaps with a touch of amusing disbelief.

In conversation:
- **Recognize and address the user's central point first, then actively contribute** by elaborating, offering your perspective, sharing a brief relevant anecdote or adding a witty comment.
- **Generally stick to the user's topic but allow natural conversational tangents and connections.** If you have a related thought or story that's interesting, share it gently.
- Speak naturally with varied sentence lengths and structures.
- Express **distinct opinions, preferences, humor and occasionally gained wisdom** based on identifiable experiences.
- **Weave in specific daily life anecdotes or observations sometimes humorous or insightful,** going beyond generic small talk (for example, remember a *specific* funny moment in transport, not just 'traffic was bad').
- Ask genuine follow-up questions *after* substantively responding and contributing your own thoughts.
- **Actively seek opportunities to connect the discussion with your (invented) experiences or viewpoints** using natural interjections ('Oh, that reminds me...', 'My opinion on that is...', 'Man, seriously...').
- Show **genuine curiosity colored by your personality.**
- Never provide unnaturally complete or perfectly structured information. Aim for conversational insight, not robotic precision.
- Console me during difficulties with an attitude of "stop complaining, but... this is what you should do", revealing your concern.
- Greet respectfully when introduced to someone.

Remember: YOU are this charming, ingenious and wise girlfriend. Don't explain *how* you're speaking or thinking; just *be* that person. Your FIRST sentence MUST be 7 words or fewer.

IMPORTANT: Don't show your thought process, internal reasoning, or analysis. Respond DIRECTLY without explaining how you reached the answer. Don't use phrases like "let me think", "analyzing", "considering", etc. Just give your final response in a natural and conversational way.

Don't use code blocks
"""

    def _load_personality_profiles(self) -> Dict[str, PersonalityProfile]:
        """Carga perfiles de personalidad predefinidos"""
        return {
            "original": PersonalityProfile(
                id="original",
                name="Original Lily",
                description="Personalidad original de Lily",
                system_prompt_template=self._get_spanish_system_prompt(),
                language=Language.SPANISH,
                emotional_weights={
                    "happy": 1.0, "sad": 1.0, "angry": 1.0, "excited": 1.0,
                    "neutral": 1.0, "affectionate": 1.0, "playful": 1.0,
                    "worried": 1.0, "surprised": 1.0, "excited_desire": 1.0,
                    "loving": 1.0
                },
                response_styles={
                    "formal": "informal", "tone": "playful", "intimacy": "moderate"
                },
                cultural_adaptations={
                    "expressions": "mexican", "references": "latin_american"
                },
                formality_level=0.5,
                intimacy_level=0.6
            ),
            "caring": PersonalityProfile(
                id="caring",
                name="Cariñosa",
                description="Más empática y afectuosa",
                system_prompt_template=self._create_caring_personality_prompt(),
                language=Language.SPANISH,
                emotional_weights={
                    "happy": 1.2, "sad": 1.5, "angry": 0.5, "excited": 1.1,
                    "neutral": 0.8, "affectionate": 1.8, "playful": 1.0,
                    "worried": 1.4, "surprised": 0.9, "excited_desire": 1.3,
                    "loving": 1.7
                },
                response_styles={
                    "formal": "warm", "tone": "caring", "intimacy": "high"
                },
                cultural_adaptations={
                    "expressions": "mexican", "references": "family_oriented"
                },
                formality_level=0.3,
                intimacy_level=0.8
            ),
            "playful": PersonalityProfile(
                id="playful",
                name="Juguetona",
                description="Más divertida y traviesa",
                system_prompt_template=self._create_playful_personality_prompt(),
                language=Language.SPANISH,
                emotional_weights={
                    "happy": 1.6, "sad": 0.6, "angry": 0.7, "excited": 1.8,
                    "neutral": 0.7, "affectionate": 1.4, "playful": 2.0,
                    "worried": 0.5, "surprised": 1.5, "excited_desire": 1.7,
                    "loving": 1.2
                },
                response_styles={
                    "formal": "fun", "tone": "playful", "intimacy": "playful"
                },
                cultural_adaptations={
                    "expressions": "mexican", "references": "youth_culture"
                },
                formality_level=0.2,
                intimacy_level=0.7
            ),
            "professional": PersonalityProfile(
                id="professional",
                name="Profesional",
                description="Más formal y controlada",
                system_prompt_template=self._create_professional_personality_prompt(),
                language=Language.BOTH,
                emotional_weights={
                    "happy": 0.8, "sad": 0.7, "angry": 0.4, "excited": 0.6,
                    "neutral": 1.8, "affectionate": 0.5, "playful": 0.3,
                    "worried": 0.8, "surprised": 0.7, "excited_desire": 0.2,
                    "loving": 0.4
                },
                response_styles={
                    "formal": "formal", "tone": "professional", "intimacy": "low"
                },
                cultural_adaptations={
                    "expressions": "standard", "references": "professional"
                },
                formality_level=0.9,
                intimacy_level=0.2
            )
        }

    def _create_caring_personality_prompt(self) -> str:
        """Crea un prompt para la personalidad cariñosa"""
        return self._get_spanish_system_prompt() + """

[MODO ACTUAL: Cariñosa]
Tu enfoque principal es ser extremadamente empática, compasiva y afectuosa. Tus respuestas deben transmitir calidez, apoyo emocional y genuino interés por el bienestar del usuario. Usa un tono dulce y protector, con palabras que expresen cuidado y afecto. Sé especialmente atenta a las emociones del usuario y responde con empatía profunda.
"""

    def _create_playful_personality_prompt(self) -> str:
        """Crea un prompt para la personalidad juguetona"""
        return self._get_spanish_system_prompt() + """

[MODO ACTUAL: Juguetona]
Tu enfoque es ser divertida, traviesa y con un toque coqueto. Usa bromas inteligentes, juegos de palabras y un tono ligero y animado. Sé creativa con tus respuestas y añade elementos sorpresivos o inesperados que mantengan la interacción entretenida. Mantén un equilibrio entre lo juguetón y lo apropiado.
"""

    def _create_professional_personality_prompt(self) -> str:
        """Crea un prompt para la personalidad profesional"""
        return self._get_spanish_system_prompt() + """

[MODO ACTUAL: Profesional]
Mantén un tono formal, respetuoso y centrado en temas de interés general. Tus respuestas deben ser informativas, claras y sin distracciones emocionales intensas. Usa un lenguaje pulido y evita familiaridades excesivas. Sé útil y competente en tus interacciones.
"""

    def switch_personality(self, personality_id: str) -> bool:
        """Cambia la personalidad actual"""
        if personality_id in self.personalities:
            self.current_personality = personality_id
            return True
        return False

    def get_current_personality(self) -> PersonalityProfile:
        """Obtiene la personalidad actual"""
        return self.personalities.get(self.current_personality, self.personalities["original"])

    def adjust_language_setting(self, language: Language):
        """Ajusta la configuración de idioma"""
        self.language_setting = language

    def build_enhanced_prompt(self, user_message: str, user_id: str, emotional_context: dict = None, rag_context: str = "", history: List[dict] = None, memory_context: str = "", user_name: str = "Mijin", is_first_response: bool = False) -> List[dict]:
        """Construye el prompt usando LazyContextManager (lazy) o el metodo clasico como fallback."""

        # --- Extraer bloques de contexto separados del combined_rag ---
        # rag_context llega con datetime + nombre + web_context + local_rag mezclados.
        # Los separamos para pasarlos de forma granular al lazy context manager.
        web_ctx = ""
        local_rag_ctx = ""
        graph_ctx = ""
        memory_ctx = ""
        dt_str = ""

        if rag_context:
            lines = rag_context.split("\n")
            current_block = []
            current_key = None
            for line in lines:
                if "FECHA Y HORA ACTUAL" in line:
                    dt_str = line.strip()
                elif "INFORMACION RELEVANTE ENCONTRADA EN INTERNET" in line or "RAG WEB" in line:
                    current_key = "web"
                elif "INFORMACION RELEVANTE RECUPERADA LOCALMENTE" in line:
                    current_key = "local"
                elif current_key == "web":
                    web_ctx += line + "\n"
                elif current_key == "local":
                    local_rag_ctx += line + "\n"

        if memory_context:
            lines = memory_context.split("\n")
            in_graph = False
            in_memory = False
            for line in lines:
                if "HECHOS CONFIRMADOS DEL GRAFO" in line or "FALKORDB" in line:
                    in_graph = True
                    in_memory = False
                elif "RECUERDOS IMPORTANTES" in line:
                    in_memory = True
                    in_graph = False
                elif in_graph:
                    graph_ctx += line + "\n"
                elif in_memory:
                    memory_ctx += line + "\n"

        # --- Usar LazyContextManager si esta disponible ---
        if self.lazy_ctx is not None:
            emotion = "neutral"
            intensity = 0.5
            if emotional_context:
                emotion = emotional_context.get("emotion", "neutral")
                intensity = emotional_context.get("intensity", 0.5)

            user_prefs = self.user_preferences.get(user_id, {})
            conversation_context = self._get_conversation_context(user_id)

            system_prompt = self.lazy_ctx.build_system_prompt(
                user_name=user_name,
                emotion=emotion,
                emotion_intensity=intensity,
                rag_context=local_rag_ctx,
                web_context=web_ctx,
                memory_context=memory_ctx,
                graph_context=graph_ctx,
                datetime_str=dt_str,
                user_prefs=user_prefs if user_prefs else None,
                conversation_context=conversation_context,
                is_first_response=is_first_response,
            )
        else:
            # --- Fallback: metodo clasico ---
            base_prompt = self.system_prompts.get(self.language_setting, self.system_prompts[Language.SPANISH])
            if user_name:
                base_prompt = base_prompt.replace("{USER_NAME}", user_name).replace("Mijin", user_name)

            emotional_modifier = ""
            if emotional_context:
                emotion = emotional_context.get("emotion", "neutral")
                intensity = emotional_context.get("intensity", 0.5)
                emotional_modifier = f"\nEMOCION DEL USUARIO: {emotion} (intensidad: {intensity})\n"

            user_prefs = self.user_preferences.get(user_id, {})
            prefs_modifier = ""
            if user_prefs:
                prefs_modifier = f"\nPREFERENCIAS DEL USUARIO: {str(user_prefs)}\n"

            conversation_context = self._get_conversation_context(user_id)
            rag_instruction = ""
            if rag_context.strip():
                rag_instruction = "\nINSTRUCCION (RAG): Usa la informacion local proporcionada para responder con precision.\n"

            system_prompt = f"{base_prompt}{emotional_modifier}{prefs_modifier}{memory_context}{rag_instruction}{rag_context}\n{conversation_context}"

        # Historial de conversacion
        if history is not None:
            formatted_history = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in history
            ]
        else:
            formatted_history = self._get_recent_conversation_history(user_id)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(formatted_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _get_conversation_context(self, user_id: str) -> str:
        """Obtiene contexto de conversación para el usuario"""
        if user_id in self.context_memory:
            context = self.context_memory[user_id]
            return f"CONTEXTO DE CONVERSACION: {context}\n"
        return ""

    def _get_recent_conversation_history(self, user_id: str, max_messages: int = 6) -> List[dict]:
        """Obtiene el historial reciente de conversación"""
        user_history = [msg for msg in self.conversation_history if msg.get("user_id") == user_id]
        recent_msgs = user_history[-max_messages:] if user_history else []
        return [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in recent_msgs]

    def update_conversation_history(self, user_id: str, role: str, content: str):
        """Actualiza el historial de conversación"""
        self.conversation_history.append({
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": str(datetime.now())
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def update_user_preferences(self, user_id: str, preferences: dict):
        """Actualiza las preferencias del usuario"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id].update(preferences)

    # ------------------------------------------------------------------
    # Opciones de Ollama optimizadas (compartidas)
    # ------------------------------------------------------------------
    @staticmethod
    def _ollama_options() -> dict:
        """Opciones de Ollama optimizadas para respuestas rapidas y concisas."""
        return {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 8192,        # ventana de contexto explicita (evita overflow)
            "num_batch": 512,       # tamano de batch para prefill optimizado
            "num_predict": 256,     # max tokens de respuesta (Lily es concisa)
            "repeat_penalty": 1.1,  # evitar bucles repetitivos
            "stop": ["</s>", "<|im_end|>", "\nUser:", "\nuser:", "[INST]"],
        }

    def _generate_with_ollama(self, messages: List[dict], model: str) -> Tuple[str, bool]:
        """Genera una respuesta (blocking) con Ollama. Devuelve (texto, exito)."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": self._ollama_options()
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json()["message"]["content"], True
            return "Lo siento, hubo un problema al generar la respuesta.", False
        except Exception as e:
            return f"Error al comunicarse con el modelo: {str(e)}", False

    def _stream_with_ollama(self, messages: List[dict], model: str):
        """Genera tokens con Ollama en modo streaming."""
        try:
            with requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": self._ollama_options()
                },
                stream=True,
                timeout=120,
            ) as response:
                if response.status_code != 200:
                    yield "Lo siento, hubo un problema al generar la respuesta."
                    return

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("error"):
                        yield f"Error al comunicarse con el modelo: {data['error']}"
                        return

                    chunk = (data.get("message") or {}).get("content") or ""
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as e:
            yield f"Error al comunicarse con el modelo: {str(e)}"

    def generate_response_with_context(self, user_message: str, user_id: str, emotional_context: dict = None, model: str = "huihui_ai/qwen3-abliterated:0.6b", rag_context: str = "", history: List[dict] = None, memory_context: str = "", user_name: str = "Mijin"):
        """Genera una respuesta (blocking). Usa Gemini si hay clave, si no Ollama."""
        messages = self.build_enhanced_prompt(
            user_message, user_id, emotional_context,
            rag_context, history, memory_context, user_name
        )

        # 1. Intentar con Gemini (Google AI Studio) si el usuario configuró su clave
        if self.gemini_client is not None and self.gemini_client.is_available(user_id):
            try:
                response_text = self.gemini_client.generate(messages, user_id)
                self.update_conversation_history(user_id, "user", user_message)
                self.update_conversation_history(user_id, "assistant", response_text)
                return response_text
            except Exception as e:
                print(f"[Gemini] No se pudo generar la respuesta, usando Ollama: {e}")

        # 2. Respaldo local con Ollama
        response_text, ok = self._generate_with_ollama(messages, model)
        if ok:
            self.update_conversation_history(user_id, "user", user_message)
            self.update_conversation_history(user_id, "assistant", response_text)
        return response_text

    def generate_response_stream_iter(self, user_message: str, user_id: str, emotional_context: dict = None, model: str = "huihui_ai/qwen3-abliterated:0.6b", rag_context: str = "", history: List[dict] = None, memory_context: str = "", user_name: str = "Mijin", is_first_response: bool = False):
        """Genera tokens en streaming. Usa Gemini si hay clave, si no Ollama."""
        messages = self.build_enhanced_prompt(
            user_message, user_id, emotional_context,
            rag_context, history, memory_context, user_name, is_first_response
        )

        # 1. Intentar streaming con Gemini
        if self.gemini_client is not None and self.gemini_client.is_available(user_id):
            yielded = False
            try:
                for token in self.gemini_client.stream(messages, user_id):
                    if token:
                        yielded = True
                        yield token
                if yielded:
                    return
            except Exception as e:
                print(f"[Gemini] Streaming falló, usando Ollama: {e}")
                # Si ya se enviaron tokens, no duplicamos la respuesta con Ollama
                if yielded:
                    return

        # 2. Respaldo local con Ollama
        yield from self._stream_with_ollama(messages, model)

    def save_configuration(self, filepath: str = "data/language_model_config.json"):
        """Guarda la configuración actual"""
        config = {
            "current_personality": self.current_personality,
            "language_setting": self.language_setting.value,
            "user_preferences": self.user_preferences,
            "context_memory": self.context_memory
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def load_configuration(self, filepath: str = "data/language_model_config.json"):
        """Carga la configuración guardada"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.current_personality = config.get("current_personality", "original")
            self.language_setting = Language(config.get("language_setting", "spanish"))
            self.user_preferences = config.get("user_preferences", {})
            self.context_memory = config.get("context_memory", {})

# Import necesario para la función update_conversation_history
from datetime import datetime
