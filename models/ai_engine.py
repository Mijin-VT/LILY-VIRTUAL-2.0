import requests
import json
import os
from typing import List, Dict, Optional
from models.emotional_intelligence import EmotionalIntelligence
from models.memory_system import MemorySystem
from models.rag_engine import RAGEngine
from models.schemas import EmotionType
from models.wake_word_engine import WakeWordEngine


class AIEngine:
    """Motor de IA que integra Mistral 7B con inteligencia emocional y memoria"""
    
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", enable_wake_word: bool = False):
        self.ollama_url = ollama_url
        self.model = "qwen3-coder:480b-cloud"     # LLM empleado mistral:7b
        self.emotional_intelligence = EmotionalIntelligence()
        self.memory_system = MemorySystem()
        try:
            self.rag_engine = RAGEngine()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar RAG Engine: {e}")
            self.rag_engine = None

        # Iniciar sistema de detección de palabra clave
        self.wake_word_enabled = enable_wake_word
        if self.wake_word_enabled:
            try:
                self.wake_word_engine = WakeWordEngine(self.on_wake_word_detected, wake_word="LILY")
                print("Sistema de palabra clave iniciado")
            except Exception as e:
                print(f"Error iniciando sistema de palabra clave: {e}")
                self.wake_word_engine = None
                self.wake_word_enabled = False

        # Cargar system prompt desde archivo
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(current_dir, "system_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.base_system_prompt = f.read()
        except Exception as e:
            print(f"Error cargando system_prompt.txt: {e}")
            # Fallback en caso de error
            print(f"La IA no tiene system prompt")
            self.base_system_prompt = ""
    
    def check_ollama_connection(self) -> bool:
        """Verifica la conexión con Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def enable_wake_word_detection(self):
        """Habilita la detección de palabra clave"""
        if self.wake_word_enabled and self.wake_word_engine:
            self.wake_word_engine.start_listening()
            print("Detección de palabra clave habilitada")

    def disable_wake_word_detection(self):
        """Deshabilita la detección de palabra clave"""
        if self.wake_word_enabled and self.wake_word_engine:
            self.wake_word_engine.stop_listening()
            print("Detección de palabra clave deshabilitada")

    def on_wake_word_detected(self):
        """Callback cuando se detecta la palabra clave"""
        print("¡Palabra clave 'LILY' detectada!")
        # Aquí puedes implementar la lógica que desees cuando se detecte la palabra clave
        # Por ejemplo, iniciar grabación de audio, mostrar una notificación, etc.
        # Por ejemplo, podrías generar una respuesta de activación
        try:
            response_text = "¡Hola! Como estas el dia de hoy?."
            # Aquí podrías generar un audio de respuesta de activación si lo deseas
            # Por ejemplo: audio_url = self.tts_engine.text_to_speech(response_text, emotion="neutral")
            print(f"Lily responde: {response_text}")
        except Exception as e:
            print(f"Error en la respuesta de activación: {e}")
    
    def build_prompt(self, user_message: str, user_id: str) -> List[Dict[str, str]]:
        """Construye el prompt con contexto emocional y memoria"""
        
        # Actualizar estado emocional
        emotional_state = self.emotional_intelligence.update_emotional_state(user_message)
        self.memory_system.add_emotional_state(user_id, emotional_state)
        
        # Obtener modificador emocional
        emotional_modifier = self.emotional_intelligence.get_emotional_modifier()
        
        # Obtener contexto de conversación
        conversation_context = self.memory_system.get_conversation_context(user_id, max_messages=6)
        emotional_summary = self.memory_system.get_emotional_summary(user_id)
        conversation_summary = self.memory_system.get_conversation_summary(user_id)
        
        # Obtener contexto RAG
        rag_context = ""
        if self.rag_engine:
            rag_docs = self.rag_engine.query(user_message, n_results=2)
            if rag_docs:
                rag_context = "INFORMACIÓN RELEVANTE RECUPERADA:\n" + "\n".join(rag_docs) + "\n"

        # Construir system prompt con contexto
        system_prompt = f"""{self.base_system_prompt}

CONTEXTO EMOCIONAL ACTUAL:
{emotional_modifier}
Tu emoción actual: {emotional_state.emotion.value} (intensidad: {emotional_state.intensity:.2f})
Razón: {emotional_state.reason}

MEMORIA DE CONVERSACIÓN:
{conversation_summary}
{emotional_summary}
{rag_context}"""
        
        # Construir mensajes
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar contexto de conversación previa
        for msg in conversation_context:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Agregar mensaje actual del usuario
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def generate_response(self, user_message: str, user_id: str = "default_user") -> tuple[str, EmotionType]:
        """Genera una respuesta usando Mistral 7B con contexto emocional"""
        
        try:
            # Construir prompt con contexto
            messages = self.build_prompt(user_message, user_id)
            
            # Llamar a Ollama
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result["message"]["content"]
                
                # Eliminar bloques de pensamiento <think>...</think>
                import re
                assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
                assistant_response = assistant_response.strip()
                
                # Guardar en memoria
                self.memory_system.add_message(user_id, "user", user_message)
                self.memory_system.add_message(
                    user_id, 
                    "assistant", 
                    assistant_response, 
                    self.emotional_intelligence.current_state.emotion.value
                )
                
                # Guardar en RAG
                if self.rag_engine:
                    self.rag_engine.add_conversation_turn(user_message, assistant_response)
                
                return assistant_response, self.emotional_intelligence.current_state.emotion
            else:
                return f"Error al conectar con el modelo: {response.status_code}", EmotionType.NEUTRAL
                
        except requests.exceptions.Timeout:
            return "Lo siento Mijin, estoy tardando mucho en pensar... ¿Podrías repetir eso?", EmotionType.PREOCUPADA
        except Exception as e:
            return f"Ay Mijin, algo salió mal: {str(e)}", EmotionType.PREOCUPADA
    
    def get_current_emotion(self) -> EmotionType:
        """Obtiene la emoción actual"""
        return self.emotional_intelligence.current_state.emotion
    
    def get_emotional_state(self):
        """Obtiene el estado emocional completo"""
        return self.emotional_intelligence.current_state


