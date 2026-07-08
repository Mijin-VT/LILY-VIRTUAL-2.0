import requests
import json
import os
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from models.advanced_emotional_intelligence import AdvancedEmotionalIntelligence, EmotionalState, EmotionType
from models.memory_system import MemorySystem
from models.rag_engine import RAGEngine
from models.wake_word_engine import WakeWordEngine
from models.youtube_controller import YouTubeController
from models.media_controller import MediaController
from models.language_model_customization import AdvancedLanguageModelCustomization
from models.web_search_controller import WebSearchController
from models.web_search_engine import WebSearchEngine
from models.gmail_controller import GmailController
from models.task_scheduler import TaskScheduler


class AIEngine:
    """Motor de IA que integra Mistral 7B con inteligencia emocional, memoria, planificador, Gmail y búsqueda web"""

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", enable_wake_word: bool = False):
        """
        Inicializa el motor de IA
        """
        self.ollama_url = ollama_url
        self.model = os.environ.get("LILY_MODEL", "mistral:7b")
        self._verify_model()
        
        # Sistemas principales
        self.emotional_intelligence = AdvancedEmotionalIntelligence()
        self.memory_system = MemorySystem()
        self.personalidades = self._load_personalidades()
        self.current_personality = "lily_default"
        
        # Sistema de personalización avanzado
        self.language_model_customization = AdvancedLanguageModelCustomization(ollama_url=ollama_url)
        
        try:
            self.rag_engine = RAGEngine()
        except Exception as e:
            print(f"Advertencia: No se pudo iniciar RAG Engine: {e}")
            self.rag_engine = None

        # Controladores de medios y búsquedas locales
        self.youtube_controller = YouTubeController()
        self.media_controller = MediaController()
        self.web_search_controller = WebSearchController()
        
        # NUEVOS: Planificador, Gmail y Motor de Búsqueda Web
        self.web_search_engine = WebSearchEngine()
        self.gmail_controller = GmailController(memory_system=self.memory_system)
        self.task_scheduler = TaskScheduler(db_path=self.memory_system.db_path)
        print("[OK] Planificador, Gmail y Motor de Búsqueda Web inicializados")

        # Palabra clave básico de 2.0 (SpeechRecognition)
        self.wake_word_enabled = enable_wake_word
        self.wake_word_engine = None

        if self.wake_word_enabled:
            try:
                self.wake_word_engine = WakeWordEngine(
                    wake_word_callback=self.on_wake_word_detected,
                    wake_word="LILY"
                )
                print("[OK] Sistema de palabra clave básico de LILY 2.0 iniciado")
            except Exception as e:
                print(f"[ERROR] Error iniciando sistema de palabra clave: {e}")
                self.wake_word_engine = None
                self.wake_word_enabled = False

        # Cargar system prompt
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(current_dir, "system_prompt.txt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.base_system_prompt = f.read()
        except Exception as e:
            print(f"Error cargando system_prompt.txt: {e}")
            self.base_system_prompt = ""
            
    def _verify_model(self):
        """Verifica que el modelo exista en Ollama; si no, usa el primero disponible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return
            models = [m["name"] for m in response.json().get("models", [])]
            if not models:
                return
            if self.model in models or any(m.split(":")[0] == self.model for m in models):
                print(f"Modelo activo: {self.model}")
                return
            self.model = models[0]
            print(f"Usando el modelo disponible: {self.model}")
        except Exception:
            pass
    
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

    def disable_wake_word_detection(self):
        """Deshabilita la detección de palabra clave"""
        if self.wake_word_enabled and self.wake_word_engine:
            self.wake_word_engine.stop_listening()

    def on_wake_word_detected(self):
        """Callback cuando se detecta la palabra clave"""
        print("¡Palabra clave 'LILY' detectada!")
    
    def process_media_command(self, user_message: str) -> tuple[bool, str]:
        """
        Detecta y ejecuta comandos de control de medios
        """
        import re
        message_lower = user_message.lower().strip()
        message_clean = re.sub(r'[¡!¿?.,;]', '', message_lower).strip()
        
        music_keywords = ["pon música", "reproduce", "ponme música", "play", "escuchar", "música de"]
        if any(keyword in message_lower for keyword in music_keywords):
            query = self._extract_music_query(user_message)
            if query:
                result = self.youtube_controller.search_and_play(query)
                if result["status"] == "success":
                    return True, f"¡Claro Mijin! Buscando {query} en YouTube 🎵"
                else:
                    return True, f"Ay Mijin, hubo un error: {result['message']}"
        
        is_pause = False
        if re.search(r'\b(pausa|pause|detén|detente|detener|pausar)\b', message_lower):
            is_pause = True
        elif message_clean == "para" or re.search(r'\bpara\s+(la\s+música|el\s+video|la\s+canción|la\s+reproducción|el\s+reproductor)\b', message_lower):
            is_pause = True
            
        if is_pause:
            result = self.media_controller.pause_play()
            if result["status"] == "success":
                return True, "¡Listo Mijin! Pausado/Reproduciendo 🎵"
            else:
                return True, f"Ups, no pude pausar: {result['message']}"
        
        if re.search(r'\b(siguiente|next|skip|salta|saltar)\b', message_lower):
            result = self.media_controller.next_video()
            if result["status"] == "success":
                return True, "¡Siguiente video Mijin! ⏭️"
            else:
                return True, f"No pude cambiar: {result['message']}"
        
        if re.search(r'\b(anterior|previous|atrás|regresa|regresar)\b', message_lower):
            result = self.media_controller.previous_video()
            if result["status"] == "success":
                return True, "¡Video anterior Mijin! ⏮️"
            else:
                return True, f"No pude regresar: {result['message']}"
        
        if any(word in message_lower for word in ["sube volumen", "más volumen", "volumen arriba", "sube el volumen"]):
            result = self.media_controller.volume_up(steps=3)
            if result["status"] == "success":
                return True, "¡Volumen subido Mijin! 🔊"
            else:
                return True, f"No pude subir el volumen: {result['message']}"
        
        if any(word in message_lower for word in ["baja volumen", "menos volumen", "volumen abajo", "baja el volumen"]):
            result = self.media_controller.volume_down(steps=3)
            if result["status"] == "success":
                return True, "¡Volumen bajado Mijin! 🔉"
            else:
                return True, f"No pude bajar el volumen: {result['message']}"
        
        if any(word in message_lower for word in ["silencio", "mute", "calla"]):
            result = self.media_controller.mute_unmute()
            if result["status"] == "success":
                return True, "¡Silencio activado/desactivado Mijin! 🔇"
            else:
                return True, f"No pude silenciar: {result['message']}"
        
        return False, ""
    
    def process_search_command(self, user_message: str) -> tuple[bool, str]:
        """
        Detecta y ejecuta comandos de búsqueda local abriendo el navegador
        """
        message_lower = user_message.lower()
        search_keywords = ["busca en internet", "busca en google", "busca información de", "busca información sobre", "buscame la receta de", "buscame", "busca", "investiga", "googlea"]
        
        if any(keyword in message_lower for keyword in search_keywords):
            # Evitar conflictos con comandos de RAG semántico en la web
            rag_keywords = ["busca en internet sobre", "busca en internet de", "busca en la web sobre", "busca en la web de", "investiga en internet sobre", "busca noticias de"]
            if any(r_kw in message_lower for r_kw in rag_keywords):
                return False, "" # Permitir que lo maneje el RAG de búsqueda semántica web
                
            query = self._extract_search_query(user_message)
            if query:
                music_keywords = ["pon música", "reproduce", "ponme música", "play", "escuchar", "música de"]
                if any(m_kw in message_lower for m_kw in music_keywords):
                    return False, ""
                
                result = self.web_search_controller.search(query)
                if result["status"] == "success":
                    return True, f"¡Claro Mijin! Buscando {query} en Google 🔍"
                else:
                    return True, f"Ay Mijin, no pude hacer la búsqueda: {result['message']}"
        
        return False, ""

    def process_gmail_command(self, user_message: str, user_id: str) -> tuple[bool, str]:
        """
        Detecta y ejecuta comandos relacionados con Gmail (enviar y leer correos)
        """
        message_lower = user_message.lower()
        
        # 1. Comprobar bandeja de entrada
        check_keywords = ["revisa mis correos", "tengo correos nuevos", "leer correos", "revisar emails", "revisar bandeja"]
        if any(kw in message_lower for kw in check_keywords):
            res = self.gmail_controller.check_emails(user_id)
            if res["status"] == "error":
                return True, f"Ay Mijin, no pude revisar los correos: {res['message']}"
            
            emails = res.get("emails", [])
            count = res.get("count", 0)
            
            if count == 0:
                return True, "No tienes correos nuevos sin leer, Mijin. 📬"
                
            summary_list = [f"Mijin, tienes {count} correos sin leer. Aquí están los últimos {len(emails)}:"]
            for idx, mail in enumerate(emails, 1):
                summary_list.append(f"{idx}. De **{mail['from']}** | Asunto: *{mail['subject']}* \n   > {mail['snippet']}")
            return True, "\n".join(summary_list)
            
        # 2. Enviar correo electrónico
        send_keywords = ["envía un correo", "manda un email", "enviar un mail", "enviar correo", "manda correo"]
        if any(kw in message_lower for kw in send_keywords):
            # Extraer dirección de correo
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
            if not email_match:
                return True, "Mijin, por favor indícame una dirección de correo válida para enviarlo."
                
            to_email = email_match.group(0)
            
            # Extraer asunto
            subject = "Mensaje de Lily AI"
            subject_match = re.search(r'asunto\s+([^cuerpomsgque]+)', user_message, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(1).strip()
                
            # Extraer cuerpo/mensaje
            body = ""
            body_match = re.search(r'(cuerpo|mensaje|que diga)\s+(.+)$', user_message, re.IGNORECASE)
            if body_match:
                body = body_match.group(2).strip()
            else:
                body = user_message
                
            res = self.gmail_controller.send_email(user_id, to_email, subject, body)
            if res["status"] == "success":
                return True, f"¡Listo Mijin! Correo enviado a {to_email} con el asunto '{subject}' ✉️"
            else:
                return True, f"No pude enviar el correo: {res['message']}"
                
        return False, ""

    def process_scheduler_command(self, user_message: str, user_id: str) -> tuple[bool, str]:
        """
        Detecta y ejecuta comandos de programación de tareas, recordatorios y ejecución de comandos
        """
        message_lower = user_message.lower()
        
        # 1. Cancelar/Eliminar tareas
        cancel_keywords = ["cancela la tarea", "borra el recordatorio", "elimina el recordatorio", "eliminar tarea", "cancelar recordatorio"]
        if any(kw in message_lower for kw in cancel_keywords):
            id_match = re.search(r'\b\d+\b', user_message)
            if not id_match:
                return True, "Indícame el número o ID de la tarea que quieres cancelar, Mijin."
            task_id = int(id_match.group(0))
            success = self.task_scheduler.cancel_task(task_id)
            if success:
                return True, f"¡Hecho Mijin! Tarea con ID {task_id} cancelada correctamente."
            else:
                return True, f"Mijin, no encontré ninguna tarea activa con el ID {task_id}."
                
        # 2. Listar tareas activas
        list_keywords = ["qué tareas tengo", "lista mis tareas", "recordatorios activos", "tareas programadas", "lista de recordatorios"]
        if any(kw in message_lower for kw in list_keywords):
            tasks = self.task_scheduler.get_active_tasks(user_id)
            if not tasks:
                return True, "No tienes tareas ni recordatorios activos programados en este momento, Mijin."
                
            task_list = ["Mijin, aquí están tus tareas activas programadas:"]
            for t in tasks:
                t_type = "Recordatorio ⏰" if t["type"] == "reminder" else "Comando 🖥️"
                interval_str = f" (repite cada {t['interval_seconds']} segs)" if t["interval_seconds"] > 0 else ""
                try:
                    dt = datetime.fromisoformat(t["run_at"])
                    date_str = dt.strftime('%Y-%m-%d a las %H:%M')
                except:
                    date_str = t["run_at"]
                    
                task_list.append(f"- **ID {t['id']}** | {t_type}: *{t['description']}* programado para el {date_str}{interval_str}")
            return True, "\n".join(task_list)
            
        # 3. Programar ejecución de comandos
        run_keywords = ["ejecuta el comando", "corre el comando", "ejecutar comando", "programa el comando"]
        if any(kw in message_lower for kw in run_keywords):
            cmd_match = re.search(r'(?:comando|corre|ejecuta)\s+`?([^`]+)`?', user_message, re.IGNORECASE)
            if not cmd_match:
                return True, "Indícame el comando del sistema que quieres que ejecute, Mijin."
            command = cmd_match.group(1).strip()
            run_at, interval, _ = self._parse_time_from_message(user_message)
            
            self.task_scheduler.add_command_task(user_id, command, run_at, interval)
            time_str = run_at.strftime('%H:%M:%S')
            interval_str = f" repetitivo cada {interval} segundos" if interval > 0 else ""
            return True, f"¡Programado Mijin! Ejecutaré el comando `{command}` a las {time_str}{interval_str} 🖥️"

        # 4. Programar recordatorios
        reminder_keywords = ["recuérdame", "programa un recordatorio", "recuerdame", "recordatorio de", "avísame"]
        if any(kw in message_lower for kw in reminder_keywords):
            run_at, interval, description = self._parse_time_from_message(user_message)
            clean_desc = re.sub(r'^(recuérdame que|recuerdame que|recuérdame de|recuerdame de|recuérdame|recuerdame|recordatorio de|recordatorio|avísame de|avisame)\s+', '', description, flags=re.IGNORECASE).strip()
            
            self.task_scheduler.add_reminder(user_id, clean_desc, run_at, interval)
            time_str = run_at.strftime('%H:%M:%S')
            interval_str = f" repetitivo cada {interval} segundos" if interval > 0 else ""
            return True, f"¡Listo Mijin! Te recordaré '{clean_desc}' a las {time_str}{interval_str} ⏰"
            
        return False, ""

    def process_semantic_web_search(self, user_message: str) -> str:
        """
        Comprueba si la consulta requiere una búsqueda web semántica RAG
        y devuelve la información recuperada de internet.
        """
        message_lower = user_message.lower()
        search_keywords = ["busca en internet sobre", "busca en internet de", "busca en la web sobre", "busca en la web de", "investiga en internet sobre", "busca noticias de", "busca en internet", "busca en la web"]
        
        if any(kw in message_lower for kw in search_keywords):
            query = ""
            for kw in search_keywords:
                if kw in message_lower:
                    idx = message_lower.index(kw)
                    query = user_message[idx+len(kw):].strip()
                    query = re.sub(r'[¿?¡!.,;]+$', '', query).strip()
                    break
                    
            if query:
                results = self.web_search_engine.search(query, limit=3)
                if results:
                    content_blocks = []
                    for r in results:
                        content_blocks.append(f"Título: {r['title']}\nContenido: {r['snippet']}\nFuente: {r['url']}\n")
                    return "\nINFORMACIÓN EN TIEMPO REAL DESDE LA WEB (RAG WEB):\n" + "\n".join(content_blocks) + "\n"
        return ""

    def _parse_time_from_message(self, message: str) -> tuple[Optional[datetime], int, str]:
        """
        Extrae fecha/hora, intervalo y descripción.
        """
        now = datetime.now()
        run_at = None
        interval_seconds = 0
        description = message
        
        recurrent_match = re.search(r'cada (\d+) (hora|minuto|segundo)s?', message, re.IGNORECASE)
        if recurrent_match:
            value = int(recurrent_match.group(1))
            unit = recurrent_match.group(2).lower()
            
            if "hora" in unit:
                interval_seconds = value * 3600
                run_at = now + timedelta(hours=value)
            elif "minuto" in unit:
                interval_seconds = value * 60
                run_at = now + timedelta(minutes=value)
            else:
                interval_seconds = value
                run_at = now + timedelta(seconds=value)
                
            description = message[:recurrent_match.start()].strip()
            return run_at, interval_seconds, description
            
        relative_match = re.search(r'en (\d+) (minuto|hora|segundo)s?', message, re.IGNORECASE)
        if relative_match:
            value = int(relative_match.group(1))
            unit = relative_match.group(2).lower()
            
            if "minuto" in unit:
                run_at = now + timedelta(minutes=value)
            elif "hora" in unit:
                run_at = now + timedelta(hours=value)
            else:
                run_at = now + timedelta(seconds=value)
                
            description = message[:relative_match.start()].strip()
            return run_at, interval_seconds, description
            
        time_match = re.search(r'a las (\d{1,2})(?::(\d{2}))?\s*(am|pm)?', message, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3)
            
            if am_pm:
                am_pm = am_pm.lower()
                if am_pm == "pm" and hour < 12:
                    hour += 12
                elif am_pm == "am" and hour == 12:
                    hour = 0
            
            run_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if run_at <= now:
                run_at += timedelta(days=1)
                
            description = message[:time_match.start()].strip()
            return run_at, interval_seconds, description
            
        run_at = now + timedelta(minutes=1)
        return run_at, interval_seconds, description
    
    def _extract_music_query(self, message: str) -> str:
        message_lower = message.lower()
        keywords = [
            "pon música de", "pon música", "ponme música de", "ponme música",
            "reproduce", "play", "escuchar", "música de", "pon", "ponme"
        ]
        for keyword in keywords:
            if keyword in message_lower:
                parts = message_lower.split(keyword, 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    if query:
                        return query
        return message.strip()

    def _extract_search_query(self, message: str) -> str:
        message_lower = message.lower()
        keywords = [
            "busca en internet sobre", "busca en internet de", "busca en internet", 
            "busca en google sobre", "busca en google de", "busca en google", 
            "busca información sobre", "busca información de", "busca información",
            "buscame la receta de", "buscame la receta", "buscame sobre", "buscame de", 
            "buscame", "busca sobre", "busca de", "busca", "investiga sobre", 
            "investiga de", "investiga", "googlea sobre", "googlea de", "googlea"
        ]
        for keyword in keywords:
            if keyword in message_lower:
                idx = message_lower.index(keyword)
                query = message[idx + len(keyword):].strip()
                query = re.sub(r'[¿?¡!.,;]+$', '', query).strip()
                if query:
                    return query
        return message.strip()
    
    def _load_personalidades(self):
        try:
            with open("data/personalidades.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"lily_default": {"prompt": self.base_system_prompt}}

    def set_personality(self, personality_id: str):
        if personality_id in self.personalidades:
            self.current_personality = personality_id
            return True
        return False
    
    def generate_response(self, user_message: str, user_id: str = "default_user") -> tuple[str, EmotionType]:
        """Genera una respuesta usando Mistral 7B con contexto emocional, Gmail, planificador y búsqueda web"""

        # Obtener nombre del usuario personalizado para concienciar a Lily
        user_name = self.memory_system.get_preference(user_id, "user_name", "Mijin")

        try:


            # 1. Comandos de Medios
            is_media_command, media_response = self.process_media_command(user_message)
            if is_media_command:
                return media_response.replace("Mijin", user_name), EmotionType.EMOCIONADA

            # 2. Comandos de Gmail
            is_gmail, gmail_res = self.process_gmail_command(user_message, user_id)
            if is_gmail:
                return gmail_res.replace("Mijin", user_name), EmotionType.FELIZ

            # 3. Comandos del Planificador (Reminders/Commands)
            is_schedule, schedule_res = self.process_scheduler_command(user_message, user_id)
            if is_schedule:
                return schedule_res.replace("Mijin", user_name), EmotionType.FELIZ

            # 4. Comandos de búsqueda local abriendo navegador
            is_search_command, search_response = self.process_search_command(user_message)
            if is_search_command:
                return search_response.replace("Mijin", user_name), EmotionType.FELIZ

            # Actualizar estado emocional
            emotional_state = self.emotional_intelligence.update_emotional_state(user_message, user_id)
            emotional_context = {
                'emotion': emotional_state.emotion.value,
                'intensity': emotional_state.intensity,
                'reason': emotional_state.reason,
                'confidence': emotional_state.confidence
            }

            # 5. Obtener contexto de búsqueda web semántica RAG si corresponde
            web_context = self.process_semantic_web_search(user_message)

            # Obtener contexto local RAG
            rag_context = ""
            
            # Palabras clave que indican una consulta web/comercial prioritaria (saltando base de datos local)
            web_priority_keywords = [
                "busca", "búscame", "buscame", "encuentra", "precio", "comprar", "costo", 
                "cuánto cuesta", "cuanto cuesta", "noticias", "clima", "dólar", "dolar", 
                "ofertas", "oferta", "dónde comprar", "donde comprar", "venden", "monitor",
                "laptop", "celular", "iphone", "tienda", "mercado", "precio de"
            ]
            
            message_clean = user_message.lower().strip()
            is_web_priority = any(kw in message_clean for kw in web_priority_keywords)
            
            if self.rag_engine and not is_web_priority:
                rag_docs, _ = self.rag_engine.query(user_message, n_results=2)
                if rag_docs:
                    rag_context = "\nINFORMACIÓN RELEVANTE RECUPERADA LOCALMENTE:\n" + "\n".join(rag_docs) + "\n"
            
            # Si no hay contexto local (o es prioridad web), intentar búsqueda web como fallback
            if not rag_context and not web_context:
                # Palabras/frases conversacionales cotidianas o personales que NO deben disparar búsquedas web
                conversational_keywords = [
                    "hola", "buenos dias", "buenas tardes", "buenas noches", "como estas", "cómo estás",
                    "que haces", "qué haces", "quien eres", "quién eres", "te amo", "te quiero", "gracias",
                    "de nada", "adios", "chao", "bye", "ok", "vale", "jajaja", "jaja", "jeje", "entendido",
                    "entiendo", "claro", "conmigo", "novia", "chiste", "cuéntame", "cuentame", "lily",
                    "amor", "cariño", "linda", "hermosa", "tonto", "tonta", "odio", "gustas", "gusta"
                ]
                
                is_conversational = any(kw in message_clean for kw in conversational_keywords)
                words = user_message.strip().split()
                is_question = any(q_word in message_clean for q_word in ["?", "cómo", "como", "qué", "que", "quién", "quien", "dónde", "donde", "cuándo", "cuando", "por qué", "porque", "cuál", "cual", "cuánto", "cuanto", "dime", "explica", "quiénes", "quienes", "cuáles", "cuales"])
                
                # Buscar en la web si es prioridad de internet o si parece una consulta informativa real
                if not is_conversational and (is_web_priority or len(words) > 3 or is_question):
                    reason = "prioridad de internet" if is_web_priority else "fallback de RAG local"
                    print(f"Buscando automáticamente en internet ({reason}) para: '{user_message}'")
                    web_results = self.web_search_engine.search(user_message, limit=3)
                    if web_results:
                        content_blocks = []
                        for r in web_results:
                            content_blocks.append(f"Título: {r['title']}\nContenido: {r['snippet']}\nFuente: {r['url']}\n")
                        web_context = "\nINFORMACIÓN RELEVANTE ENCONTRADA EN INTERNET (RAG WEB FALLBACK):\n" + "\n".join(content_blocks) + "\n"

            # 6. Conciencia de Fecha y Hora del sistema
            days_es = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
            day_name = days_es.get(datetime.now().strftime('%A'), datetime.now().strftime('%A'))
            current_time_str = f"FECHA Y HORA ACTUAL DEL SISTEMA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Día: {day_name})"
            datetime_context = f"\n{current_time_str}\n"

            name_context = f"El nombre del usuario actual es '{user_name}'. Te estás dirigiendo a él y debes llamarle '{user_name}' en tus respuestas.\n"

            # Mezclar contextos RAG local, RAG web, tiempo y nombre
            combined_rag = datetime_context + name_context + web_context + rag_context

            # Obtener memorias semánticas del pasado
            semantic_memories = self.memory_system.get_semantic_memories(user_id, user_message)
            memory_context = ""
            if semantic_memories:
                memory_context = "\nRECUERDOS IMPORTANTES DEL PASADO RELATADOS POR EL USUARIO:\n" + "\n".join([f"- {m}" for m in semantic_memories]) + "\n"

            # Obtener historial reciente de SQLite
            recent_history = self.memory_system.get_conversation_context(user_id, max_messages=8)

            # Generar respuesta
            response_text = self.language_model_customization.generate_response_with_context(
                user_message, 
                user_id, 
                emotional_context,
                model=self.model,
                rag_context=combined_rag,
                history=recent_history,
                memory_context=memory_context,
                user_name=user_name
            )

            # Guardar en memoria
            self.memory_system.add_message(user_id, "user", user_message)
            self.memory_system.add_message(user_id, "assistant", response_text, emotional_state.emotion.value, ai_engine=self)

            # Guardar en RAG local
            if self.rag_engine:
                self.rag_engine.add_conversation_turn(user_message, response_text)

            return response_text, emotional_state.emotion

        except requests.exceptions.Timeout:
            return f"Lo siento {user_name}, estoy tardando mucho en pensar... ¿Podrías repetir eso?", EmotionType.PREOCUPADA
        except Exception as e:
            return f"Ay {user_name}, algo salió mal: {str(e)}", EmotionType.PREOCUPADA
    
    def get_current_emotion(self) -> EmotionType:
        return self.emotional_intelligence.current_state.emotion

    def get_emotional_state(self):
        return self.emotional_intelligence.current_state
    
    def get_emotional_summary(self, user_id: str = "default_user") -> str:
        return self.emotional_intelligence.get_emotional_summary(user_id)
    
    def set_emotional_personality(self, personality_id: str) -> bool:
        return self.emotional_intelligence.set_personality(personality_id)
    
    def get_current_emotional_personality(self) -> dict:
        return self.emotional_intelligence.get_current_personality_info()
    
    def switch_language_model_personality(self, personality_id: str) -> bool:
        return self.language_model_customization.switch_personality(personality_id)
