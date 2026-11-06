import os
import re
import unicodedata
import hashlib
import random
import numpy as np
import requests
import traceback
from gtts import gTTS
from pydub import AudioSegment

try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False


class TTSEngine:
    """Motor de texto a voz con personalización de voz usando Kokoro y fallback a gTTS"""

    _EMOJI_RE = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # banderas (indicadores regionales)
        "\U0001F300-\U0001F5FF"  # simbolos y pictogramas
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F680-\U0001F6FF"  # transporte y mapas
        "\U0001F700-\U0001F77F"  # alquimia
        "\U0001F780-\U0001F7FF"  # formas geometricas extendidas
        "\U0001F800-\U0001F8FF"  # flechas suplementarias-C
        "\U0001F900-\U0001F9FF"  # simbolos y pictogramas suplementarios
        "\U0001FA00-\U0001FAFF"  # simbolos y pictogramas extendidos-A
        "\U0001F000-\U0001F02F"  # mahjong
        "\U0001F0A0-\U0001F0FF"  # cartas
        "\U0001F100-\U0001F1FF"  # alfanumericos encerrados
        "\U0001F200-\U0001F2FF"  # ideogramas encerrados
        "\U00002600-\U000026FF"  # simbolos varios
        "\U00002700-\U000027BF"  # dingbats
        "\U00002B00-\U00002BFF"  # simbolos y flechas varios
        "\U00002300-\U000023FF"  # simbolos tecnicos
        "\U00002500-\U000025FF"  # dibujo de cajas, bloques y formas
        "\U0000FE00-\U0000FE0F"  # selectores de variacion
        "\U0001F3FB-\U0001F3FF"  # tonos de piel
        "\U0000200D"             # zero-width joiner
        "\U000020E3"             # keycap
        "\U000024C2"             # M encerrada
        "\U00003030"             # ola
        "\U0000303D"             # marca de alternancia
        "\U00003297"             # ideograma encerrado
        "\U00003299"             # ideograma encerrado
        "\U0001F170-\U0001F171"
        "\U0001F17E-\U0001F17F"
        "\U0001F18E"
        "\U0001F191-\U0001F19A"
        "\U0001F201-\U0001F202"
        "\U0001F21A"
        "\U0001F22F"
        "\U0001F232-\U0001F23A"
        "\U0001F250-\U0001F251"
        "\U0000FE30-\U0000FE4F"  # formas de compatibilidad CJK (kaomoji)
        "\U000E0020-\U000E007F"  # etiquetas de emoji
        "\U000E0100-\U000E01EF"  # selectores de variacion suplementarios
        "]+",
        flags=re.UNICODE,
    )

    # Caracteres invisibles de formato (ZWSP, BOM, marcas direccionales...).
    _FORMAT_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")

    # Secuencias keycap completas (1️⃣, #️⃣, *️⃣).
    _KEYCAP_RE = re.compile(r"[0-9#*]\ufe0f?\u20e3")

    # Emoticonos de texto: :) :D :P :( ;) XD uwu owo >w< T_T ^_^ <3 ...
    _EMOTICON_RE = re.compile(
        r"(?<!\w)(?:"
        r"[:;=8][\-o'^]?[\)\]\(\[dDpPbBxX/\\|3oOvV0]+"
        r"|(?:x|X)[dD]"
        r"|(?:>|<|T|t|Q|q|W|w)[._\-]?(?:<|>|T|t|Q|q|W|w)(?:<|>|T|t|Q|q|W|w)?"
        r"|\^[_\-]?\^"
        r"|(?:u|o)w(?:u|o)"
        r"|</?3"
        r")(?!\w)",
        flags=re.IGNORECASE,
    )

    # Kaomoji con el rostro (ツ), con o sin el marco ¯\_(...)_/¯.
    _KAOMOJI_RE = re.compile(r"(?:\s*(?:¯|\u0304))?\\?_?\(ツ\)_?/?(?:\s*(?:¯|\u0304))?")

    # Parentesis/corchetes que solo contenian simbolos (kaomoji).
    _SYMBOL_BRACKETS_RE = re.compile(r"\([\W_]*\)|\[[\W_]*\]|\{[\W_]*\}")
    _EMPTY_BRACKETS_RE = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")

    # URLs y marcas de Markdown que el TTS no debe leer.
    _URL_RE = re.compile(r"<?(?:https?://|www\.)[^\s<>]+>?", re.IGNORECASE)
    _MARKDOWN_RE = re.compile(r"[*_~`]+")
    _LINE_PREFIX_RE = re.compile(r"(?m)^\s*[#>]+\s*")

    @classmethod
    def clean_text_for_tts(cls, text: str) -> str:
        """Devuelve el texto listo para voz, sin emotes ni marcas de formato.

        El texto del chat no se modifica: esto solo se aplica antes de
        sintetizar audio. Si el resultado queda vacio, no se genera audio.
        """
        if not text:
            return ""

        cleaned = unicodedata.normalize("NFKC", str(text))
        cleaned = cls._URL_RE.sub(" ", cleaned)
        cleaned = cls._KEYCAP_RE.sub(" ", cleaned)
        cleaned = cls._EMOJI_RE.sub(" ", cleaned)
        cleaned = cls._FORMAT_RE.sub("", cleaned)
        cleaned = cls._KAOMOJI_RE.sub(" ", cleaned)
        cleaned = cls._EMOTICON_RE.sub(" ", cleaned)
        cleaned = cls._SYMBOL_BRACKETS_RE.sub(" ", cleaned)
        cleaned = cls._EMPTY_BRACKETS_RE.sub(" ", cleaned)
        cleaned = cls._MARKDOWN_RE.sub(" ", cleaned)
        cleaned = cls._LINE_PREFIX_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def __init__(self, audio_dir: str = "static/audio", reference_audio_dir: str = "audio_samples"):
        self.audio_dir = audio_dir
        self.reference_audio_dir = reference_audio_dir
        os.makedirs(audio_dir, exist_ok=True)

        # Rutas de modelos Kokoro
        self.model_path = os.path.join("models", "kokoro-v1.0.onnx")
        self.voices_path = os.path.join("models", "voices-v1.0.bin")
        self.kokoro = None

        # Parámetros de voz basados en análisis de muestras
        self.voice_params = {
            "speed": 1.1,  # Ligeramente más rápido
            "pitch_shift": 2,  # Tono más alto (voz femenina)
            "emotion_modifiers": {
                "feliz": {"speed": 1.15, "pitch": 3},
                "triste": {"speed": 0.9, "pitch": -1},
                "enojada": {"speed": 1.2, "pitch": 1},
                "emocionada": {"speed": 1.25, "pitch": 4},
                "neutral": {"speed": 1.1, "pitch": 2},
                "cariñosa": {"speed": 1.0, "pitch": 3},
                "juguetona": {"speed": 1.2, "pitch": 3},
                "preocupada": {"speed": 0.95, "pitch": 1},
                "sorprendida": {"speed": 1.15, "pitch": 4}
            }
        }

        # Inicializar Kokoro
        if KOKORO_AVAILABLE:
            self.init_kokoro()
        else:
            print("Librería kokoro-onnx no disponible. Usando gTTS de forma predeterminada.")

    def download_models(self):
        """Descarga los modelos necesarios de Kokoro desde GitHub de forma segura"""
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

        os.makedirs("models", exist_ok=True)

        def download_file(url, dest):
            if os.path.exists(dest):
                return
            print(f"Descargando archivo necesario de Kokoro: {os.path.basename(dest)}...")
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Archivo {os.path.basename(dest)} descargado con éxito.")
            except Exception as e:
                print(f"Error al descargar {url}: {e}")
                if os.path.exists(dest):
                    os.remove(dest)
                raise e

        try:
            download_file(model_url, self.model_path)
            download_file(voices_url, self.voices_path)
        except Exception as e:
            print(f"Fallo al descargar los archivos del modelo Kokoro: {e}")
            raise e

    def init_kokoro(self):
        """Inicializa el motor de Kokoro TTS, descargando los modelos si es necesario"""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
                self.download_models()

            print("Cargando modelo Kokoro TTS en ONNX...")
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            print("Motor Kokoro TTS cargado exitosamente.")
        except Exception as e:
            print(f"No se pudo inicializar Kokoro TTS. Se utilizará gTTS como alternativa. Error: {e}")
            self.kokoro = None

    def generate_audio_filename(self, text: str) -> str:
        """Genera un nombre de archivo único basado en el texto y timestamp"""
        import time
        timestamp = str(int(time.time() * 1000))
        text_hash = hashlib.md5((text + timestamp).encode()).hexdigest()[:8]
        return f"lily_{timestamp}_{text_hash}.mp3"

    def text_to_speech(self, text: str, emotion: str = "neutral", save: bool = True) -> str:
        """
        Convierte texto a voz con características emocionales usando Kokoro y fallback a gTTS

        Args:
            text: Texto a convertir
            emotion: Emoción para modular la voz
            save: Si guardar el archivo o no

        Returns:
            Ruta del archivo de audio generado
        """
        try:
            # Quitar emojis, emoticonos y marcas de formato antes de hablar.
            text = self.clean_text_for_tts(text)
            if not text:
                print("[TTS] Texto sin contenido hablable (solo emotes); se omite el audio.")
                return None

            # Generar nombre de archivo
            filename = self.generate_audio_filename(f"{text}_{emotion}")
            output_path = os.path.join(self.audio_dir, filename)

            # Si ya existe, retornar
            if os.path.exists(output_path):
                return f"/static/audio/{filename}"

            # Obtener parámetros de emoción
            emotion_params = self.voice_params["emotion_modifiers"].get(
                emotion,
                self.voice_params["emotion_modifiers"]["neutral"]
            )

            # 1. INTENTO CON KOKORO ONNX (LOCAL)
            if self.kokoro:
                try:
                    # Generar audio con Kokoro
                    samples, sample_rate = self.kokoro.create(
                        text,
                        voice="ef_dora",
                        speed=emotion_params["speed"],
                        lang="es"
                    )

                    # Convertir array float32 de numpy a 16-bit PCM bytes
                    audio_bytes = (samples * 32767).astype(np.int16).tobytes()

                    # Cargar a pydub
                    audio = AudioSegment(
                        data=audio_bytes,
                        sample_width=2,
                        frame_rate=sample_rate,
                        channels=1
                    )

                    # Ajustar volumen según emoción
                    volume_adjustments = {
                        "feliz": 2,
                        "enojada": 3,
                        "emocionada": 4,
                        "triste": -2,
                        "preocupada": -1
                    }

                    volume_change = volume_adjustments.get(emotion, 0)
                    if volume_change != 0:
                        audio = audio + volume_change

                    # Exportar archivo procesado como MP3
                    audio.export(output_path, format="mp3", bitrate="128k")
                    return f"/static/audio/{filename}"

                except Exception as ex_kokoro:
                    print(f"Error generando audio con Kokoro: {ex_kokoro}. Recurriendo a gTTS de respaldo...")
                    # Continuar al fallback de gTTS

            # 2. FALLBACK A gTTS (ONLINE)
            temp_file = os.path.join(self.audio_dir, f"temp_{filename}")
            tts = gTTS(text=text, lang='es', slow=False)
            tts.save(temp_file)

            # Cargar audio con pydub
            audio = AudioSegment.from_mp3(temp_file)

            # Aplicar modificaciones de velocidad
            if emotion_params["speed"] != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={
                        "frame_rate": int(audio.frame_rate * emotion_params["speed"])
                    }
                ).set_frame_rate(audio.frame_rate)

            # Ajustar volumen según emoción
            volume_adjustments = {
                "feliz": 2,
                "enojada": 3,
                "emocionada": 4,
                "triste": -2,
                "preocupada": -1
            }

            volume_change = volume_adjustments.get(emotion, 0)
            if volume_change != 0:
                audio = audio + volume_change

            # Exportar audio procesado
            audio.export(output_path, format="mp3", bitrate="128k")

            # Limpiar archivo temporal
            if os.path.exists(temp_file):
                os.remove(temp_file)

            return f"/static/audio/{filename}"

        except Exception as e:
            print(f"Error generando audio: {e}")
            traceback.print_exc()
            return None

    def generate_speech_async(self, text: str, emotion: str = "neutral"):
        """
        Versión asíncrona para generar voz en background
        """
        return self.text_to_speech(text, emotion)

    def delete_audio_file(self, audio_url: str) -> bool:
        """Elimina un archivo de audio específico"""
        try:
            if audio_url:
                # Convertir URL a ruta de archivo
                audio_path = audio_url.replace("/static/audio/", "")
                full_path = os.path.join(self.audio_dir, audio_path)

                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"Audio eliminado: {audio_path}")
                    return True
            return False
        except Exception as e:
            print(f"Error eliminando archivo de audio: {e}")
            return False

    def clean_old_audio_files(self, max_age_seconds: int = 300):
        """Limpia archivos de audio más antiguos que max_age_seconds (default: 5 minutos)"""
        try:
            import time
            current_time = time.time()
            audio_files = [
                f for f in os.listdir(self.audio_dir)
                if f.startswith("lily_") and f.endswith(".mp3")
            ]

            deleted_count = 0
            for file in audio_files:
                file_path = os.path.join(self.audio_dir, file)
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1

            if deleted_count > 0:
                print(f"Limpiados {deleted_count} archivos de audio antiguos")

        except Exception as e:
            print(f"Error limpiando archivos de audio: {e}")

    def analyze_reference_audio(self):
        """
        Analiza los archivos de audio de referencia para extraer características
        """
        try:
            lily_audio = os.path.join(self.reference_audio_dir, "LILY.wav")
            lily2_audio = os.path.join(self.reference_audio_dir, "LILY2.wav")

            if os.path.exists(lily_audio):
                audio = AudioSegment.from_wav(lily_audio)
                duration = len(audio) / 1000.0  # en segundos

                print(f"Análisis de LILY.wav:")
                print(f"  - Duración: {duration:.2f}s")
                print(f"  - Canales: {audio.channels}")
                print(f"  - Frame rate: {audio.frame_rate}Hz")
                print(f"  - Sample width: {audio.sample_width} bytes")

            if os.path.exists(lily2_audio):
                audio = AudioSegment.from_wav(lily2_audio)
                duration = len(audio) / 1000.0

                print(f"Análisis de LILY2.wav:")
                print(f"  - Duración: {duration:.2f}s")
                print(f"  - Canales: {audio.channels}")
                print(f"  - Frame rate: {audio.frame_rate}Hz")
                print(f"  - Sample width: {audio.sample_width} bytes")

        except Exception as e:
            print(f"Error analizando audio de referencia: {e}")


# Instancia global
tts_engine = TTSEngine()
