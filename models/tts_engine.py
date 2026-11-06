import os
import hashlib
import re
import traceback
import unicodedata
from gtts import gTTS
from pydub import AudioSegment

# PocketTTS (motor de respaldo) -----------------------------------------------
try:
    from pocket_tts import TTSModel as _PocketTTSModel
    import scipy.io.wavfile as _wavfile
    import numpy as _np_pocket
    POCKET_TTS_AVAILABLE = True
except ImportError:
    POCKET_TTS_AVAILABLE = False


class TTSEngine:
    """Motor de texto a voz.

    Prioridad:
        1. gTTS         - voz femenina, espanol, acento Mexico (online)
        2. PocketTTS    - pocket-tts Python, modelo safetensors local
        3. AudioCpp     - audiocpp_cli.exe + pocket-tts-spanish-bf16.gguf (offline, CPU)
    """

    # Rutas relativas al directorio del proyecto
    _AUDIOCPP_CLI_RELPATH = os.path.join("tools", "audiocpp", "audiocpp_cli.exe")
    _AUDIOCPP_MODEL_RELPATH = os.path.join(
        "models", "pocket_tts", "PocketTTS-GGUF", "spanish", "pocket-tts-spanish-bf16.gguf"
    )

    # ------------------------------------------------------------------
    # Limpieza de texto para TTS (emojis, emoticonos y formato)
    # ------------------------------------------------------------------
    # Emojis, pictogramas, simbolos decorativos y selectores de variacion.
    # Se eliminan antes de sintetizar para que la voz no lea "cara sonriente",
    # "llama", "corazon", etc.
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

        # Parametros de voz / emocion
        self.voice_params = {
            "speed": 1.1,
            "emotion_modifiers": {
                "feliz":       {"speed": 1.15, "volume": 2},
                "triste":      {"speed": 0.9,  "volume": -2},
                "enojada":     {"speed": 1.2,  "volume": 3},
                "emocionada":  {"speed": 1.25, "volume": 4},
                "neutral":     {"speed": 1.1,  "volume": 0},
                "carinosa":    {"speed": 1.0,  "volume": 0},
                "juguetona":   {"speed": 1.2,  "volume": 1},
                "preocupada":  {"speed": 0.95, "volume": -1},
                "sorprendida": {"speed": 1.15, "volume": 2},
            },
        }

        # Inicializar PocketTTS en segundo plano (carga pesada, solo una vez)
        self._pocket_model = None
        self._pocket_voice_state = None
        if POCKET_TTS_AVAILABLE:
            self._init_pocket_tts()
        else:
            print("[TTS] pocket-tts no disponible. Solo se usara gTTS.")

        # Detectar AudioCpp CLI + modelo GGUF
        self._audiocpp_cli = None
        self._audiocpp_model = None
        self._audiocpp_backend = None
        self._init_audiocpp()

    # --------------------------------------------------------------------------
    # PocketTTS - inicializacion
    # --------------------------------------------------------------------------
    def _init_pocket_tts(self):
        """Carga el modelo PocketTTS espanol de forma silenciosa."""
        try:
            print("[TTS] Cargando modelo PocketTTS (spanish_24l)...")
            self._pocket_model = _PocketTTSModel.load_model(language="spanish_24l")
            # Voz femenina 'lola' (predefinida en pocket-tts)
            self._pocket_voice_state = self._pocket_model.get_state_for_audio_prompt("lola")
            print("[TTS] Motor PocketTTS cargado correctamente (voz: lola).")
        except Exception as e:
            print(f"[TTS] No se pudo cargar PocketTTS: {e}. Solo se usara gTTS.")
            self._pocket_model = None
            self._pocket_voice_state = None

    # --------------------------------------------------------------------------
    # AudioCpp GGUF - inicializacion
    # --------------------------------------------------------------------------
    def _init_audiocpp(self):
        """Detecta audiocpp_cli.exe y el modelo GGUF. Prueba CUDA primero, luego CPU."""
        # Resolver rutas absolutas desde el directorio de trabajo actual
        base_dir = os.path.abspath(os.getcwd())
        cli_path = os.path.join(base_dir, self._AUDIOCPP_CLI_RELPATH)
        model_path = os.path.join(base_dir, self._AUDIOCPP_MODEL_RELPATH)

        # Tambien intentar relativo al directorio de este archivo
        if not os.path.exists(cli_path):
            file_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(file_dir)  # sube un nivel desde models/
            cli_path = os.path.join(project_dir, self._AUDIOCPP_CLI_RELPATH)
            model_path = os.path.join(project_dir, self._AUDIOCPP_MODEL_RELPATH)

        if not os.path.exists(cli_path):
            print("[TTS-GGUF] audiocpp_cli.exe no encontrado. Motor GGUF no disponible.")
            return
        if not os.path.exists(model_path):
            print(f"[TTS-GGUF] Modelo GGUF no encontrado: {model_path}")
            return

        self._audiocpp_cli = cli_path
        self._audiocpp_model = model_path

        # Probar backend: CUDA primero, CPU como fallback
        import subprocess, tempfile
        for backend in ("cuda", "cpu"):
            try:
                tmp = tempfile.mktemp(suffix=".wav")
                result = subprocess.run(
                    [cli_path, "--task", "tts", "--family", "pocket_tts",
                     "--model", model_path, "--backend", backend,
                     "--text", "test", "--voice-id", "lola", "--out", tmp],
                    capture_output=True, text=True, timeout=30
                )
                if os.path.exists(tmp):
                    os.remove(tmp)
                if result.returncode == 0:
                    self._audiocpp_backend = backend
                    print(f"[TTS-GGUF] AudioCpp listo. Backend: {backend.upper()} | Modelo: pocket-tts-spanish-bf16.gguf")
                    break
            except Exception:
                continue

        if not self._audiocpp_backend:
            print("[TTS-GGUF] AudioCpp no pudo inicializar en ningun backend.")
            self._audiocpp_cli = None

    # --------------------------------------------------------------------------
    # Nombre de archivo
    # --------------------------------------------------------------------------
    def generate_audio_filename(self, text: str) -> str:
        """Genera un nombre de archivo MP3 unico basado en texto + timestamp."""
        import time
        timestamp = str(int(time.time() * 1000))
        text_hash = hashlib.md5((text + timestamp).encode()).hexdigest()[:8]
        return f"lily_{timestamp}_{text_hash}.mp3"

    # --------------------------------------------------------------------------
    # Motor principal: gTTS
    # --------------------------------------------------------------------------
    def _generate_gtts(self, text: str, emotion: str, output_path: str) -> bool:
        """Genera audio con gTTS (espanol Mexico). Devuelve True si tuvo exito."""
        try:
            params = self.voice_params["emotion_modifiers"].get(
                emotion, self.voice_params["emotion_modifiers"]["neutral"]
            )
            temp_file = output_path + ".tmp.mp3"

            # gTTS: espanol, dominio Mexico (.com.mx) -> acento y voz femenina
            tts = gTTS(text=text, lang="es", tld="com.mx", slow=False)
            tts.save(temp_file)

            audio = AudioSegment.from_mp3(temp_file)

            # Ajuste de velocidad
            speed = params.get("speed", 1.0)
            if speed != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": int(audio.frame_rate * speed)},
                ).set_frame_rate(audio.frame_rate)

            # Ajuste de volumen
            vol = params.get("volume", 0)
            if vol != 0:
                audio = audio + vol

            audio.export(output_path, format="mp3", bitrate="128k")

            # Limpiar temporal
            try:
                os.remove(temp_file)
            except OSError:
                pass

            return True

        except Exception as e:
            print(f"[TTS] gTTS fallo: {e}. Intentando PocketTTS de respaldo...")
            return False

    # --------------------------------------------------------------------------
    # Motor de respaldo 1: PocketTTS Python
    # --------------------------------------------------------------------------
    def _generate_pocket_tts(self, text: str, emotion: str, output_path: str) -> bool:
        """Genera audio con PocketTTS. Devuelve True si tuvo exito."""
        if self._pocket_model is None or self._pocket_voice_state is None:
            return False
        try:
            import numpy as np
            params = self.voice_params["emotion_modifiers"].get(
                emotion, self.voice_params["emotion_modifiers"]["neutral"]
            )

            # Generar audio (tensor float)
            audio_tensor = self._pocket_model.generate_audio(
                self._pocket_voice_state,
                text,
                max_tokens=300,
            )

            # Convertir a numpy int16 y guardar como WAV temporal
            sample_rate = self._pocket_model.sample_rate
            audio_np = audio_tensor.cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]  # primer canal
            audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

            wav_tmp = output_path + ".tmp.wav"
            _wavfile.write(wav_tmp, sample_rate, audio_int16)

            # Convertir WAV -> MP3 con pydub + aplicar emocion
            audio = AudioSegment.from_wav(wav_tmp)

            speed = params.get("speed", 1.0)
            if speed != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": int(audio.frame_rate * speed)},
                ).set_frame_rate(audio.frame_rate)

            vol = params.get("volume", 0)
            if vol != 0:
                audio = audio + vol

            audio.export(output_path, format="mp3", bitrate="128k")

            try:
                os.remove(wav_tmp)
            except OSError:
                pass

            return True

        except Exception as e:
            print(f"[TTS] PocketTTS tambien fallo: {e}")
            traceback.print_exc()
            return False

    # --------------------------------------------------------------------------
    # Motor de respaldo 2: AudioCpp GGUF
    # --------------------------------------------------------------------------
    def _generate_audiocpp(self, text: str, emotion: str, output_path: str) -> bool:
        """Genera audio con audiocpp_cli.exe + pocket-tts-spanish-bf16.gguf.
        Devuelve True si tuvo exito."""
        if not self._audiocpp_cli or not self._audiocpp_backend:
            return False
        import subprocess, tempfile

        wav_tmp = output_path + ".gguf.tmp.wav"
        try:
            params = self.voice_params["emotion_modifiers"].get(
                emotion, self.voice_params["emotion_modifiers"]["neutral"]
            )
            speed = params.get("speed", 1.0)

            cmd = [
                self._audiocpp_cli,
                "--task",    "tts",
                "--family",  "pocket_tts",
                "--model",   self._audiocpp_model,
                "--backend", self._audiocpp_backend,
                "--text",    text,
                "--voice-id","lola",
                "--out",     wav_tmp,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,   # max 2 minutos para textos largos
            )

            if result.returncode != 0:
                print(f"[TTS-GGUF] audiocpp_cli fallo (codigo {result.returncode}): {result.stderr[:200]}")
                return False

            if not os.path.exists(wav_tmp):
                print("[TTS-GGUF] Archivo WAV no generado.")
                return False

            # WAV -> MP3 con ajustes de emocion via pydub
            audio = AudioSegment.from_wav(wav_tmp)

            if speed != 1.0:
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": int(audio.frame_rate * speed)},
                ).set_frame_rate(audio.frame_rate)

            vol = params.get("volume", 0)
            if vol != 0:
                audio = audio + vol

            audio.export(output_path, format="mp3", bitrate="128k")
            print(f"[TTS-GGUF] Audio generado con AudioCpp ({self._audiocpp_backend.upper()}).")
            return True

        except subprocess.TimeoutExpired:
            print("[TTS-GGUF] Timeout al generar audio con AudioCpp.")
            return False
        except Exception as e:
            print(f"[TTS-GGUF] Error inesperado en AudioCpp: {e}")
            traceback.print_exc()
            return False
        finally:
            try:
                if os.path.exists(wav_tmp):
                    os.remove(wav_tmp)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # Metodo publico principal
    # --------------------------------------------------------------------------
    def text_to_speech(self, text: str, emotion: str = "neutral", save: bool = True):
        """Convierte texto a voz con la cadena de prioridad de motores.

        Prioridad:
            1. gTTS (Google, acento Mexico) — requiere internet
            2. PocketTTS Python (safetensors local)
            3. AudioCpp GGUF (audiocpp_cli.exe, completamente offline)

        Args:
            text:    Texto a convertir.
            emotion: Emocion para modular la voz ('neutral', 'feliz', 'triste', ...).
            save:    Parametro reservado para compatibilidad.

        Returns:
            Ruta URL del archivo de audio generado ('/static/audio/...'),
            o None si todos los motores fallaron.
        """
        try:
            # Quitar emojis, emoticonos y marcas de formato antes de hablar.
            text = self.clean_text_for_tts(text)
            if not text:
                print("[TTS] Texto sin contenido hablable (solo emotes); se omite el audio.")
                return None

            filename = self.generate_audio_filename(f"{text}_{emotion}")
            output_path = os.path.join(self.audio_dir, filename)

            # Evitar regenerar si ya existe
            if os.path.exists(output_path):
                return f"/static/audio/{filename}"

            # 1) Motor principal: gTTS (espanol Mexico)
            if self._generate_gtts(text, emotion, output_path):
                return f"/static/audio/{filename}"

            # 2) Respaldo 1: PocketTTS Python
            if self._generate_pocket_tts(text, emotion, output_path):
                return f"/static/audio/{filename}"

            # 3) Respaldo 2: AudioCpp GGUF (offline total)
            if self._generate_audiocpp(text, emotion, output_path):
                return f"/static/audio/{filename}"

            print("[TTS] Todos los motores TTS fallaron.")
            return None

        except Exception as e:
            print(f"[TTS] Error inesperado en text_to_speech: {e}")
            traceback.print_exc()
            return None

    # --------------------------------------------------------------------------
    # Metodos auxiliares / compatibilidad
    # --------------------------------------------------------------------------
    def generate_speech_async(self, text: str, emotion: str = "neutral"):
        """Alias sincrono para compatibilidad con el codigo existente."""
        return self.text_to_speech(text, emotion)

    def delete_audio_file(self, audio_url: str) -> bool:
        """Elimina un archivo de audio especifico dado su URL relativa."""
        try:
            if audio_url:
                audio_path = audio_url.replace("/static/audio/", "")
                full_path = os.path.join(self.audio_dir, audio_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"[TTS] Audio eliminado: {audio_path}")
                    return True
            return False
        except Exception as e:
            print(f"[TTS] Error eliminando archivo de audio: {e}")
            return False

    def clean_old_audio_files(self, max_age_seconds: int = 300):
        """Limpia archivos MP3 generados mas antiguos que max_age_seconds."""
        try:
            import time
            current_time = time.time()
            deleted = 0
            for fname in os.listdir(self.audio_dir):
                if fname.startswith("lily_") and fname.endswith(".mp3"):
                    fpath = os.path.join(self.audio_dir, fname)
                    if current_time - os.path.getmtime(fpath) > max_age_seconds:
                        os.remove(fpath)
                        deleted += 1
            if deleted:
                print(f"[TTS] Limpiados {deleted} archivos de audio antiguos.")
        except Exception as e:
            print(f"[TTS] Error limpiando archivos de audio: {e}")


# Instancia global
tts_engine = TTSEngine()
