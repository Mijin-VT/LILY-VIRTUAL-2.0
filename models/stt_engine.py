import os
import time
from faster_whisper import WhisperModel

class STTEngine:
    """Motor de reconocimiento de voz (Speech-to-Text) usando Faster Whisper"""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        print(f"Cargando modelo Whisper ({model_size}) en {device}...")
        # faster-whisper usa ctranslate2 y no necesita torch.
        # Si device="cuda" falla (sin GPU o sin CUDA), caemos a CPU automaticamente.
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"Modelo Whisper cargado correctamente en {device}.")
        except Exception as e:
            if device != "cpu":
                print(f"No se pudo cargar Whisper en {device} ({e}), intentando CPU...")
                try:
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print("Modelo Whisper cargado correctamente en CPU.")
                    return
                except Exception as e2:
                    e = e2
            print(f"Error cargando Whisper: {e}")
            self.model = None

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe un archivo de audio a texto"""
        if not self.model:
            return "Error: Modelo Whisper no cargado."
            
        if not os.path.exists(audio_file_path):
            return "Error: Archivo de audio no encontrado."
            
        try:
            segments, info = self.model.transcribe(audio_file_path, beam_size=5, language="es")
            
            # Recopilar todos los segmentos
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"Error en transcripción: {e}")
            return f"Error transcribiendo: {str(e)}"
