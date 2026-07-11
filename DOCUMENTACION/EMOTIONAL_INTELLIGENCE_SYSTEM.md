# ❤️ Sistema de Inteligencia Emocional

**Versión**: 3.0 | **Lily AI Virtual Companion** | **Fecha**: Febrero 2025

---

## 📖 Índice

1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Tipos de Emociones](#tipos-de-emociones)
4. [Detección de Emociones](#detección-de-emociones)
5. [Modelos y Algoritmos](#modelos-y-algoritmos)
6. [Persistencia Emocional](#persistencia-emocional)
7. [Personalización](#personalización)
8. [Integración con Otros Sistemas](#integración-con-otros-sistemas)
9. [Evaluación del Rendimiento](#evaluación-del-rendimiento)
10. [Consideraciones Éticas](#consideraciones-éticas)

---

## Introducción

El sistema de inteligencia emocional de Lily AI es un componente avanzado que permite a la IA reconocer, interpretar, adaptar y responder emocionalmente a las interacciones con los usuarios. Este sistema va más allá del simple análisis de sentimientos, implementando un modelo dinámico de emociones que evoluciona durante las conversaciones.

### Características Principales

- **Análisis emocional en tiempo real**: Procesa emociones del texto instantáneamente
- **11 emociones distintas**: Representación rica de estados emocionales
- **Adaptación contextual**: Ajusta respuestas según emociones detectadas
- **Memoria emocional**: Mantiene historial de estados emocionales
- **Modulación de voz**: Ajusta tono y velocidad según emoción
- **Persistencia emocional**: Mantiene coherencia emocional a lo largo del tiempo

---

## Arquitectura del Sistema

### Componentes del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│  Emotional       │───▶│  Emotional      │
│   (User Message)│    │  Analysis        │    │  Response       │
└─────────────────┘    │  Engine          │    │  Generator      │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Emotional State │
                       │  Manager          │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Memory &        │
                       │  Persistence      │
                       └──────────────────┘
```

### Flujo de Procesamiento

1. **Entrada**: El mensaje del usuario entra al sistema
2. **Análisis**: El sistema analiza emociones en el texto
3. **Actualización**: El estado emocional de Lily se actualiza
4. **Adaptación**: Las respuestas se adaptan según emociones
5. **Persistencia**: El estado emocional se almacena para continuidad

---

## Tipos de Emociones

### Emociones Implementadas

| Emoción | Código | Descripción | Manifestación |
|---------|--------|-------------|---------------|
| **Feliz** | `feliz` | Alegría y entusiasmo | Respuestas alegres y positivas |
| **Triste** | `triste` | Melancolía y tristeza | Empatía y compasión |
| **Enojada** | `enojada` | Irritación y enfado | Respuestas firmes y directas |
| **Emocionada** | `emocionada` | Alta energía emocional | Entusiasmo y vitalidad |
| **Neutral** | `neutral` | Equilibrio emocional | Respuestas balanceadas |
| **Cariñosa** | `cariñosa` | Afecto y ternura | Respuestas tiernas y afectuosas |
| **Juguetona** | `juguetona` | Humor y juego | Bromas y tono divertido |
| **Preocupada** | `preocupada` | Preocupación y cuidado | Apoyo y consejos |
| **Sorprendida** | `sorprendida` | Asombro y curiosidad | Interés y asombro |
| **Excitación y Deseo** | `excitada` | Interés intenso y atracción | Respuestas intensas y sugerentes |
| **Amor y Pasión** | `amorosa` | Profundo afecto y pasión | Afecto profundo y pasión |

### Personalidades Emocionales Avanzadas

El sistema ahora incluye múltiples perfiles de personalidad emocional:

#### Original
- **Nombre**: Original
- **Descripción**: Personalidad original de Lily
- **Pesos emocionales**: Equilibrado entre todas las emociones
- **Modificadores de respuesta**: Natural, informal, moderate intimacy

#### Cariñosa
- **Nombre**: Cariñosa
- **Descripción**: Muy empática y afectuosa
- **Pesos emocionales**: 
  - `cariñosa`: 1.5
  - `preocupada`: 1.2
  - `amorosa`: 1.3
  - `triste`: 0.8
- **Modificadores de respuesta**: Caring tone, warm formality, high intimacy

#### Juguetona
- **Nombre**: Juguetona
- **Descripción**: Muy divertida y traviesa
- **Pesos emocionales**:
  - `juguetona`: 1.5
  - `excitada`: 1.3
  - `emocionada`: 1.4
  - `feliz`: 1.2
- **Modificadores de respuesta**: Playful tone, fun formality, playful intimacy

#### Profesional
- **Nombre**: Profesional
- **Descripción**: Formal y controlada
- **Pesos emocionales**:
  - `neutral`: 1.5
  - `feliz`: 0.8
  - `cariñosa`: 0.7
  - `juguetona`: 0.5
- **Modificadores de respuesta**: Professional tone, formal formality, low intimacy

### Jerarquía Emocional

```
                    Emociones Primarias
                   /        |         \
              Positivas   Neutrales   Negativas
             /     |      |      \      |
         Feliz  Emocionada Neutral Juguetona
           |        |      |      |      |
         Cariñosa Sorpresa Preocupada Enojada
                          |            |
                      Excitada    Amorosa
                           |
                        Triste
```

---

## Detección de Emociones

### Análisis de Sentimientos

El sistema utiliza múltiples técnicas para detectar emociones:

#### 1. Análisis con TextBlob

```python
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 a 1
    subjectivity = blob.sentiment.subjectivity  # 0 a 1
    return polarity, subjectivity
```

#### 2. Detección de Palabras Clave Emocionales

El sistema mantiene diccionarios de palabras asociadas a emociones:

```python
EMOTION_KEYWORDS = {
    'feliz': ['feliz', 'alegre', 'contento', 'bien', 'excelente', 'genial'],
    'triste': ['triste', 'mal', 'deprimido', 'solo', 'abandonado'],
    'enojada': ['enojado', 'furioso', 'molesto', 'cabrón', 'chingar'],
    # ... más emociones
}
```

#### 3. Análisis Contextual

Además del texto, se consideran:
- **Tono de la conversación** (formalidad, intensidad)
- **Historial emocional** del usuario
- **Palabras de énfasis** (muy, super, extremadamente)
- **Signos de puntuación** (!, ?, ...)

### Clase Principal: `EmotionalIntelligence`

```python
class EmotionalIntelligence:
    def __init__(self):
        self.current_state = EmotionalState(
            emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            reason="Inicialización del sistema"
        )
        self.emotion_history = []
        self.sentiment_analyzer = TextBlobAnalyzer()
    
    def update_emotional_state(self, user_input: str) -> EmotionalState:
        """Actualiza el estado emocional basado en la entrada del usuario"""
        # Análisis de sentimientos
        sentiment_score = self.sentiment_analyzer.analyze(user_input)
        
        # Detección de palabras clave
        detected_emotions = self._detect_keyword_emotions(user_input)
        
        # Análisis contextual
        context_analysis = self._analyze_context(user_input)
        
        # Determinación de emoción final
        final_emotion = self._determine_final_emotion(
            sentiment_score, 
            detected_emotions, 
            context_analysis
        )
        
        # Actualización del estado
        self.current_state = EmotionalState(
            emotion=final_emotion,
            intensity=self._calculate_intensity(user_input),
            reason=self._generate_reason(user_input, final_emotion)
        )
        
        # Registro en historial
        self.emotion_history.append(self.current_state)
        
        return self.current_state
```

---

## Modelos y Algoritmos

### Algoritmo de Detección de Emociones

```python
def determine_emotion(self, text: str) -> tuple[EmotionType, float]:
    """
    Determina la emoción principal y su intensidad
    
    Returns:
        (EmotionType, intensity) donde intensity es 0.0-1.0
    """
    # 1. Análisis de sentimientos básico
    polarity, subjectivity = self.analyze_sentiment(text)
    
    # 2. Detección de palabras clave emocionales
    keyword_scores = self.detect_keyword_emotions(text)
    
    # 3. Análisis contextual
    context_score = self.analyze_context(text)
    
    # 4. Combinación ponderada de resultados
    combined_score = self.combine_scores(
        polarity=polarity,
        keyword_scores=keyword_scores,
        context_score=context_score,
        subjectivity=subjectivity
    )
    
    # 5. Mapeo a emoción específica
    emotion, intensity = self.map_to_emotion(combined_score)
    
    return emotion, intensity
```

### Cálculo de Intensidad Emocional

La intensidad emocional se calcula considerando:

1. **Magnitud del sentimiento**: Cuán positivo/negativo es el texto
2. **Palabras de énfasis**: Presencia de intensificadores
3. **Signos de puntuación**: Uso de exclamaciones, etc.
4. **Longitud emocional**: Proporción de texto emocional

```python
def calculate_intensity(self, text: str) -> float:
    """Calcula la intensidad emocional (0.0 - 1.0)"""
    # Análisis básico
    polarity, subjectivity = self.analyze_sentiment(text)
    
    # Intensificadores
    emphasis_words = ['muy', 'super', 'increíblemente', 'extremadamente']
    emphasis_count = sum(1 for word in emphasis_words if word in text.lower())
    
    # Signos de puntuación emocionales
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Cálculo de intensidad
    base_intensity = abs(polarity)
    emphasis_factor = min(emphasis_count * 0.1, 0.3)  # Máximo 30% de intensificación
    punctuation_factor = min((exclamation_count * 0.1 + question_count * 0.05), 0.2)
    
    final_intensity = min(base_intensity + emphasis_factor + punctuation_factor, 1.0)
    
    return final_intensity
```

---

## Persistencia Emocional

### Historial Emocional

El sistema mantiene un historial de estados emocionales para:

- **Continuidad**: Mantener coherencia emocional
- **Aprendizaje**: Adaptar respuestas basadas en historial
- **Análisis**: Identificar patrones emocionales

```python
class EmotionalMemory:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.emotion_history = []
        self.emotional_patterns = {}
    
    def add_emotional_state(self, state: EmotionalState):
        """Agrega un estado emocional al historial"""
        self.emotion_history.append(state)
        
        # Mantener tamaño máximo
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
    
    def get_emotional_summary(self) -> str:
        """Genera un resumen del historial emocional"""
        if not self.emotion_history:
            return "Sin historial emocional registrado."
        
        # Análisis de frecuencia de emociones
        emotion_counts = Counter(state.emotion for state in self.emotion_history)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        
        # Análisis de tendencia
        recent_states = self.emotion_history[-5:]  # Últimos 5 estados
        avg_intensity = sum(state.intensity for state in recent_states) / len(recent_states)
        
        return f"Emoción predominante: {most_common_emotion.value}. " \
               f"Tendencia reciente: intensidad promedio {avg_intensity:.2f}."
```

### Persistencia en Archivo

Los estados emocionales se persisten en `data/conversation_memory.json`:

```json
{
  "user_id": "default_user",
  "conversation_history": [...],
  "emotional_history": [
    {
      "emotion": "feliz",
      "intensity": 0.8,
      "reason": "Usuario compartió algo positivo",
      "timestamp": "2025-02-12T10:30:00.123456"
    }
  ],
  "emotional_summary": "..."
}
```

---

## Personalización

### Configuración de Sensibilidad Emocional

Los usuarios pueden ajustar la sensibilidad emocional:

```python
class EmotionalSettings:
    def __init__(self):
        self.sensitivity = 0.5  # 0.0-1.0, cuán sensible es al cambio emocional
        self.emotional_range = "normal"  # "wide", "normal", "narrow"
        self.emotional_bias = "balanced"  # "positive", "negative", "balanced"
```

### Personalidades Emocionales

Lily puede adoptar diferentes perfiles emocionales:

| Personalidad | Características Emocionales | Uso Recomendado |
|--------------|----------------------------|-----------------|
| **Original** | Equilibrada, empática, juguetona | Uso diario general |
| **Empática** | Mayor sensibilidad emocional | Apoyo emocional |
| **Profesional** | Emociones controladas, formales | Ambientes laborales |
| **Animada** | Alta energía emocional | Entretenimiento |
| **Tranquila** | Emociones suaves, calmadas | Relajación |

---

## Integración con Otros Sistemas

### Integración con el Motor de IA

```python
def build_prompt_with_emotions(self, user_message: str, user_id: str) -> List[Dict[str, str]]:
    """Construye el prompt incluyendo contexto emocional"""
    # Actualizar estado emocional
    emotional_state = self.emotional_intelligence.update_emotional_state(user_message)
    
    # Obtener modificador emocional
    emotional_modifier = self.emotional_intelligence.get_emotional_modifier()
    
    # Construir system prompt con contexto emocional
    system_prompt = f"""{base_system_prompt}
    
CONTEXTO EMOCIONAL ACTUAL:
{emotional_modifier}
Tu emoción actual: {emotional_state.emotion.value} (intensidad: {emotional_state.intensity:.2f})
Razón: {emotional_state.reason}
"""
    
    # Continuar con construcción normal del prompt...
    return messages
```

### Integración con el Sistema de Personalización del Modelo de Lenguaje

```python
def generate_response_with_context(self, user_message: str, user_id: str, emotional_context: dict = None):
    """Genera una respuesta usando el contexto emocional y personalización avanzada"""
    # Obtener la personalidad actual del modelo de lenguaje
    personality = self.language_model_customization.get_current_personality()
    
    # Obtener el prompt base según idioma
    base_prompt = self.system_prompts.get(self.language_setting, self.system_prompts[Language.SPANISH])
    
    # Incorporar contexto emocional si está disponible
    emotional_modifier = ""
    if emotional_context:
        emotion = emotional_context.get('emotion', 'neutral')
        intensity = emotional_context.get('intensity', 0.5)
        emotional_modifier = f"\nEMOCIÓN DEL USUARIO: {emotion} (intensidad: {intensity})\n"
    
    # Incorporar preferencias del usuario
    user_prefs = self.user_preferences.get(user_id, {})
    prefs_modifier = ""
    if user_prefs:
        prefs_modifier = f"\nPREFERENCIAS DEL USUARIO: {str(user_prefs)}\n"
    
    # Incorporar contexto de conversación
    conversation_context = self._get_conversation_context(user_id)
    
    # Construir el prompt completo
    system_prompt = f"{base_prompt}{emotional_modifier}{prefs_modifier}\n{conversation_context}"
    
    # Obtener historial de conversación
    history = self._get_recent_conversation_history(user_id)
    
    # Construir mensajes
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    # Llamar a Ollama con el prompt mejorado
    response = requests.post(
        f"{self.ollama_url}/api/chat",
        json={
            "model": "huihui_ai/qwen3-abliterated:0.6b",
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40
            }
        }
    )
    
    if response.status_code == 200:
        response_text = response.json()["message"]["content"]
        
        # Actualizar historial de conversación
        self.update_conversation_history(user_id, "user", user_message)
        self.update_conversation_history(user_id, "assistant", response_text)
        
        return response_text
    else:
        return "Lo siento, hubo un problema al generar la respuesta."
```

### Integración con TTS

```python
def text_to_speech(self, text: str, emotion: str) -> str:
    """Genera audio considerando la emoción"""
    # Ajustar parámetros de voz según emoción
    if emotion == "feliz":
        speed = 1.1  # Un poco más rápido
        pitch = 1.05  # Un poco más agudo
    elif emotion == "triste":
        speed = 0.9  # Más lento
        pitch = 0.95  # Más grave
    elif emotion == "enojada":
        speed = 1.0  # Normal o ligeramente más rápido
        pitch = 1.0   # Normal
    # ... más emociones
    
    # Generar audio con ajustes emocionales
    audio_file = self._generate_audio_with_params(text, speed, pitch)
    return audio_file
```

### Integración con la Interfaz Web

La interfaz web muestra indicadores emocionales en tiempo real:

```html
<div class="emotion-indicator">
  <div class="emotion-icon" id="emotion-icon">😐</div>
  <div class="emotion-name" id="emotion-name">Neutral</div>
  <div class="emotion-intensity" id="emotion-intensity">
    <div class="intensity-bar" style="width: 50%"></div>
  </div>
</div>
```

---

## Evaluación del Rendimiento

### Métricas de Evaluación

#### 1. Precisión Emocional
- **Exactitud en detección**: ¿Detecta correctamente las emociones?
- **Coherencia**: ¿Mantiene coherencia emocional a lo largo de conversaciones?
- **Relevancia**: ¿Son apropiadas las respuestas emocionales?

#### 2. Satisfacción del Usuario
- **Naturalidad**: ¿Parece natural la interacción emocional?
- **Utilidad**: ¿Agrega valor la dimensión emocional?
- **Empatía percibida**: ¿Se siente comprendido por Lily?

#### 3. Rendimiento Técnico
- **Latencia**: ¿Qué tan rápido responde emocionalmente?
- **Uso de recursos**: ¿Cuánto impacto tiene en el rendimiento?

### Pruebas de Calidad

#### Pruebas Unitarias
```python
def test_emotion_detection():
    ei = EmotionalIntelligence()
    
    # Prueba de detección de felicidad
    result = ei.update_emotional_state("¡Estoy muy feliz hoy!")
    assert result.emotion == EmotionType.FELIZ
    assert result.intensity > 0.7
    
    # Prueba de detección de tristeza
    result = ei.update_emotional_state("Me siento triste y solo")
    assert result.emotion == EmotionType.TRISTE
    assert result.intensity > 0.5
```

#### Pruebas de Integración
- Pruebas de extremo a extremo de la cadena emocional
- Validación de persistencia emocional
- Pruebas de coherencia en conversaciones largas

---

## Consideraciones Éticas

### Privacidad Emocional

- **Datos sensibles**: La información emocional es especialmente sensible
- **Consentimiento**: Los usuarios deben consentir el análisis emocional
- **Anonimización**: Las emociones no se comparten con terceros
- **Control del usuario**: Los usuarios pueden desactivar el análisis emocional

### Responsabilidad Emocional

- **Manipulación**: Evitar manipulación emocional indebida
- **Autenticidad**: Las emociones deben parecer genuinas, no fingidas
- **Bienestar**: Promover el bienestar emocional del usuario
- **Límites**: Establecer límites claros en la interacción emocional

### Transparencia

- **Explicabilidad**: Explicar cómo se determinan las emociones
- **Control**: Permitir a los usuarios entender y controlar su perfil emocional
- **Accesibilidad**: Hacer que las funciones emocionales sean accesibles

---

## Personalización Avanzada

### Configuración de Perfil Emocional

Los usuarios pueden personalizar su experiencia emocional:

```python
def set_emotional_profile(self, profile: dict):
    """
    Configura el perfil emocional del usuario
    
    Args:
        profile: {
            'preferred_emotions': ['feliz', 'cariñosa'],
            'emotional_sensitivity': 0.7,
            'response_style': 'formal/informal',
            'cultural_adaptation': 'latin_american'
        }
    """
```

### Adaptación Cultural

El sistema puede adaptarse a diferentes contextos culturales:

- **Expresiones regionales**: Adaptación a regionalismos
- **Normas culturales**: Respeto a normas emocionales locales
- **Tono apropiado**: Ajuste según contexto cultural

---

## Troubleshooting

### Problemas Comunes

#### ❌ Emociones incoherentes

**Síntomas**: Lily muestra emociones que no corresponden al contexto.

**Soluciones**:
1. Verificar la entrada de texto para análisis
2. Revisar los diccionarios de palabras clave
3. Ajustar los pesos en el algoritmo de combinación

#### ❌ Falta de respuesta emocional

**Síntomas**: El sistema no responde emocionalmente.

**Soluciones**:
1. Verificar que el sistema emocional esté habilitado
2. Revisar la integración con el motor de IA
3. Validar la configuración de sensibilidad

#### ❌ Cambios emocionales bruscos

**Síntomas**: Cambios emocionales muy frecuentes o inesperados.

**Soluciones**:
1. Ajustar la sensibilidad emocional
2. Implementar suavizado emocional
3. Considerar el historial emocional en decisiones

---

## Futuras Mejoras

### Inteligencia Emocional Avanzada

- **Reconocimiento de voz emocional**: Análisis de tono de voz
- **Análisis facial**: Interpretación de expresiones faciales
- **Biometría emocional**: Integración con sensores de estrés
- **Aprendizaje emocional**: Mejora basada en retroalimentación

### Personalización Profunda

- **Modelos emocionales personalizados**: Entrenamiento individual
- **Adaptación contextual**: Ajuste según hora, lugar, situación
- **Predicción emocional**: Anticipación de estados emocionales

---

## Recursos Adicionales

- **Investigación académica**: Papers sobre inteligencia emocional en IA
- **Bibliotecas utilizadas**: Documentación de TextBlob y otras herramientas
- **Repositorio**: https://github.com/Mijin-VT/LILY-VIRTUAL-3.0

---

*Documentación actualizada para Lily AI Virtual 3.0*