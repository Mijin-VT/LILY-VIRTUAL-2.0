// Estado de la aplicación
const state = {
    userId: 'default_user',
    isConnected: false,
    currentEmotion: 'neutral',
    isTyping: false,
    isVoiceModeActive: false, // Modo de conversación por voz continua
    currentAudio: null, // Referencia al audio actualmente en reproducción
    theme: 'light' // Tema actual ('light' o 'dark')
};

// Elementos DOM
const elements = {
    chatContainer: document.getElementById('chatContainer'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    micButton: document.getElementById('micButton'),
    clearButton: document.getElementById('clearButton'),
    memoryButton: document.getElementById('memoryButton'),
    settingsButton: document.getElementById('settingsButton'),
    themeToggle: document.getElementById('themeToggle'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    emotionText: document.getElementById('emotionText'),
    emotionIndicator: document.getElementById('emotionIndicator'),
    charCount: document.getElementById('charCount'),
    memoryModal: document.getElementById('memoryModal'),
    memoryContent: document.getElementById('memoryContent'),
    commandsModal: document.getElementById('commandsModal'),
    closeCommands: document.querySelector('.close-commands'),
    commandsButton: document.getElementById('commandsButton'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettings: document.querySelector('.close-settings'),
    selectFileBtn: document.getElementById('selectFileBtn'),
    uploadFileBtn: document.getElementById('uploadFileBtn'),
    ragFileInput: document.getElementById('ragFileInput'),
    selectedFileName: document.getElementById('selectedFileName'),
    uploadStatus: document.getElementById('uploadStatus'),
    ragDocumentsList: document.getElementById('ragDocumentsList'),
    reindexBtn: document.getElementById('reindexBtn'),
    ragStats: document.getElementById('ragStats'),
    gmailUserInput: document.getElementById('gmailUserInput'),
    gmailPasswordInput: document.getElementById('gmailPasswordInput'),
    saveGmailBtn: document.getElementById('saveGmailBtn'),
    gmailConfigStatus: document.getElementById('gmailConfigStatus'),
    welcomeUserHeader: document.getElementById('welcomeUserHeader'),
    userNameInput: document.getElementById('userNameInput'),
    saveUserNameBtn: document.getElementById('saveUserNameBtn'),
    userNameStatus: document.getElementById('userNameStatus'),
    avatar: document.getElementById('avatar'),
    mouth: document.getElementById('mouth')
};

// Reconocimiento de voz
let recognition = null;
let wakeWordRecognition = null;
let isRecording = false;
let isListeningForWakeWord = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    // Reconocimiento principal para mensajes
    // recognition = new SpeechRecognition(); -- REEMPLAZADO POR FASTER WHISPER LOCAL
    // recognition.lang = 'es-ES';
    // recognition.continuous = false;
    // recognition.interimResults = false;

    // recognition.onresult = (event) => { ... }

    let mediaRecorder = null;
    let audioChunks = [];

    // Event listeners para SpeechRecognition principal eliminados
    // Ahora se usa MediaRecorder + Backend

    // Configuración para wake word (aún usa Web Speech API si está disponible)

    // Reconocimiento continuo para palabra clave "Lily"
    wakeWordRecognition = new SpeechRecognition();
    wakeWordRecognition.lang = 'es-ES';
    wakeWordRecognition.continuous = true;
    wakeWordRecognition.interimResults = true;

    wakeWordRecognition.onresult = (event) => {
        const last = event.results.length - 1;
        const transcript = event.results[last][0].transcript.toLowerCase();

        // Detectar "lily" en el texto
        if (transcript.includes('lily') || transcript.includes('lili')) {
            console.log('Palabra clave detectada: Lily');
            if (!isRecording) {
                stopWakeWordListening();
                toggleRecording();
            }
        }
    };

    wakeWordRecognition.onend = () => {
        // Reiniciar automáticamente si debe seguir escuchando
        if (isListeningForWakeWord && !isRecording) {
            setTimeout(() => {
                try {
                    wakeWordRecognition.start();
                } catch (e) {
                    console.log('Wake word recognition ya está activo');
                }
            }, 100);
        }
    };

    wakeWordRecognition.onerror = (event) => {
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
            console.error('Error en wake word recognition:', event.error);
        }
    };
}

// Mapeo de emociones a colores y expresiones
const emotionConfig = {
    feliz: { color: '#ffd700', mouth: 'happy', emoji: '😊' },
    triste: { color: '#4a90e2', mouth: 'sad', emoji: '😢' },
    enojada: { color: '#e74c3c', mouth: 'angry', emoji: '😠' },
    emocionada: { color: '#ff6b9d', mouth: 'happy', emoji: '🤩' },
    neutral: { color: '#95a5a6', mouth: '', emoji: '😐' },
    cariñosa: { color: '#ff69b4', mouth: 'happy', emoji: '🥰' },
    juguetona: { color: '#9b59b6', mouth: 'happy', emoji: '😜' },
    preocupada: { color: '#f39c12', mouth: 'sad', emoji: '😟' },
    sorprendida: { color: '#1abc9c', mouth: 'surprised', emoji: '😲' }
};

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    // Cargar e inicializar tema guardado
    const savedTheme = localStorage.getItem('lily-theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        if (elements.themeToggle) elements.themeToggle.textContent = '☀️';
        state.theme = 'dark';
    } else {
        state.theme = 'light';
    }

    checkHealth();
    setupEventListeners();
    autoResizeTextarea();
    loadUserPreferences(); // Cargar nombre de usuario al inicio

    // Verificar salud cada 30 segundos
    setInterval(checkHealth, 30000);

    // Actualizar emoción cada 5 segundos
    setInterval(updateEmotion, 5000);

    // Iniciar escucha de palabra clave después de 2 segundos
    setTimeout(() => {
        if (wakeWordRecognition) {
            startWakeWordListening();
        }
    }, 2000);
});

// Configurar event listeners
function setupEventListeners() {
    elements.sendButton.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    elements.messageInput.addEventListener('input', () => {
        const length = elements.messageInput.value.length;
        elements.charCount.textContent = length;
        autoResizeTextarea();
    });

    elements.clearButton.addEventListener('click', clearChat);
    elements.memoryButton.addEventListener('click', showMemory);
    
    if (elements.commandsButton) {
        elements.commandsButton.addEventListener('click', showCommands);
    }

    if (elements.settingsButton) {
        elements.settingsButton.addEventListener('click', showSettings);
    }
    
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', toggleTheme);
    }

    // Botón de micrófono
    elements.micButton.addEventListener('click', toggleRecording);

    // Modal
    const closeBtn = document.querySelector('.close');
    closeBtn.addEventListener('click', () => {
        elements.memoryModal.style.display = 'none';
    });
    
    if (elements.closeSettings) {
        elements.closeSettings.addEventListener('click', () => {
            elements.settingsModal.style.display = 'none';
        });
    }

    if (elements.closeCommands) {
        elements.closeCommands.addEventListener('click', () => {
            elements.commandsModal.style.display = 'none';
        });
    }

    if (elements.selectFileBtn) {
        elements.selectFileBtn.addEventListener('click', () => {
            elements.ragFileInput.click();
        });
    }

    if (elements.ragFileInput) {
        elements.ragFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                elements.selectedFileName.textContent = file.name;
                elements.uploadFileBtn.disabled = false;
            } else {
                elements.selectedFileName.textContent = "Ningún archivo seleccionado";
                elements.uploadFileBtn.disabled = true;
            }
        });
    }

    if (elements.uploadFileBtn) {
        elements.uploadFileBtn.addEventListener('click', uploadRagFile);
    }

    if (elements.reindexBtn) {
        elements.reindexBtn.addEventListener('click', reindexRagKnowledge);
    }

    if (elements.saveGmailBtn) {
        elements.saveGmailBtn.addEventListener('click', saveGmailPreferences);
    }

    if (elements.saveUserNameBtn) {
        elements.saveUserNameBtn.addEventListener('click', saveUserName);
    }

    window.addEventListener('click', (e) => {
        if (e.target === elements.memoryModal) {
            elements.memoryModal.style.display = 'none';
        }
        if (e.target === elements.settingsModal) {
            elements.settingsModal.style.display = 'none';
        }
        if (e.target === elements.commandsModal) {
            elements.commandsModal.style.display = 'none';
        }
    });
}

// Auto-resize textarea
function autoResizeTextarea() {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
}

// Verificar salud del sistema
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        state.isConnected = data.ollama_connected;

        if (state.isConnected) {
            elements.statusDot.classList.remove('disconnected');
            elements.statusText.textContent = 'Conectada ✨';
        } else {
            elements.statusDot.classList.add('disconnected');
            elements.statusText.textContent = 'Desconectada (Ollama offline)';
        }
    } catch (error) {
        console.error('Error verificando salud:', error);
        state.isConnected = false;
        elements.statusDot.classList.add('disconnected');
        elements.statusText.textContent = 'Error de conexión';
    }
}

// Actualizar emoción
async function updateEmotion() {
    try {
        const response = await fetch('/api/emotion');
        const data = await response.json();

        updateEmotionDisplay(data.emotion);
    } catch (error) {
        console.error('Error obteniendo emoción:', error);
    }
}

// Actualizar display de emoción
function updateEmotionDisplay(emotion) {
    state.currentEmotion = emotion;
    const config = emotionConfig[emotion] || emotionConfig.neutral;

    elements.emotionText.textContent = `${config.emoji} ${emotion}`;
    elements.emotionIndicator.style.background = `linear-gradient(135deg, ${config.color}, ${adjustColor(config.color, -20)})`;

    // Actualizar expresión facial
    elements.mouth.className = 'mouth ' + (config.mouth || '');
}

// Ajustar color (para gradientes)
function adjustColor(color, amount) {
    return '#' + color.replace(/^#/, '').replace(/../g, color =>
        ('0' + Math.min(255, Math.max(0, parseInt(color, 16) + amount)).toString(16)).substr(-2)
    );
}

// Enviar mensaje
async function sendMessage() {
    const message = elements.messageInput.value.trim();

    if (!message || state.isTyping) return;

    if (!state.isConnected) {
        showNotification('⚠️ Sistema desconectado. Por favor espera...', 'warning');
        return;
    }

    // Limpiar input
    elements.messageInput.value = '';
    elements.charCount.textContent = '0';
    autoResizeTextarea();

    // Mostrar mensaje del usuario
    addMessage('user', message);

    // Mostrar indicador de escritura
    showTypingIndicator();
    state.isTyping = true;
    elements.sendButton.disabled = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: state.userId
            })
        });

        if (!response.ok) {
            throw new Error('Error en la respuesta del servidor');
        }

        const data = await response.json();

        // Ocultar indicador de escritura
        hideTypingIndicator();

        // Mostrar respuesta
        addMessage('assistant', data.response, data.emotion);

        // Actualizar emoción
        updateEmotionDisplay(data.emotion);

        // Reproducir audio si está disponible
        if (data.audio_url) {
            playAudio(data.audio_url);
        }

    } catch (error) {
        console.error('Error enviando mensaje:', error);
        hideTypingIndicator();
        addMessage('assistant', 'Ay Mijin, algo salió mal... 😢 ¿Podrías intentar de nuevo?', 'preocupada');
    } finally {
        state.isTyping = false;
        elements.sendButton.disabled = false;
        elements.messageInput.focus();
    }
}

// Agregar mensaje al chat
function addMessage(role, content, emotion = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🌸';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const text = document.createElement('div');
    text.textContent = content;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('es-ES', {
        hour: '2-digit',
        minute: '2-digit'
    });

    contentDiv.appendChild(text);
    contentDiv.appendChild(time);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    // Remover mensaje de bienvenida si existe
    const welcomeMsg = elements.chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

// Mostrar indicador de escritura
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-message';
    typingDiv.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🌸';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    contentDiv.appendChild(typingIndicator);
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(contentDiv);

    elements.chatContainer.appendChild(typingDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

// Ocultar indicador de escritura
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Limpiar chat
function clearChat() {
    if (confirm('¿Estás seguro de que quieres limpiar el chat?')) {
        elements.chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2 id="welcomeUserHeader">¡Hola ${state.userName || 'Mijin'}! 💞</h2>
                <p>Soy Lily, tu asistente virtual. Puedo hablar de cualquier tema sin restricciones. ¿En qué puedo ayudarte hoy?</p>
            </div>
        `;
    }
}

// Mostrar memoria
async function showMemory() {
    elements.memoryModal.style.display = 'block';
    elements.memoryContent.innerHTML = '<p>Cargando memoria...</p>';

    try {
        const response = await fetch(`/api/memory/${state.userId}`);
        const data = await response.json();

        let html = `
            <div class="memory-section">
                <h3>📊 Resumen de Conversación</h3>
                <p>${data.conversation_summary}</p>
            </div>
            
            <div class="memory-section">
                <h3>💭 Resumen Emocional</h3>
                <p>${data.emotional_summary}</p>
            </div>
            
            <div class="memory-section">
                <h3>💬 Mensajes Recientes</h3>
        `;

        if (data.recent_messages.length > 0) {
            data.recent_messages.forEach(msg => {
                const roleEmoji = msg.role === 'user' ? '👤' : '🌸';
                const emotion = msg.emotion ? ` [${msg.emotion}]` : '';
                html += `
                    <p style="margin-bottom: 10px;">
                        <strong>${roleEmoji} ${msg.role}${emotion}:</strong><br>
                        ${msg.content}
                    </p>
                `;
            });
        } else {
            html += '<p>No hay mensajes recientes.</p>';
        }

        html += '</div>';

        elements.memoryContent.innerHTML = html;

    } catch (error) {
        console.error('Error obteniendo memoria:', error);
        elements.memoryContent.innerHTML = '<p>Error cargando memoria. Por favor intenta de nuevo.</p>';
    }
}

// Mostrar notificación
function showNotification(message, type = 'info') {
    // Implementación simple - se puede mejorar
    alert(message);
}

// Manejar errores globales
window.addEventListener('error', (e) => {
    console.error('Error global:', e.error);
});

// Reproducir audio y eliminarlo después
function playAudio(audioUrl) {
    try {
        // Si hay un audio reproduciéndose actualmente, detenerlo
        if (state.currentAudio) {
            state.currentAudio.pause();
            state.currentAudio = null;
        }

        const audio = new Audio(audioUrl);
        state.currentAudio = audio;
        audio.volume = 0.8;

        // Eliminar el audio después de que termine de reproducirse
        audio.addEventListener('ended', () => {
            state.currentAudio = null;
            deleteAudioFile(audioUrl);
            
            // Si el modo de conversación continua está activo, iniciar captura automáticamente
            if (state.isVoiceModeActive) {
                console.log("Audio finalizado, reactivando captura de voz...");
                startVoiceCapture();
            }
        });

        // También eliminar si hay error
        audio.addEventListener('error', () => {
            console.error('Error reproduciendo audio');
            state.currentAudio = null;
            deleteAudioFile(audioUrl);
            
            // Intentar reactivar captura en caso de error
            if (state.isVoiceModeActive) {
                startVoiceCapture();
            }
        });

        audio.play().catch(error => {
            console.error('Error reproduciendo audio:', error);
            state.currentAudio = null;
            deleteAudioFile(audioUrl);
            
            // Intentar reactivar captura
            if (state.isVoiceModeActive) {
                startVoiceCapture();
            }
        });
    } catch (error) {
        console.error('Error creando audio:', error);
    }
}

// Eliminar archivo de audio del servidor
async function deleteAudioFile(audioUrl) {
    try {
        // Extraer el nombre del archivo de la URL
        const filename = audioUrl.split('/').pop();

        const response = await fetch(`/api/audio/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            console.log(`Audio eliminado: ${filename}`);
        }
    } catch (error) {
        console.error('Error eliminando audio:', error);
    }
}

// Iniciar escucha de palabra clave
function startWakeWordListening() {
    if (!wakeWordRecognition || isListeningForWakeWord) return;

    try {
        isListeningForWakeWord = true;
        wakeWordRecognition.start();
        console.log('🌸 Escuchando palabra clave "Lily"...');
    } catch (error) {
        console.error('Error al iniciar wake word listening:', error);
    }
}

// Detener escucha de palabra clave
function stopWakeWordListening() {
    if (!wakeWordRecognition || !isListeningForWakeWord) return;

    try {
        isListeningForWakeWord = false;
        wakeWordRecognition.stop();
        console.log('Palabra clave detenida');
    } catch (error) {
        console.error('Error al detener wake word listening:', error);
    }
}
// Activar/desactivar modo de conversación continua por voz
function toggleRecording() {
    if (state.isVoiceModeActive) {
        state.isVoiceModeActive = false;
        stopVoiceMode();
    } else {
        state.isVoiceModeActive = true;
        startVoiceMode();
    }
}

// Iniciar el modo de conversación continua por voz
function startVoiceMode() {
    console.log("Modo de conversación por voz activado");
    elements.micButton.classList.add('recording');
    elements.micButton.title = "Desactivar conversación por voz";
    
    // Detener detección de wake word en el navegador
    stopWakeWordListening();
    
    // Detener reproducción actual si existe
    if (state.currentAudio) {
        state.currentAudio.pause();
        state.currentAudio = null;
    }
    
    startVoiceCapture();
}

// Detener el modo de conversación por voz
function stopVoiceMode() {
    console.log("Modo de conversación por voz desactivado");
    elements.micButton.classList.remove('recording');
    elements.micButton.title = "Hablar";
    
    // Detener grabación si está en progreso
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    
    // Detener reproducción de voz actual
    if (state.currentAudio) {
        state.currentAudio.pause();
        state.currentAudio = null;
    }
    
    // Detener analizador
    cleanupAudioAnalyser();
    elements.messageInput.placeholder = `Escribe tu mensaje aquí, ${state.userName || "Mijin"}...`;
    
    // Reactivar detección de palabra clave
    setTimeout(() => {
        if (wakeWordRecognition && !isListeningForWakeWord) {
            startWakeWordListening();
        }
    }, 1000);
}

// Capturar voz del micrófono
function startVoiceCapture() {
    if (!state.isVoiceModeActive) return;
    
    isRecording = true;
    elements.micButton.classList.add('recording');
    elements.messageInput.placeholder = "Escuchando...";
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                if (state.isVoiceModeActive && audioChunks.length > 0) {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    sendAudioToTranscribe(audioBlob);
                }
                cleanupAudioAnalyser();
            });

            mediaRecorder.start();
            
            // Iniciar detección de silencio
            startSilenceDetection(stream);
        })
        .catch(error => {
            console.error("Error al acceder al micrófono:", error);
            alert("No se pudo acceder al micrófono.");
            state.isVoiceModeActive = false;
            stopVoiceMode();
        });
}

// Analizador de silencio para finalizar grabación automáticamente
let audioContext = null;
let analyser = null;
let microphone = null;
let speakingStarted = false;

function startSilenceDetection(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        
        analyser.fftSize = 512;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        microphone.connect(analyser);
        
        const threshold = 12; // Umbral de volumen para detectar sonido
        const silenceDuration = 1800; // 1.8s de silencio continuo para detener
        let lastSoundTime = Date.now();
        speakingStarted = false;
        
        function checkAudio() {
            if (!isRecording || !state.isVoiceModeActive) return;
            
            analyser.getByteFrequencyData(dataArray);
            let values = 0;
            for (let i = 0; i < bufferLength; i++) {
                values += dataArray[i];
            }
            const average = values / bufferLength;
            
            if (average > threshold) {
                lastSoundTime = Date.now();
                if (!speakingStarted) {
                    speakingStarted = true;
                    console.log("Vocalización detectada...");
                    elements.messageInput.placeholder = "Escuchando tu voz...";
                }
            } else {
                if (speakingStarted && (Date.now() - lastSoundTime > silenceDuration)) {
                    console.log("Silencio detectado, deteniendo captura de voz...");
                    isRecording = false;
                    elements.micButton.classList.remove('recording');
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                    return;
                }
            }
            
            requestAnimationFrame(checkAudio);
        }
        
        requestAnimationFrame(checkAudio);
    } catch (e) {
        console.error("Error inicializando analizador de silencio:", e);
    }
}

// Limpiar recursos de audio
function cleanupAudioAnalyser() {
    try {
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
        analyser = null;
        if (microphone) {
            microphone.disconnect();
            microphone = null;
        }
    } catch (e) {
        console.error("Error limpiando analizador de audio:", e);
    }
}

// Enviar audio al backend para transcripción
async function sendAudioToTranscribe(audioBlob) {
    elements.messageInput.placeholder = "Escuchando y transcribiendo...";

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en transcripción');

        const data = await response.json();
        const transcript = data.text;

        if (transcript) {
            elements.messageInput.value = transcript;
            if (elements.charCount) {
                elements.charCount.textContent = transcript.length;
            }
            autoResizeTextarea();

            // Enviar automáticamente el mensaje al chat
            setTimeout(() => {
                sendMessage();
            }, 500);
        } else {
            // Si la transcripción está vacía (ruido), reactivar escucha si el modo está activo
            if (state.isVoiceModeActive) {
                console.log("Transcripción vacía, reactivando escucha...");
                startVoiceCapture();
            }
        }

    } catch (error) {
        console.error("Error transcribiendo:", error);
        elements.messageInput.value = "Error al escuchar.";
        
        // Intentar reactivar en caso de error
        if (state.isVoiceModeActive) {
            setTimeout(startVoiceCapture, 2000);
        }
    } finally {
        elements.messageInput.placeholder = `Escribe tu mensaje aquí, ${state.userName || "Mijin"}...`;
    }
}

// Mostrar modal de configuración y cargar RAG
async function showSettings() {
    console.log("Abriendo ventana de configuración (RAG)...");
    elements.settingsModal.style.display = 'block';
    loadRagStats();
    loadRagDocuments();
    loadUserPreferences();
}

// Mostrar modal de comandos
function showCommands() {
    elements.commandsModal.style.display = 'block';
}

// Cargar estadísticas RAG
async function loadRagStats() {
    try {
        elements.ragStats.innerHTML = "<p>Cargando estadísticas...</p>";
        const response = await fetch('/api/rag/stats');
        if (!response.ok) throw new Error('Error al cargar estadísticas');
        const stats = await response.json();
        
        elements.ragStats.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_documents || 0}</div>
                <div class="stat-label">Documentos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.total_chunks || 0}</div>
                <div class="stat-label">Pasajes (Chunks)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="font-size: 14px; padding-top: 5px;">${stats.similarity_threshold || 0.0}</div>
                <div class="stat-label">Umbral de Similitud</div>
            </div>
        `;
    } catch (error) {
        console.error('Error cargando stats RAG:', error);
        elements.ragStats.innerHTML = `<p style="color: red; font-size: 13px;">Error al cargar estadísticas</p>`;
    }
}

// Cargar lista de documentos RAG
async function loadRagDocuments() {
    try {
        elements.ragDocumentsList.innerHTML = "<p>Cargando documentos indexados...</p>";
        const response = await fetch('/api/rag/documents');
        if (!response.ok) throw new Error('Error al obtener documentos');
        const data = await response.json();
        
        const docs = data.documents || [];
        if (docs.length === 0) {
            elements.ragDocumentsList.innerHTML = "<p style='color: #666; font-size: 13px; text-align: center; padding: 10px;'>No hay documentos indexados. Sube un archivo arriba o presiona Sincronizar.</p>";
            return;
        }
        
        elements.ragDocumentsList.innerHTML = docs.map(doc => {
            const name = doc.metadata.source || doc.id;
            const shortName = name.split(/[\\/]/).pop();
            const chunkCount = doc.metadata.chunks || "N/A";
            
            return `
                <div class="doc-item">
                    <div class="doc-name" title="${name}">
                        📄 ${shortName}
                        <span class="doc-meta">(${chunkCount} chunks)</span>
                    </div>
                    <button class="delete-doc-btn" onclick="deleteRagDocument('${doc.id}')" title="Eliminar de base de conocimiento">🗑️</button>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error cargando documentos RAG:', error);
        elements.ragDocumentsList.innerHTML = `<p style="color: red; font-size: 13px;">Error al cargar lista de documentos</p>`;
    }
}

// Subir un archivo al RAG
async function uploadRagFile() {
    const fileInput = elements.ragFileInput;
    if (fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    elements.uploadStatus.className = "status-msg";
    elements.uploadStatus.textContent = "Subiendo y procesando archivo... Por favor espera.";
    elements.uploadFileBtn.disabled = true;
    
    try {
        const response = await fetch('/api/rag/upload-document', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Error en el servidor');
        }
        
        const data = await response.json();
        
        elements.uploadStatus.className = "status-msg success";
        elements.uploadStatus.textContent = `✅ Archivo indexado correctamente: ${file.name}`;
        
        // Reset file input
        fileInput.value = "";
        elements.selectedFileName.textContent = "Ningún archivo seleccionado";
        
        // Recargar stats y docs
        loadRagStats();
        loadRagDocuments();
    } catch (error) {
        console.error('Error subiendo archivo al RAG:', error);
        elements.uploadStatus.className = "status-msg error";
        elements.uploadStatus.textContent = `❌ Error: ${error.message}`;
        elements.uploadFileBtn.disabled = false;
    }
}

// Eliminar un documento del RAG
async function deleteRagDocument(docId) {
    if (!confirm("¿Estás seguro de que deseas eliminar este documento de la base de conocimiento?")) return;
    
    try {
        const response = await fetch(`/api/rag/document/${docId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Error al eliminar documento');
        
        // Recargar stats y docs
        loadRagStats();
        loadRagDocuments();
    } catch (error) {
        console.error('Error eliminando documento RAG:', error);
        alert("No se pudo eliminar el documento de la base de conocimiento.");
    }
}

// Sincronizar directorio knowledge
async function reindexRagKnowledge() {
    elements.reindexBtn.disabled = true;
    const oldText = elements.reindexBtn.textContent;
    elements.reindexBtn.textContent = "🔄 Sincronizando...";
    
    try {
        const response = await fetch('/api/rag/ingest-knowledge', {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Error en el servidor');
        const data = await response.json();
        
        alert(`Sincronización completada. Archivos procesados: ${data.files_ingested}`);
        loadRagStats();
        loadRagDocuments();
    } catch (error) {
        console.error('Error sincronizando directorio knowledge:', error);
        alert("Error al sincronizar directorio.");
    } finally {
        elements.reindexBtn.disabled = false;
        elements.reindexBtn.textContent = oldText;
    }
}

// Cargar preferencias del usuario (Gmail y Nombre)
async function loadUserPreferences() {
    try {
        if (elements.gmailConfigStatus) elements.gmailConfigStatus.textContent = "";
        if (elements.userNameStatus) elements.userNameStatus.textContent = "";
        
        const response = await fetch(`/api/preferences/${state.userId}`);
        if (!response.ok) throw new Error('Error al cargar preferencias');
        
        const data = await response.json();
        
        // 1. Cargar datos de Gmail
        if (elements.gmailUserInput && elements.gmailPasswordInput) {
            elements.gmailUserInput.value = data.gmail_user || "";
            
            if (data.has_password) {
                elements.gmailPasswordInput.placeholder = "******** (Contraseña guardada)";
                elements.gmailPasswordInput.value = ""; 
            } else {
                elements.gmailPasswordInput.placeholder = "Contraseña de aplicación de 16 caracteres";
                elements.gmailPasswordInput.value = "";
            }
        }
        
        // 2. Cargar datos de Nombre del usuario
        state.userName = data.user_name || "Mijin";
        if (elements.userNameInput) {
            elements.userNameInput.value = state.userName;
        }
        if (elements.welcomeUserHeader) {
            elements.welcomeUserHeader.innerHTML = `¡Hola ${state.userName}! 💞`;
        }
        
        // Actualizar placeholder principal de mensajes
        if (elements.messageInput) {
            elements.messageInput.placeholder = `Escribe tu mensaje aquí, ${state.userName}...`;
        }
    } catch (error) {
        console.error('Error cargando preferencias de usuario:', error);
    }
}

// Guardar preferencias de Gmail en SQLite
async function saveGmailPreferences() {
    try {
        if (!elements.gmailUserInput || !elements.gmailPasswordInput) return;
        
        const user = elements.gmailUserInput.value.trim();
        const password = elements.gmailPasswordInput.value;
        
        const payload = {
            "gmail_user": user
        };
        
        if (password) {
            payload["gmail_password"] = password;
        }
        
        elements.gmailConfigStatus.className = "status-msg";
        elements.gmailConfigStatus.textContent = "Guardando credenciales...";
        elements.saveGmailBtn.disabled = true;
        
        const response = await fetch(`/api/preferences/${state.userId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error('Error al guardar credenciales');
        
        elements.gmailConfigStatus.className = "status-msg success";
        elements.gmailConfigStatus.textContent = "✅ Credenciales de Gmail guardadas correctamente";
        
        loadUserPreferences();
    } catch (error) {
        console.error('Error guardando credenciales:', error);
        elements.gmailConfigStatus.className = "status-msg error";
        elements.gmailConfigStatus.textContent = `❌ Error: {error.message}`;
    } finally {
        elements.saveGmailBtn.disabled = false;
    }
}

// Guardar nombre de usuario personalizado en SQLite
async function saveUserName() {
    try {
        if (!elements.userNameInput) return;
        
        const name = elements.userNameInput.value.trim();
        if (!name) {
            elements.userNameStatus.className = "status-msg error";
            elements.userNameStatus.textContent = "❌ El nombre no puede estar vacío";
            return;
        }
        
        elements.userNameStatus.className = "status-msg";
        elements.userNameStatus.textContent = "Guardando nombre...";
        elements.saveUserNameBtn.disabled = true;
        
        const response = await fetch(`/api/preferences/${state.userId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "user_name": name })
        });
        
        if (!response.ok) throw new Error('Error al guardar nombre');
        
        elements.userNameStatus.className = "status-msg success";
        elements.userNameStatus.textContent = "✅ Nombre guardado correctamente";
        
        await loadUserPreferences();
    } catch (error) {
        console.error('Error guardando nombre:', error);
        elements.userNameStatus.className = "status-msg error";
        elements.userNameStatus.textContent = `❌ Error: ${error.message}`;
    } finally {
        elements.saveUserNameBtn.disabled = false;
    }
}

// Exponer deleteRagDocument globalmente para los botones HTML onclick
window.deleteRagDocument = deleteRagDocument;

// Cambiar de tema claro a oscuro
function toggleTheme() {
    if (document.body.classList.contains('dark-theme')) {
        document.body.classList.remove('dark-theme');
        if (elements.themeToggle) elements.themeToggle.textContent = '🌙';
        localStorage.setItem('lily-theme', 'light');
        state.theme = 'light';
    } else {
        document.body.classList.add('dark-theme');
        if (elements.themeToggle) elements.themeToggle.textContent = '☀️';
        localStorage.setItem('lily-theme', 'dark');
        state.theme = 'dark';
    }
}

// Log de inicio
console.log('🌸 Lily AI Assistant - Interfaz cargada correctamente');

// Sintetizar voz para un texto dado y reproducirlo
async function speakText(text, emotion = "neutral") {
    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                emotion: emotion
            })
        });
        if (!response.ok) throw new Error('Error en TTS');
        const data = await response.json();
        if (data.audio_url) {
            playAudio(data.audio_url);
        }
    } catch (e) {
        console.error('Error en speakText:', e);
    }
}
