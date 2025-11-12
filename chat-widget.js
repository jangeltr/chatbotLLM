document.addEventListener("DOMContentLoaded", () => {
    const toggleButton = document.getElementById('chat-toggle-button');
    const chatWindow = document.getElementById('chat-window');
    const messagesContainer = document.getElementById('chat-messages');
    const input = document.getElementById('chat-input');
    const sendButton = document.getElementById('chat-send-button');

    // URL de tu API de FastAPI. Asegúrate que la IP y puerto sean correctos.
    // Si la landing page está en el mismo servidor, puedes usar localhost o la IP del servidor.
    const API_URL = 'http://localhost:8000/chat'; 

    toggleButton.addEventListener('click', () => {
        chatWindow.classList.toggle('hidden');
    });

    sendButton.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight; // Auto-scroll
        return messageDiv;
    }

    async function sendMessage() {
        const userMessage = input.value.trim();
        if (userMessage === '') return;

        addMessage(userMessage, 'user');
        input.value = '';

        const loadingMessage = addMessage('Pensando...', 'bot');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: userMessage }),
            });

            if (!response.ok) {
                throw new Error('Error en la comunicación con el servidor.');
            }

            const data = await response.json();
            
            // Actualizar el mensaje de "Pensando..." con la respuesta real
            loadingMessage.textContent = data.respuesta;

            // Opcional: Mostrar las fuentes
            if (data.fuentes && data.fuentes.length > 0) {
                const sourcesText = `Fuentes: ${data.fuentes.join(', ')}`;
                addMessage(sourcesText, 'bot');
            }

        } catch (error) {
            loadingMessage.textContent = 'Lo siento, ocurrió un error al procesar tu solicitud.';
            console.error('Error:', error);
        }
    }
});