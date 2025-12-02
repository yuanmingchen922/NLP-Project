/**
 * Chatbot Widget - Floating chat interface component
 */

class ChatbotWidget {
    constructor() {
        this.isOpen = false;
        this.isProcessing = false;
        this.init();
    }

    init() {
        // Create widget HTML
        this.createWidget();
        // Attach event listeners
        this.attachEventListeners();
    }

    createWidget() {
        const widgetHTML = `
            <!-- Floating Chat Button -->
            <div id="chatbot-button" class="chatbot-button">
                <svg width="24" height="24" fill="white" viewBox="0 0 16 16">
                    <path d="M2.678 11.894a1 1 0 0 1 .287.801 10.97 10.97 0 0 1-.398 2c1.395-.323 2.247-.697 2.634-.893a1 1 0 0 1 .71-.074A8.06 8.06 0 0 0 8 14c3.996 0 7-2.807 7-6 0-3.192-3.004-6-7-6S1 4.808 1 8c0 1.468.617 2.83 1.678 3.894zm-.493 3.905a21.682 21.682 0 0 1-.713.129c-.2.032-.352-.176-.273-.362a9.68 9.68 0 0 0 .244-.637l.003-.01c.248-.72.45-1.548.524-2.319C.743 11.37 0 9.76 0 8c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7a9.06 9.06 0 0 1-2.347-.306c-.52.263-1.639.742-3.468 1.105z"/>
                </svg>
                <span class="chatbot-badge">AI</span>
            </div>

            <!-- Chat Window -->
            <div id="chatbot-window" class="chatbot-window">
                <!-- Header -->
                <div class="chatbot-header">
                    <div>
                        <h5 style="margin: 0; font-weight: 700;">Restaurant AI Assistant</h5>
                        <small style="opacity: 0.9;">Powered by GPT-4</small>
                    </div>
                    <button id="chatbot-close" class="chatbot-close-btn">
                        <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M2.146 2.854a.5.5 0 1 1 .708-.708L8 7.293l5.146-5.147a.5.5 0 0 1 .708.708L8.707 8l5.147 5.146a.5.5 0 0 1-.708.708L8 8.707l-5.146 5.147a.5.5 0 0 1-.708-.708L7.293 8 2.146 2.854Z"/>
                        </svg>
                    </button>
                </div>

                <!-- Chat Messages -->
                <div id="chatbot-messages" class="chatbot-messages">
                    <div class="chatbot-message bot-message">
                        <div class="message-content">
                            <strong>👋 Welcome!</strong><br>
                            I can help you with:
                            <ul style="margin: 8px 0 0 0; padding-left: 20px;">
                                <li>Restaurant recommendations</li>
                                <li>Factual information (WiFi, parking, etc.)</li>
                                <li>Review summaries and insights</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Suggestions -->
                <div id="chatbot-suggestions" class="chatbot-suggestions">
                    <button class="suggestion-chip" data-message="Recommend a good Italian restaurant">
                        💡 Recommend Italian
                    </button>
                    <button class="suggestion-chip" data-message="Does Starbucks have WiFi?">
                        ℹ️ Ask about WiFi
                    </button>
                    <button class="suggestion-chip" data-message="How is McDonald's?">
                        📝 Review summary
                    </button>
                </div>

                <!-- Input Area -->
                <div class="chatbot-input-area">
                    <input
                        type="text"
                        id="chatbot-input"
                        class="chatbot-input"
                        placeholder="Ask me anything..."
                    />
                    <button id="chatbot-send" class="chatbot-send-btn">
                        <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576 6.636 10.07Zm6.787-8.201L1.591 6.602l4.339 2.76 7.494-7.493Z"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', widgetHTML);
    }

    attachEventListeners() {
        // Toggle chat window
        document.getElementById('chatbot-button').addEventListener('click', () => {
            this.toggleWindow();
        });

        // Close button
        document.getElementById('chatbot-close').addEventListener('click', () => {
            this.closeWindow();
        });

        // Send message
        document.getElementById('chatbot-send').addEventListener('click', () => {
            this.sendMessage();
        });

        // Enter key to send
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isProcessing) {
                this.sendMessage();
            }
        });

        // Suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                const message = e.target.getAttribute('data-message');
                document.getElementById('chatbot-input').value = message;
                this.sendMessage();
            });
        });
    }

    toggleWindow() {
        this.isOpen = !this.isOpen;
        const window = document.getElementById('chatbot-window');
        const button = document.getElementById('chatbot-button');

        if (this.isOpen) {
            window.classList.add('open');
            button.classList.add('hidden');
            document.getElementById('chatbot-input').focus();
        } else {
            window.classList.remove('open');
            button.classList.remove('hidden');
        }
    }

    closeWindow() {
        this.isOpen = false;
        document.getElementById('chatbot-window').classList.remove('open');
        document.getElementById('chatbot-button').classList.remove('hidden');
    }

    async sendMessage() {
        if (this.isProcessing) return;

        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();

        if (!message) return;

        this.isProcessing = true;
        input.value = '';

        // Add user message
        this.addMessage(message, 'user');

        // Show loading
        this.addLoadingMessage();

        try {
            const response = await fetch('/api/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    type: 'auto'
                })
            });

            const data = await response.json();

            // Remove loading
            this.removeLoadingMessage();

            // Add bot response
            this.addBotResponse(data);

        } catch (error) {
            this.removeLoadingMessage();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Chatbot error:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    addMessage(message, sender) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatbot-message ${sender}-message fade-in`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = this.formatMessage(message);

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    addBotResponse(data) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chatbot-message bot-message fade-in';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Add module badge
        let badge = '';
        if (data.type === 'recommendation') {
            badge = '<span class="module-badge module-recommendation">💡 Recommendations</span><br>';
        } else if (data.type === 'factual') {
            badge = '<span class="module-badge module-factual">ℹ️ Factual Info</span><br>';
        } else if (data.type === 'summary') {
            badge = '<span class="module-badge module-summary">📝 Review Summary</span><br>';
        }

        contentDiv.innerHTML = badge + this.formatMessage(data.message || 'No response');

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    addLoadingMessage() {
        const messagesContainer = document.getElementById('chatbot-messages');
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chatbot-message bot-message loading-message';
        loadingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeLoadingMessage() {
        const loadingMessage = document.querySelector('.loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    formatMessage(message) {
        // Convert markdown-style bold to HTML
        message = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Convert newlines to <br>
        message = message.replace(/\n/g, '<br>');
        return message;
    }
}

// Initialize chatbot widget when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChatbotWidget();
});
