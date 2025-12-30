<template>
    <div class="docling-kb">
        <!-- Header -->
        <header class="docling-header">
            <div class="header-left">
                <span class="app-icon">📚</span>
                <h1>Docling Knowledge Base</h1>
            </div>
            <div class="header-stats">
                <span class="stat">
                    <span class="stat-value">{{ stats.total_documents }}</span>
                    <span class="stat-label">Documents</span>
                </span>
                <span class="stat">
                    <span class="stat-value">{{ stats.total_chunks }}</span>
                    <span class="stat-label">Chunks</span>
                </span>
            </div>
        </header>

        <!-- Main content -->
        <div class="docling-content">
            <!-- Sidebar: Document List -->
            <aside class="docling-sidebar">
                <div class="sidebar-header">
                    <h2>📁 Documents</h2>
                    <button class="btn-upload" @click="triggerUpload">
                        <span>+ Add</span>
                    </button>
                    <input 
                        ref="fileInput" 
                        type="file" 
                        accept=".pdf,.docx,.pptx,.xlsx,.html,.md,.txt"
                        multiple
                        hidden 
                        @change="handleFileUpload"
                    />
                </div>

                <!-- Processing Queue -->
                <div v-if="processingJobs.length" class="processing-queue">
                    <h3>⏳ Processing</h3>
                    <div 
                        v-for="job in processingJobs" 
                        :key="job.doc_id" 
                        class="processing-item"
                    >
                        <span class="filename">{{ job.filename }}</span>
                        <div class="progress-bar">
                            <div 
                                class="progress-fill" 
                                :style="{ width: job.progress + '%' }"
                            ></div>
                        </div>
                    </div>
                </div>

                <!-- Document List -->
                <div class="document-list">
                    <div 
                        v-for="doc in documents" 
                        :key="doc.doc_id"
                        class="document-item"
                        :class="{ active: selectedDoc === doc.doc_id }"
                        @click="selectDocument(doc)"
                    >
                        <span class="doc-icon">{{ getDocIcon(doc.filename) }}</span>
                        <div class="doc-info">
                            <span class="doc-name">{{ doc.filename }}</span>
                            <span class="doc-meta">{{ doc.chunks }} chunks · {{ formatDate(doc.processed_at) }}</span>
                        </div>
                    </div>
                    <div v-if="!documents.length" class="empty-state">
                        <p>No documents yet</p>
                        <p class="hint">Upload PDFs, DOCX, or other files to get started</p>
                    </div>
                </div>
            </aside>

            <!-- Main: Chat Interface -->
            <main class="docling-main">
                <div class="chat-container">
                    <!-- Chat Messages -->
                    <div class="chat-messages" ref="chatMessages">
                        <div 
                            v-for="(msg, index) in chatHistory" 
                            :key="index"
                            class="chat-message"
                            :class="msg.role"
                        >
                            <div class="message-avatar">
                                {{ msg.role === 'user' ? '👤' : '🤖' }}
                            </div>
                            <div class="message-content">
                                <div class="message-text" v-html="formatMessage(msg.content)"></div>
                                <div v-if="msg.sources" class="message-sources">
                                    <span class="sources-label">Sources:</span>
                                    <span 
                                        v-for="src in msg.sources" 
                                        :key="src" 
                                        class="source-tag"
                                    >{{ src }}</span>
                                </div>
                            </div>
                        </div>
                        <div v-if="isThinking" class="chat-message assistant thinking">
                            <div class="message-avatar">🤖</div>
                            <div class="message-content">
                                <div class="typing-indicator">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Chat Input -->
                    <div class="chat-input-container">
                        <textarea
                            v-model="chatInput"
                            class="chat-input"
                            placeholder="Ask anything about your documents..."
                            @keydown.enter.exact.prevent="sendMessage"
                            rows="1"
                        ></textarea>
                        <button 
                            class="btn-send" 
                            @click="sendMessage"
                            :disabled="!chatInput.trim() || isThinking"
                        >
                            Send
                        </button>
                    </div>
                </div>
            </main>
        </div>
    </div>
</template>

<script>
export default {
    name: 'DoclingKB',
    
    data() {
        return {
            stats: { total_documents: 0, total_chunks: 0 },
            documents: [],
            processingJobs: [],
            selectedDoc: null,
            chatHistory: [],
            chatInput: '',
            isThinking: false,
            pollInterval: null,
        }
    },

    mounted() {
        this.loadStats()
        this.loadDocuments()
        this.startPolling()
        
        // Welcome message
        this.chatHistory.push({
            role: 'assistant',
            content: '👋 Welcome to Docling Knowledge Base!\n\nI can help you search and understand your documents. Upload some files or ask me anything about the documents you\'ve already processed.',
        })
    },

    beforeDestroy() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval)
        }
    },

    methods: {
        async loadStats() {
            try {
                this.stats = await this.$api.getStats()
            } catch (e) {
                console.error('Failed to load stats:', e)
            }
        },

        async loadDocuments() {
            try {
                const response = await this.$api.getDocuments()
                this.documents = response.documents || []
            } catch (e) {
                console.error('Failed to load documents:', e)
            }
        },

        triggerUpload() {
            this.$refs.fileInput.click()
        },

        async handleFileUpload(event) {
            const files = event.target.files
            for (const file of files) {
                try {
                    const result = await this.$api.processDocument(file)
                    this.processingJobs.push({
                        doc_id: result.doc_id,
                        filename: file.name,
                        progress: 0,
                    })
                } catch (e) {
                    console.error('Failed to upload:', e)
                }
            }
            event.target.value = ''
        },

        selectDocument(doc) {
            this.selectedDoc = doc.doc_id
        },

        async sendMessage() {
            if (!this.chatInput.trim() || this.isThinking) return

            const message = this.chatInput.trim()
            this.chatInput = ''
            
            this.chatHistory.push({ role: 'user', content: message })
            this.isThinking = true
            this.scrollToBottom()

            try {
                const response = await this.$api.chat(message, this.selectedDoc ? [this.selectedDoc] : null)
                this.chatHistory.push({
                    role: 'assistant',
                    content: response.response,
                    sources: response.sources,
                })
            } catch (e) {
                this.chatHistory.push({
                    role: 'assistant',
                    content: '❌ Sorry, I encountered an error. Please try again.',
                })
            }

            this.isThinking = false
            this.scrollToBottom()
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const el = this.$refs.chatMessages
                if (el) el.scrollTop = el.scrollHeight
            })
        },

        startPolling() {
            this.pollInterval = setInterval(() => {
                this.loadStats()
                this.loadDocuments()
            }, 5000)
        },

        getDocIcon(filename) {
            const ext = filename.split('.').pop().toLowerCase()
            const icons = {
                pdf: '📄',
                docx: '📝',
                doc: '📝',
                pptx: '📊',
                xlsx: '📈',
                html: '🌐',
                md: '📋',
                txt: '📃',
            }
            return icons[ext] || '📄'
        },

        formatDate(dateStr) {
            if (!dateStr) return ''
            const date = new Date(dateStr)
            return date.toLocaleDateString()
        },

        formatMessage(content) {
            // Simple markdown-like formatting
            return content
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
        },
    },
}
</script>

<style scoped>
.docling-kb {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-main-background);
    font-family: var(--font-face);
}

.docling-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-icon {
    font-size: 28px;
}

.header-left h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
}

.header-stats {
    display: flex;
    gap: 24px;
}

.stat {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.stat-value {
    font-size: 24px;
    font-weight: 700;
}

.stat-label {
    font-size: 11px;
    opacity: 0.8;
    text-transform: uppercase;
}

.docling-content {
    display: flex;
    flex: 1;
    overflow: hidden;
}

.docling-sidebar {
    width: 300px;
    border-right: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    background: var(--color-main-background);
}

.sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--color-border);
}

.sidebar-header h2 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
}

.btn-upload {
    background: #667eea;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
}

.btn-upload:hover {
    background: #5a6fd6;
}

.processing-queue {
    padding: 12px 16px;
    background: #fff8e1;
    border-bottom: 1px solid var(--color-border);
}

.processing-queue h3 {
    margin: 0 0 8px 0;
    font-size: 12px;
    font-weight: 600;
}

.processing-item {
    margin-bottom: 8px;
}

.processing-item .filename {
    font-size: 12px;
    display: block;
    margin-bottom: 4px;
}

.progress-bar {
    height: 4px;
    background: #e0e0e0;
    border-radius: 2px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: #667eea;
    transition: width 0.3s;
}

.document-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.document-item {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    gap: 10px;
}

.document-item:hover {
    background: var(--color-background-hover);
}

.document-item.active {
    background: #e8eaf6;
}

.doc-icon {
    font-size: 20px;
}

.doc-info {
    flex: 1;
    min-width: 0;
}

.doc-name {
    display: block;
    font-size: 13px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.doc-meta {
    font-size: 11px;
    color: var(--color-text-lighter);
}

.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--color-text-lighter);
}

.empty-state .hint {
    font-size: 12px;
}

.docling-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.chat-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.chat-message {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    max-width: 80%;
}

.chat-message.user {
    margin-left: auto;
    flex-direction: row-reverse;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--color-background-dark);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.chat-message.user .message-avatar {
    background: #667eea;
}

.message-content {
    background: var(--color-background-dark);
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 100%;
}

.chat-message.user .message-content {
    background: #667eea;
    color: white;
}

.message-text {
    font-size: 14px;
    line-height: 1.5;
}

.message-sources {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(0,0,0,0.1);
    font-size: 11px;
}

.source-tag {
    background: rgba(0,0,0,0.1);
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 4px;
}

.typing-indicator {
    display: flex;
    gap: 4px;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: var(--color-text-lighter);
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
}

.chat-input-container {
    display: flex;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid var(--color-border);
    background: var(--color-main-background);
}

.chat-input {
    flex: 1;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    resize: none;
    min-height: 44px;
    max-height: 120px;
}

.chat-input:focus {
    outline: none;
    border-color: #667eea;
}

.btn-send {
    background: #667eea;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
}

.btn-send:hover:not(:disabled) {
    background: #5a6fd6;
}

.btn-send:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
</style>

