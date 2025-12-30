/**
 * Docling Knowledge Base - Nextcloud App
 * Main entry point for the Vue.js frontend
 * 
 * @copyright Copyright (c) 2024-2025 Ihsan Mokhlis
 * @license CC-BY-NC-SA-4.0
 */

import Vue from 'vue'
import App from './App.vue'
import { generateUrl } from '@nextcloud/router'
import { showSuccess, showError } from '@nextcloud/dialogs'
import '@nextcloud/dialogs/styles/toast.scss'

Vue.mixin({
    methods: {
        t: (app, text) => text, // Simple translation stub
    },
})

// API base URL for the ExApp
const EXAPP_BASE_URL = generateUrl('/apps/docling_kb/api')

Vue.prototype.$api = {
    baseUrl: EXAPP_BASE_URL,
    
    async getStats() {
        const response = await fetch(`${EXAPP_BASE_URL}/stats`)
        return response.json()
    },
    
    async getDocuments() {
        const response = await fetch(`${EXAPP_BASE_URL}/documents`)
        return response.json()
    },
    
    async processDocument(file) {
        const formData = new FormData()
        formData.append('file', file)
        const response = await fetch(`${EXAPP_BASE_URL}/process`, {
            method: 'POST',
            body: formData,
        })
        return response.json()
    },
    
    async searchDocuments(query) {
        const response = await fetch(`${EXAPP_BASE_URL}/search?query=${encodeURIComponent(query)}`)
        return response.json()
    },
    
    async chat(message, documentIds = null) {
        const response = await fetch(`${EXAPP_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, document_ids: documentIds }),
        })
        return response.json()
    },
}

new Vue({
    el: '#docling-kb-app',
    render: h => h(App),
})

