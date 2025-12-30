# 📚 Docling Knowledge Base for Nextcloud

[![Build](https://img.shields.io/github/actions/workflow/status/your-org/docling-nextcloud/build.yml?branch=main)](https://github.com/your-org/docling-nextcloud/actions)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Nextcloud](https://img.shields.io/badge/Nextcloud-28%2B-blue?logo=nextcloud)](https://nextcloud.com)

**Transform your Nextcloud into an AI-powered Knowledge Base!**

Chat with your documents, search by meaning, and extract structured data—all without sending your files to external services.

![Screenshot](img/screenshot.png)

---

## 🎯 Vision

### Phase 1: Self-Contained App ✅ (Current)
Everything runs **locally within your Nextcloud**:
- No external services required
- Your data never leaves your server
- Full control over your infrastructure

### Phase 2: Optional SaaS Backend 🔮 (Future)
For users who want managed infrastructure:
- Cloud-hosted processing
- Enterprise scalability
- Managed updates and maintenance
- Same app, optional cloud backend

---

## ✨ Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR NEXTCLOUD                                │
│                                                                      │
│   📁 Upload/Sync      →    🔄 Auto-Process        →    💬 Chat      │
│      Documents              with Docling                with AI      │
│                                  ↓                                   │
│                         📊 Knowledge Base                            │
│                         ┌─────────────────┐                          │
│                         │ • Vector Search │                          │
│                         │ • Tables/Data   │                          │
│                         │ • Full Text     │                          │
│                         └─────────────────┘                          │
│                                  ↓                                   │
│                         🔍 "What were Q4 sales?"                     │
│                         📋 "Summarize the contract"                  │
│                         📊 "Show me the budget table"                │
└─────────────────────────────────────────────────────────────────────┘
```

| Feature | Description |
|---------|-------------|
| **📁 Auto-Processing** | Documents processed automatically when added |
| **💬 Chat Interface** | Ask questions across ALL your documents |
| **🔍 Semantic Search** | Find by meaning, not just keywords |
| **📊 Table Extraction** | Access structured data from documents |
| **🏷️ Smart Chunking** | Documents split intelligently for better answers |
| **🔒 100% Private** | Everything runs locally on your server |

---

## 🏗️ Architecture

### Self-Contained Stack (Phase 1)

Everything runs inside a single Docker container managed by Nextcloud's AppAPI:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docling KB ExApp                            │
│                    (Docker Container)                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    FastAPI Application                      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  📄 Docling          - Document Processing                  ││
│  │  🗄️ ChromaDB         - Vector Database                      ││
│  │  🧮 Sentence-BERT    - Embeddings                           ││
│  │  🤖 Embedded LLM     - Local AI (no external service!)     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  🎯 FULLY SELF-CONTAINED - No Ollama or external LLM needed!   │
└─────────────────────────────────────────────────────────────────┘
```

### 🤖 Embedded LLM Models (Built-in)

| Model | Size | Quality | Best For |
|-------|------|---------|----------|
| `qwen2-0.5b` | 350MB | Good | Default, fast responses |
| `tinyllama-1.1b` | 670MB | Better | More detailed answers |
| `phi-3-mini` | 2.3GB | Best | Complex questions |
| `smollm-360m` | 380MB | Basic | Minimal resources |

### Backend Abstraction Layer

The app uses an abstraction layer that supports two backends:

```python
# Phase 1: Local Backend (default)
BACKEND_MODE=local

# Phase 2: Cloud Backend (future SaaS)
BACKEND_MODE=cloud
DOCLING_CLOUD_URL=https://api.your-saas.com
DOCLING_CLOUD_API_KEY=your-api-key
```

---

## 📦 Supported Formats

### Input
| Format | Extensions |
|--------|------------|
| Documents | `.pdf`, `.docx`, `.pptx`, `.xlsx` |
| Web | `.html`, `.htm` |
| Text | `.txt`, `.md` |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff` |

### Extraction
- ✅ Text content with structure
- ✅ Tables (preserves rows/columns)
- ✅ Headings and hierarchy
- ✅ Code blocks
- ✅ Mathematical formulas
- ✅ Image OCR

---

## 🚀 Quick Start

### Prerequisites

1. **Nextcloud 28+** with **AppAPI** installed
2. **Docker** configured as AppAPI deploy daemon
3. **4GB+ RAM** recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/docling-nextcloud.git
cd docling-nextcloud

# Start everything with Docker Compose
docker compose -f docker-compose.dev.yml up -d
```

This starts:
- 🌐 Nextcloud on `http://localhost:8080`
- 📚 Docling KB on `http://localhost:9000`
- 🤖 **No external LLM needed!** (embedded model downloads automatically)

### Register with Nextcloud

```bash
sudo -u www-data php occ app_api:app:register docling \
  --daemon-config-name docker_local \
  --force-scopes

sudo -u www-data php occ app:enable docling
```

---

## 💬 Usage Examples

### Chat with Your Documents

```
You: What were the total sales in Q4?

AI: Based on the Q4 Financial Report, total sales were $2.4M, 
    representing a 15% increase from Q3. 
    
    The breakdown by region:
    - North America: $1.2M
    - Europe: $800K
    - Asia Pacific: $400K
    
    [Source: Q4_Financial_Report.pdf - 92% relevance]
```

### Semantic Search

```
Query: "budget allocation for marketing"

Results:
1. Annual_Budget_2024.xlsx (94% match)
   "Marketing department allocated $500K..."

2. Board_Meeting_Notes.pdf (87% match)
   "Discussion on marketing spend increase..."
```

### Extract Structured Data

```
Query: "Show all tables from financial reports"

Results:
- Q4_Report.pdf: 3 tables (Revenue, Expenses, Projections)
- Budget_2024.xlsx: 12 sheets extracted
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen2-0.5b` | Embedded LLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `DISABLE_LLM` | `false` | Set to `true` to disable chat |
| `BACKEND_MODE` | `local` | `local` or `cloud` |

### Embedded LLM Models (No External Service!)

```yaml
# Fast & small (default)
LLM_MODEL: qwen2-0.5b

# Better quality
LLM_MODEL: tinyllama-1.1b

# Best quality (needs more RAM)
LLM_MODEL: phi-3-mini

# Disable chat features entirely
DISABLE_LLM: true
```

### Optional: External LLM (Advanced Users)

If you prefer to use an external LLM service:

```yaml
# Ollama (if you have it running)
LLM_PROVIDER: ollama
LLM_BASE_URL: http://localhost:11434/v1
LLM_EXTERNAL_MODEL: llama3.2

# OpenAI
LLM_PROVIDER: openai
LLM_API_KEY: sk-your-key
LLM_EXTERNAL_MODEL: gpt-4o-mini
```

---

## 📊 Resource Requirements

### Phase 1 (Self-Contained)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| CPU | 4 cores | 8+ cores |
| Disk | 5 GB | 20+ GB |
| GPU | Optional | NVIDIA for faster LLM |

### Phase 2 (With Cloud Backend)

| Requirement | Minimum | 
|-------------|---------|
| RAM | 1 GB |
| CPU | 2 cores |
| Disk | 1 GB |

---

## 🗺️ Roadmap

### ✅ Phase 1: Self-Contained (Current)
- [x] Docling document processing
- [x] ChromaDB vector storage
- [x] Semantic search
- [x] Chat with documents (RAG)
- [x] Table extraction
- [x] Web UI
- [x] File action integration
- [ ] Nextcloud Assistant integration
- [ ] Auto-processing file watcher

### 🔮 Phase 2: Cloud Backend (Future)
- [ ] Cloud API service
- [ ] Multi-tenant architecture
- [ ] Usage-based billing
- [ ] Enterprise SSO
- [ ] Managed infrastructure
- [ ] SLA guarantees

### 🚀 Phase 3: Advanced Features
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Workflow automation
- [ ] API for third-party apps
- [ ] Mobile app integration

---

## 🔒 Privacy & Security

- ✅ **No external API calls** (Phase 1)
- ✅ **Documents stay on your server**
- ✅ **Containerized** - Isolated in Docker
- ✅ **AppAPI Authentication** - Secure communication
- ✅ **No telemetry** - We don't collect data
- ✅ **Open source** - Audit the code yourself

---

## 💖 Support the Project

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love">
</p>

**Docling KB is free and open-source**, but building and maintaining quality software takes time and resources. If this project helps you or your organization, please consider supporting its development!

### Why Support?

- 🚀 **Accelerate development** of new features
- 🐛 **Faster bug fixes** and security updates
- 📚 **Better documentation** and tutorials
- 🌍 **Multi-language support**
- 💬 **Priority support** for sponsors

### How to Support

| Method | Link |
|--------|------|
| ⭐ **Star on GitHub** | Give us a star to show your support! |
| 🐛 **Report Issues** | Help us improve by reporting bugs |
| 💻 **Contribute Code** | PRs are always welcome |
| ☕ **Buy Me a Coffee** | [buymeacoffee.com/ihsanmokhlis](https://buymeacoffee.com/ihsanmokhlis) |
| 💝 **GitHub Sponsors** | [github.com/sponsors/ihsanmokhlis](https://github.com/sponsors/ihsanmokhlis) |
| 🏢 **Enterprise Support** | Contact for commercial licensing |

<p align="center">
  <a href="https://buymeacoffee.com/ihsanmokhlis">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
  </a>
  <a href="https://github.com/sponsors/ihsanmokhlis">
    <img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=github-sponsors&logoColor=white" alt="GitHub Sponsors">
  </a>
</p>

> *"If Docling KB saves you time or helps your workflow, a small donation helps keep the project alive and growing. Every contribution, no matter how small, is deeply appreciated!"*
>
> — **Ihsan Mokhlis**, Creator

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

**CC BY-NC-SA 4.0** - [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

| ✅ You CAN | ❌ You CANNOT |
|-----------|--------------|
| Use for personal projects | Use commercially |
| Modify and adapt | Remove attribution |
| Share with others | Use different license for adaptations |
| Use for education/research | Sublicense |

📧 **Need commercial license?** Contact [Ihsan Mokhlis](https://github.com/ihsanmokhlis)

## 🙏 Acknowledgments

- [Docling](https://github.com/DS4SD/docling) - Document processing
- [ChromaDB](https://www.trychroma.com/) - Vector database  
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Embedded LLM runtime
- [Nextcloud AppAPI](https://github.com/cloud-py-api/app_api) - ExApp framework
- [HuggingFace](https://huggingface.co/) - Model hosting

---

<p align="center">
  <strong>Your documents. Your AI. Your server.</strong><br><br>
  Phase 1: Everything local, no compromises.<br>
  Phase 2: Optional cloud, same privacy-first approach.<br><br>
  <sub>Created with ❤️ by <strong>Ihsan Mokhlis</strong></sub><br>
  <sub>
    <a href="https://github.com/sponsors/ihsanmokhlis">💖 Sponsor</a> · 
    <a href="https://buymeacoffee.com/ihsanmokhlis">☕ Buy me a coffee</a>
  </sub>
</p>
