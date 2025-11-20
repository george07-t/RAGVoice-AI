# Voice-Enabled RAG AI Agent

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-green.svg)](https://livekit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple.svg)](https://openai.com/)

A sophisticated voice-enabled AI agent combining **Retrieval-Augmented Generation (RAG)**, **multi-agent orchestration**, and **real-time function calling** for intelligent document Q&A and dynamic information retrieval.

## 🎯 Features

- **📚 RAG System**: Query 5 comprehensive documents (80+ pages) on AI, Climate, History, Blockchain, and Health
- **🎤 Voice Interface**: Natural conversation with Deepgram STT and Cartesia TTS via LiveKit
- **🤖 Multi-Agent System**: LangGraph orchestrator with specialized agents (RAG, Tool, General)
- **🌦️ Dynamic Tools**: Real-time weather, currency exchange, and timezone information
- **⚡ High Performance**: <2-5s end-to-end latency with comprehensive metrics logging
- **🐳 Production Ready**: Docker deployment with environment variable configuration

## 🏗️ Architecture

<img width="455" height="357" alt="image" src="https://github.com/user-attachments/assets/ce157892-47a3-42f3-8308-6b81a7f1047a" />


## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- LiveKit account (for voice features)
- Deepgram API key (for STT)
- Cartesia API key (for TTS)
- WeatherAPI key (for weather tools)

### Installation

```bash
# Clone repository
git clone https://github.com/george07-t/RAGVoice-AI.git
cd soffotech-rag-agent

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env.local` and fill in your API keys:

```env
OPENAI_API_KEY="your-openai-key"
WEATHER_API_KEY="your-weather-key"
LIVEKIT_URL="your-livekit-url"
LIVEKIT_API_KEY="your-livekit-key"
LIVEKIT_API_SECRET="your-livekit-secret"
DEEPGRAM_API_KEY="your-deepgram-key"
CARTESIA_API_KEY="your-cartesia-key"
```

### Initialize RAG System

```bash
python src/rag_system.py
```

This will:
- Load 5 documents from `documents/`
- Create 112 text chunks
- Generate OpenAI embeddings
- Build FAISS vector store

### Run the Agent

**Console Mode** (Terminal Voice Chat):
```bash
python src/agent.py console
```

**Development Mode**:
```bash
python src/agent.py dev
```

**Production Mode**:
```bash
python src/agent.py start
```

## 📝 Usage Examples

### Document-Based Queries (RAG)
- "What is deep learning?"
- "Explain climate change causes"
- "Tell me about World War II"
- "What is blockchain technology?"
- "How much sleep do I need?"

### Real-Time Information (Tools)
- "What's the weather in Dhaka?"
- "Give me a 3-day forecast for New York"
- "What's the USD to BDT exchange rate?"
- "What time is it in Tokyo?"

### General Conversation
- "Hello! How are you?"
- "What can you help me with?"

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t rag-voice-agent .
```

### Run Container

```bash
docker run -d \
  -e OPENAI_API_KEY="your-key" \
  -e LIVEKIT_URL="your-url" \
  -e LIVEKIT_API_KEY="your-key" \
  -e LIVEKIT_API_SECRET="your-secret" \
  -e DEEPGRAM_API_KEY="your-key" \
  -e CARTESIA_API_KEY="your-key" \
  -e WEATHER_API_KEY="your-key" \
  -p 8080:8080 \
  rag-voice-agent
```

### Deploy to LiveKit Cloud

```bash
lk deploy create
```

## 📊 Performance Metrics

| Component | Latency |
|-----------|---------|
| RAG Retrieval (3 docs) | 0.5-1.5s |
| Orchestrator Routing | <0.1s |
| LLM Response | 1-3s |
| Weather API | 0.2-0.5s |
| **Total End-to-End** | **2-5s** |

## 🛠️ Technology Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: FAISS (CPU)
- **Orchestration**: LangChain + LangGraph
- **Voice**: LiveKit (Deepgram STT, Cartesia TTS)
- **APIs**: WeatherAPI.com, ExchangeRate-API.com

## 📚 Project Structure

```
soffotech-rag-agent/
├── src/
│   ├── agent.py                    # LiveKit voice agent entry point
│   ├── multi_agent_rag.py          # LangGraph multi-agent orchestrator
│   ├── rag_system.py               # RAG pipeline (FAISS + embeddings)
│   ├── dynamic_tools.py            # Real-time API tools
│   └── __init__.py
├── documents/                       # Knowledge base (5 docs, 80+ pages)
│   ├── artificial_intelligence_guide.txt
│   ├── climate_change_report.txt
│   ├── world_history_modern.txt
│   ├── blockchain_technology.txt
│   └── health_wellness_guide.txt
├── vector_store/                    # FAISS index (auto-generated)
├── tests/
│   └── test_agent.py
├── Dockerfile
├── requirements.txt
├── livekit.toml
├── .env.example                    # Environment template
├── PROJECT_README.md               # Detailed technical documentation
├── Soffotech_RAG_Voice_Agent_Demo.ipynb  # Google Colab demo
├── LICENSE
└── README.md                       # This file
```

## 🧪 Testing

### Test RAG System
```bash
python -c "from src.rag_system import get_rag_system; rag = get_rag_system(); print(rag.retrieve('What is deep learning?'))"
```

### Test Weather Tool
```bash
python -c "from src.dynamic_tools import get_current_weather; print(get_current_weather.invoke({'city': 'Dhaka'}))"
```

### Test Multi-Agent System
```bash
python src/multi_agent_rag.py
```

### Run Unit Tests
```bash
pytest tests/
```

## 📖 Documentation

- **[PROJECT_README.md](PROJECT_README.md)**: Comprehensive technical documentation with architecture details, scaling strategies, and troubleshooting
- **[Soffotech_RAG_Voice_Agent_Demo.ipynb](Soffotech_RAG_Voice_Agent_Demo.ipynb)**: Interactive Google Colab notebook with step-by-step demo

## 🔒 Security

- API keys stored in `.env.local` (never committed)
- Docker runs as non-privileged user
- Input sanitization for all queries
- HTTPS/TLS for LiveKit connections

## 🤝 Contributing

This is an assessment project. For questions or suggestions, please open an issue.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 👤 Author

**George Tonmoy Roy**  
Soffotech Technical Assessment - November 2025

## 🙏 Acknowledgments

- [LiveKit](https://livekit.io/) - Voice AI infrastructure
- [LangChain](https://langchain.com/) - LLM orchestration framework
- [OpenAI](https://openai.com/) - Language models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [WeatherAPI.com](https://weatherapi.com/) - Weather data provider

---

⭐ **Star this repo if you find it helpful!**
