# PinPoint AI: Video Retrieval-Augmented Generation System

PinPoint AI is a technical workspace that converts video content into an interactive knowledge base. By combining local transcription, vector embeddings, and Large Language Models (LLMs), the system enables semantic search and automated technical assessment based on video data.



## System Architecture

The application follows a modular architecture to separate concerns between data processing, user authentication, and the search engine:

### 1. File Responsibilities

* **app.py**: The central coordinator and UI router. It manages the Streamlit session state, navigation logic, and handles the high-level coordination between the Chat UI and the Query Engine.
* **video_processor.py**: The data ingestion engine. It manages the Whisper transcription model, thumbnail generation, and the ChromaDB collection lifecycle (creation, updates, and deletion).
* **query_engine.py**: The retrieval and reasoning core. It handles the two-stage search process (semantic search + cross-encoder reranking), context expansion for LLM prompts, and communication with the Gemini API.
* **auth.py**: The security layer. It implements a local SQLite3 database for user management and handles salt-based password hashing using bcrypt.

### 2. Technical Stack

* **Transcription**: OpenAI Whisper (Small model running on local GPU/CPU).
* **Vector Store**: ChromaDB for persistent storage of text embeddings.
* **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2).
* **Reranker**: Cross-Encoder (ms-marco-MiniLM-L-6-v2) for improved precision in search results.
* **LLM Layer**: Google Gemini 1.5 Flash for final response synthesis and quiz generation.



## The Data Pipeline

1. **Ingestion**: Videos are uploaded and stored in user-specific directories. A background thread extracts audio and generates a visual thumbnail.
2. **Indexing**: Whisper converts audio to text segments. These segments are grouped, embedded into 384-dimensional vectors, and stored in ChromaDB alongside temporal metadata.
3. **Retrieval**: When a query is received, the system performs a semantic search. The top candidates are then passed through a Cross-Encoder reranker to verify relevance.
4. **Augmentation**: The most relevant segments are expanded with surrounding context (neighboring transcript lines) and injected into the LLM prompt as "ground truth".

## Project Structure

```text
PROJECT/
├── .streamlit/          # Configuration and secrets
├── Database/
│   ├── processing/      # Temporary sync flags for background tasks
│   ├── users/           # Root for all user-specific data
│   │   └── [username]/
│   │       ├── chroma_db/   # Vector embedding storage
│   │       ├── thumbnails/  # Video preview images
│   │       └── videos/      # Local video files
│   └── users.db         # Relational database for credentials
├── app.py               # Main application entry point
├── auth.py              # Authentication logic
├── query_engine.py      # AI search and reasoning engine
├── video_processor.py   # Data processing and indexing engine
└── requirements.txt     # Project dependencies
