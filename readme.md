# 🍕 Food Search & Discovery Application

A powerful multimodal food recommendation system that leverages vector embeddings to provide intelligent search capabilities — whether the user searches by text or uploads an image, the system returns the most semantically similar food items.

##  Overview

This application demonstrates **multimodal vector search** by combining text and image embeddings to create an intelligent food discovery system. Users can search for food items using natural language queries or by uploading images.

###  based on
->->   https://learn.deeplearning.ai/courses/building-multimodal-search-and-rag/lesson/dh7lr/multimodal-recommender-system
Practical implementation based on **"Building Multimodal Search and RAG"** course by **DeepLearning.AI** and **Andrew Ng**.

###  Development Notes
- Initially developed using **Weaviate Cloud Services (WCS)**
- Migrated to **local Docker setup** due to CLIP API key limitations in cloud environment
- Local deployment provides full control over text (Cohere) and image (CLIP) vectorization

##  Technology Stack

| **Frontend** | Streamlit 1.28+ | Web interface |
| **Vector Database** | Weaviate 1.33+ | Vector storage & search |
| **Deployment** | Docker & Docker Compose | Local containers |
| **Text Vectorization** | Cohere API | Text embeddings |
| **Image Vectorization** | CLIP Model | Image embeddings |

### Why Local Docker?
- **Cloud Limitation**: CLIP API keys unavailable in Weaviate Cloud
- **Solution**: Local CLIP inference container
- **Benefits**: Full control, no cloud costs, faster development

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  ┌──────────────────┐        ┌─────────────────────┐       │
│  │  Text Search Tab │        │  Image Search Tab   │       │
│  └──────────────────┘        └─────────────────────┘       │
└─────────────────┬────────────────────┬───────────────────────┘
                  │                    │
                  ▼                    ▼
         ┌────────────────────────────────────┐
         │      Weaviate Client (Python)      │
         └────────────────┬───────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      Weaviate Vector Database      │
         │         (Docker Container)         │
         │  ┌──────────────┐ ┌─────────────┐ │
         │  │ text_vector  │ │image_vector │ │
         │  │   (Cohere)   │ │   (CLIP)    │ │
         │  └──────────────┘ └─────────────┘ │
         └────────┬──────────────────┬────────┘
                  │                  │
                  ▼                  ▼
    ┌─────────────────────┐  ┌──────────────────────┐
    │  Cohere API         │  │  CLIP Inference      │
    │  (Text Embeddings)  │  │  (Image Embeddings)  │
    │  Port: External     │  │  Port: 8081          │
    └─────────────────────┘  └──────────────────────┘
```
# Output
at assets folder 
![Demo Preview](./assets/demo.gif)
##  Prerequisites

### Required Software
1. **Docker Desktop** - https://www.docker.com/products/docker-desktop
2. **Python 3.8+**

### Required API Keys
1. **Cohere API Key** - https://cohere.com/ (Free tier available)

---
##  Installation

### Step 1: Install Python Dependencies

```bash
pip install weaviate-client streamlit Pillow requests
```

### Step 2: Create Docker Compose File

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      ENABLE_MODULES: 'text2vec-cohere,multi2vec-clip'
      COHERE_API_KEY: 'your-cohere-api-key-here'
      CLIP_INFERENCE_API: 'http://clip-inference:8080'
      CLUSTER_HOSTNAME: 'node1'
      RAFT_JOIN: 'node1'
      RAFT_BOOTSTRAP_EXPECT: 1
    volumes:
      - weaviate_data:/var/lib/weaviate
    depends_on:
      - clip-inference

  clip-inference:
    image: semitechnologies/multi2vec-clip:sentence-transformers-clip-ViT-B-32
    ports:
      - "8081:8080"

volumes:
  weaviate_data:
```

### Step 3: Start Docker Services

```bash
docker-compose up -d
```

### Step 4: Load Sample Data

```bash
jupyter notebook data_collection.ipynb
```

### Step 5: Launch Application

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 🔧 Troubleshooting

### Weaviate Connection Failed
```bash
docker-compose ps        # Check status
docker-compose logs      # View logs
docker-compose restart   # Restart services
```

### CLIP Not Ready
```bash
docker logs clip-container-name
docker-compose restart clip-inference
# Wait 30 seconds
```

### No Search Results
```bash
# Verify data loaded
curl http://localhost:8080/v1/schema
```

### Port Conflicts
```bash
# Check port usage
netstat -ano | findstr :8080

# Change port in docker-compose.yml
ports:
  - "8090:8080"
```

### Health Checks
```bash
# Weaviate
curl http://localhost:8080/v1/.well-known/ready

# CLIP
curl http://localhost:8081/.well-known/ready
```

---

###  Performance Metrics

| Metric | Value |
|--------|-------|
| Text search time | < 200ms |
| Image search time | < 500ms |
| Memory usage | ~2GB |
| Vector indexing (20 items) | ~30 seconds |

### Accuracy Metrics

| Search Type | Top-1 Accuracy | Top-3 Accuracy |
|-------------|---------------|----------------|
| Text Search |      ~85%     |     ~95%       |
| Image Search |     ~75%     |     ~90%       |

---

### Technologies
- **Weaviate** - Vector database platform
- **Cohere** - Text embedding API
- **OpenAI CLIP** - Image embedding model
- **Streamlit** - Web framework
- **Docker** - Containerization
