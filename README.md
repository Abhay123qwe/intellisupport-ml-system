# IntelliSupport ML System

IntelliSupport is a production-ready end-to-end machine learning system for automated support ticket classification and semantic retrieval. It leverages modern NLP and vector search techniques to categorize incoming support tickets and find similar past tickets, enabling faster response and resolution.

This project includes:

  - A classification model to assign categories to support tickets.

  - A semantic search component to retrieve similar historic tickets.

  - A REST API built with FastAPI for serving predictions.

  - Pre-trained artifacts ready to use out of the box.

## 📌 Features
- 🧠 Ticket Classification – Predicts the category of a support query.

- 🔍 Semantic Ticket Retrieval – Finds similar tickets using FAISS embeddings.

- ⚡ FastAPI REST API – Simple endpoints to integrate with helpdesk systems.

- 📦 Pre-built artifacts (model weights, embeddings).

## 📁 Project Structure
intellisupport-ml-system/

├── intellisupports/

│   ├── api/                  # FastAPI application

│   │   └── app.py

│   ├── artifacts/            # Pre-trained models & embeddings

│   ├── data/                 # Raw datasets (train/test)

│   ├── src/                  # Core ML code

│   │    ├── dataset.py

│   │    ├── embeddings.py

│   │    ├── train.py

│   │    ├── inference.py

│   │    └── retrieve.py        # Raw datasets (train/test)

│   ├── requirement.txt

│ 

└── README.md

## 🧠 How It Works
### 1. Model Training

  - Prepare labeled support ticket data.

  - Train a classification model using PyTorch.

  - Generate dense embeddings for text and categories.

### 2. Semantic Indexing

  - Use FAISS to index support ticket vectors.

  - Enables fast similarity search (approx. nearest neighbors).

### 3. Inference API

  - Load the classification model and embedding index on startup.

  - Expose /predict endpoint for classification + retrieval in one call.

## 🛠️ Installation & Setup

### 1. Clone the repository

  - git clone https://github.com/Abhay123qwe/intellisupport-ml-system.git
  - cd intellisupport-ml-system


### 2. Setup Python environment

  - python3 -m venv venv
  - source venv/bin/activate
  - pip install -r requirement.txt


### 3. Run API

  - uvicorn intellisupports.api.app:app --reload

#### Server will be available at http://localhost:8000
#### Health check: GET /health

## 📌 API Endpoints
| Endpoint   | Method | Description                       |
| ---------- | ------ | --------------------------------- |
| `/health`  | GET    | Health check service              |
| `/predict` | POST   | Predict ticket + retrieve similar |

#### Prediction Request
{

  "text": "<ticket text>",
  
  "top_k": 5
  
}
#### Prediction Response
{

  "predicted_category": "<label>",
  
  "confidence": 0.00,
  
  "similar_tickets": [
  
    {
    
      "category": "<label>",
      
      "score": 0.00,
      
      "original_index": 0
      
    }
    
  ]
  
}

## 🧩 Dependencies

Core technologies used:

  - PyTorch – Deep learning model training

  - FAISS – Fast similarity search

  - FastAPI – Web API framework

  - Pydantic – Data validation

  - See full requirements in requirement.txt.

### 📈 Future Improvements

  - Add multi-label classification support.

  - Integrate training pipeline with CI/CD workflows.

  - Provide Docker support & deployment scripts.

  - Implement UI dashboard for analytics.

## 🙌 Contribution
Contributions, issues, and feature requests are welcome!
Feel free to open issues or send pull requests.
