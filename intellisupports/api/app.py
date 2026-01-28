# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from intellisupports.src.inference import IntelliSupportPredictor

app = FastAPI(
    title="IntelliSupport API",
    description="AI-powered support ticket classification and retrieval system",
    version="1.0.0"
)

# Load model ONCE at startup
predictor = IntelliSupportPredictor()


class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    


class SimilarTicket(BaseModel):
    category: str
    score: float
    original_index: int


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    similar_tickets: List[SimilarTicket]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = predictor.predict(
        text=request.text,
        top_k=request.top_k,
        
    )
    return result

@app.get("/")
def root():
    return {"status": "IntelliSupport API is running"}
