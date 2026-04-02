from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib


app = FastAPI()

classifier_model = joblib.load("models/classifier.joblib")
transformer_model = SentenceTransformer("models/sentence_transformer.model")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    embedding = transformer_model.encode([request.text])
    predicted_class = classifier_model.predict(embedding)[0]
    return PredictResponse(prediction=str(predicted_class))
