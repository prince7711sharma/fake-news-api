from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from utils import clean_text

# CORS IMPORT
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("fake_news_model.pkl")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(
    title="Fake News Detection API",
    version="1.0"
)

# CORS SETTINGS
origins = ["*"]   # allow all domains

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class NewsInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake News Detection API Running"}


@app.post("/predict")
def predict_news(news: NewsInput):

    cleaned_text = clean_text(news.text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return {
        "prediction": result,
        "confidence": float(confidence)
    }