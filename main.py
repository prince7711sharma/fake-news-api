from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from utils import clean_text

# Load trained model
model = joblib.load("fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API for detecting Fake or Real news using Machine Learning",
    version="1.0"
)

# Request body format
class NewsInput(BaseModel):
    text: str


# Home route
@app.get("/")
def home():
    return {
        "message": "Fake News Detection API is running"
    }


# Prediction route
@app.post("/predict")
def predict_news(news: NewsInput):

    # Step 1: Clean text
    cleaned_text = clean_text(news.text)

    # Step 2: Convert text to vector
    vector = vectorizer.transform([cleaned_text])

    # Step 3: Predict
    prediction = model.predict(vector)[0]

    # Step 4: Get confidence score
    confidence = model.predict_proba(vector).max()

    # Step 5: Convert prediction to label
    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return {
        "prediction": result,
        "confidence": float(confidence)
    }