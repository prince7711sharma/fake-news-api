from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from utils import clean_text

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# CORS
from fastapi.middleware.cors import CORSMiddleware

# GROQ
from groq import Groq


# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Get API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Hybrid Fake News Detection API")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsInput(BaseModel):
    text: str


CONFIDENCE_THRESHOLD = 0.75


def verify_with_llm(news_text):

    prompt = f"""
You are a fact-checking AI.

Analyze the following news text and determine whether it is Real News or Fake News.

News:
{news_text}

Respond with only:
Real News
or
Fake News
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


@app.get("/")
def home():
    return {"message": "Hybrid Fake News Detection API running"}


@app.post("/predict")
def predict_news(news: NewsInput):

    cleaned = clean_text(news.text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    if prediction == 1:
        ml_result = "Real News"
    else:
        ml_result = "Fake News"

    if confidence < CONFIDENCE_THRESHOLD:

        llm_result = verify_with_llm(news.text)

        return {
            "prediction": llm_result,
            "ml_prediction": ml_result,
            "confidence": float(confidence),
            "source": "LLM Verification"
        }

    else:

        return {
            "prediction": ml_result,
            "confidence": float(confidence),
            "source": "ML Model"
        }