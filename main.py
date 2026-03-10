from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import json
import feedparser
from urllib.parse import quote

from utils import clean_text

from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from groq import Groq


# Load ML model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Groq API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Fake News Detection API (RAG + LLM)")

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


# -------- GOOGLE NEWS SEARCH --------
def search_recent_news(query):

    url = f"https://news.google.com/rss/search?q={quote(query)}"

    feed = feedparser.parse(url)

    articles = []

    for entry in feed.entries[:5]:
        articles.append({
            "title": entry.title,
            "source": entry.source.title if "source" in entry else "Unknown"
        })

    return articles


# -------- LLM VERIFICATION --------
def verify_with_llm(news_text, articles):

    context = "\n".join(
        [f"{a['title']} (Source: {a['source']})" for a in articles]
    )

    prompt = f"""
You are a professional fact-checking AI.

News to verify:
{news_text}

Recent news articles:
{context}

Based on the articles above, determine if the news is REAL NEWS or FAKE NEWS.

Respond ONLY in JSON format:

{{
"prediction": "Real News or Fake News",
"explanation": "Short reason using the articles as evidence"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except:
        result = {
            "prediction": "Unknown",
            "explanation": content
        }

    return result


@app.get("/")
def home():
    return {"message": "Fake News Detection API (RAG Enabled)"}


@app.post("/predict")
def predict_news(news: NewsInput):

    # ---- ML prediction ----
    cleaned = clean_text(news.text)
    vector = vectorizer.transform([cleaned])

    ml_prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()

    if ml_prediction == 1:
        ml_result = "Real News"
    else:
        ml_result = "Fake News"

    # ---- Search recent news ----
    articles = search_recent_news(news.text)

    # ---- LLM verification ----
    llm_result = verify_with_llm(news.text, articles)

    return {
        "prediction": llm_result["prediction"],
        "explanation": llm_result["explanation"],
        "ml_prediction": ml_result,
        "ml_confidence": round(float(confidence)*100,2),
        "evidence_articles": articles
    }