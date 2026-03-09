import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text: str):

    # convert to lowercase
    text = text.lower()

    # remove numbers and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split words
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # stemming
    words = [stemmer.stem(w) for w in words]

    # join words
    cleaned_text = " ".join(words)

    return cleaned_text