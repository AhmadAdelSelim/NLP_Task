from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data if needed
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__)

# Load the model
model = joblib.load("sentiment_model.pkl")


# Preprocessing functions
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "<HASHTAG>", text)
    text = re.sub(r"[^\w\s<><HASHTAG>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess_tokens(tokens):
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word.isalnum()]
    tokens = ["<NUM>" if word.isdigit() else word for word in tokens]
    return tokens


lemmatizer = WordNetLemmatizer()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "")
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Preprocess
    cleaned = clean_tweet(tweet)
    tokens = word_tokenize(cleaned)
    processed = preprocess_tokens(tokens)
    lemmatized = [lemmatizer.lemmatize(word) for word in processed]
    text = " ".join(lemmatized)

    # Predict
    pred = model.predict([text])[0]
    sentiment = "positive" if pred == 1 else "negative"

    return jsonify({"sentiment": sentiment, "tweet": tweet})


if __name__ == "__main__":
    app.run(debug=True)
