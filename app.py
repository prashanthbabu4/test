from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("lr_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Map numeric label to sentiment
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    
    # Transform text using TF-IDF
    text_tfidf = vectorizer.transform([text])
    
    # Predict
    pred = model.predict(text_tfidf)[0]
    
    return jsonify({"sentiment": id2label[pred]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
