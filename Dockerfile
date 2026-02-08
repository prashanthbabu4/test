FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY lr_sentiment_model.pkl .
COPY tfidf_vectorizer.pkl .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
