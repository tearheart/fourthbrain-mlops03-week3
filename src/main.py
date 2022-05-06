from fastapi import FastAPI

import logging
from logging.config import dictConfig
from log_config import log_config # this is your local file

from pydantic import BaseModel

from transformers import pipeline


dictConfig(log_config)
logger = logging.getLogger("my-project-logger")

app = FastAPI()

sentiment_model = pipeline("sentiment-analysis")

class PredictionRequest(BaseModel):
  query_string: str


@app.get("/health")
def health():
    logger.info("It worked")
    return {"message": "Hello World"}


@app.post("/my-endpoint")
def my_endpoint(request: PredictionRequest):
    sentiment_query_sentence = request.query_string
    sentiment = sentiment_model(sentiment_query_sentence)
    return f"Sentiment test: {sentiment_query_sentence} == {sentiment}"
