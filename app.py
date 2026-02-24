import pandas as pd
from fastapi import FastAPI
from typing import Annotated
from pydantic import Field, BaseModel
from src.nextWordPrediction.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI()

class Input(BaseModel):
      word: Annotated[str, Field(..., description="Give a word or sentence to predict next word")]

@app.get("/")
def home():
      return {"message": "Hello"}

prediction_pipeline = PredictionPipeline()
@app.post("/predict")
def prediction(UserInput: Input):
      data = pd.DataFrame([{
            "text": UserInput.word
      }])

      input_seq = prediction_pipeline.transform_input(data=data)

      pred = prediction_pipeline.predict_next_word(input_seq)
      return {
            "prediction": pred
      }

