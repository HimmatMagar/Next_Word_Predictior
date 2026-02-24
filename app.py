import pandas as pd
from typing import Annotated
from pydantic import Field, BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from src.nextWordPrediction.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI()

templates = Jinja2Templates(directory="templates")


class Input(BaseModel):
      word: Annotated[str, Field(..., description="Give a word or sentence to predict next word")]


@app.get("/")
def home(request: Request):
      return templates.TemplateResponse("index.html", {"request": request})

prediction_pipeline = PredictionPipeline()
@app.post("/predict")
def prediction(request: Request, word: str = Form(...)):

      user_input = Input(word=word)
      data = pd.DataFrame([{
            "text": user_input.word
      }])

      input_seq = prediction_pipeline.transform_input(data=data)

      pred = prediction_pipeline.predict_next_word(input_seq)
      return templates.TemplateResponse(
            "index.html",
            {
                  "request": request,
                  "prediction": pred
            }
      )

