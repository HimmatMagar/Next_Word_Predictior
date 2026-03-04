import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from pydantic import Field, BaseModel
from src.nextWordPrediction.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Input(BaseModel):
      word: Annotated[str, Field(..., description="Give a word or sentence to predict next word")]


@app.get("/")
async def home():
      return FileResponse("templates/index.html")



@app.get("/training")
async def training():
      os.system("main.py")
      return {"message": "Training endpoint is under development. Please check back later."}


prediction_pipeline = PredictionPipeline()
@app.post("/predict")
async def prediction(UserInput: Input):
      text = (UserInput.word or "").strip()
      if not text:
            raise HTTPException(status_code=422, detail="'word' must be a non-empty string")

      try:
            data = pd.DataFrame([{"text": text}])
            input_seq = prediction_pipeline.transform_input(data=data)
            pred = prediction_pipeline.predict_next_word(input_seq)

            # make prediction JSON-serializable
            try:
                  import numpy as _np
                  if isinstance(pred, _np.ndarray):
                        serial_pred = pred.tolist()
                  else:
                        serial_pred = pred
            except Exception:
                  serial_pred = pred

            return {
                  "status": "success",
                  "input": text,
                  "prediction": serial_pred
            }
      except HTTPException:
            raise
      except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

