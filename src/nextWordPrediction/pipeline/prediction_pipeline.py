import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class PredictionPipeline:
      def __init__(self):
            self.model = load_model(Path("artificate/build_model/model.h5"))
            with open(Path("artificate/data_transformation/tokenizer.pkl"), "rb") as f:
                  self.tokenizer = pickle.load(f)


      def transform_input(self, data):
            seq = self.tokenizer.texts_to_sequences(data)
            input_seq = pad_sequences(
                  sequences=seq,
                  maxlen=252,
                  padding='pre'
            )
            return input_seq


      def predict_next_word(self, data):
            prediction_index = np.argmax(self.model.predict(data))

            for word, index in self.tokenizer.word_index.items():
                  if index == prediction_index:
                        return word
            
            return None