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
            data['text'] = data['text'].str.lower()
            seq = self.tokenizer.texts_to_sequences(data['text'].tolist())
            input_seq = pad_sequences(
                  sequences=seq,
                  maxlen=252,
                  padding='pre'
            )
            return input_seq


      def predict_next_word(self, data):
            predictions = self.model.predict(data)
            # Get probabilities for the last sequence (batch size = 1)
            probs = predictions[0]
            
            # Get indices of top 5 highest probabilities
            top_5_indices = np.argsort(probs)[-5:][::-1]
            
            # Map indices back to words
            top_5_words = []
            for idx in top_5_indices:
                  for word, word_idx in self.tokenizer.word_index.items():
                        if word_idx == idx:
                              top_5_words.append(word)
                              break
            
            return top_5_words if top_5_words else None