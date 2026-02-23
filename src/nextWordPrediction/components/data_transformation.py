import re
import os
import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from src.nextWordPrediction import logger
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.nextWordPrediction.config import DataTransformationConfig


class DataTransform:
      def __init__(self, config: DataTransformationConfig):
            self.config = config
      
      def load_data(self):
            """
                  It load the data from the configuration and convert it into pandas data frame
                  
                  Return:
                        It return the pandas dataframe
            """
            df = pd.read_csv(self.config.data_file_path)
            return df


      def clean_text(self, text: str) -> str:
            """
                  Removes punctuation and numbers from text.
                  
                  Args:
                        text (str): Input text
                        
                  Returns:
                        str: Cleaned text with only letters and spaces
            """

            if isinstance(text, str):
                  text = re.sub(r'[^a-zA-Z\s]', '', text)  
                  text = re.sub(r'\s+', ' ', text)         
                  return text.lower().strip()          
            return text


      def cleaning_pipeline(self):
            data = self.load_data()

            cleaned_data = data['quote'].apply(self.clean_text).tolist()

            tokenizer = Tokenizer(num_words=self.config.vocab_size)
            tokenizer.fit_on_texts(cleaned_data)
            sequence = tokenizer.texts_to_sequences(cleaned_data)

            X = []
            y = []

            for seq in sequence:
                  if len(seq) < 2:
                        continue

                  for i in range(1, len(seq)):
                        x_input = seq[:i]
                        y_output = seq[i]
                        X.append(x_input)
                        y.append(y_output)
            
            input_seq = pad_sequences(
                  sequences=X,
                  maxlen=self.config.seq_len,
                  padding='pre'
            )

            y_out = np.array(y)
            y_one_hot = to_categorical(
                  y_out, num_classes=self.config.vocab_size
            )

            with open(os.path.join(self.config.root_dir, "input_seq.pkl"), "wb") as f:
                  pickle.dump(input_seq, f)

            with open(os.path.join(self.config.root_dir, "output.pkl"), "wb") as f:
                  pickle.dump(y_one_hot, f)

            with open(os.path.join(self.config.root_dir, "tokenizer.pkl"), "wb") as f:
                  pickle.dump(tokenizer, f)

            logger.info("Data Transormation is completed")