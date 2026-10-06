import re
import os
import pickle
from nltk import word_tokenize
from nextWordPrediction import logger
from nextWordPrediction.config import DataTransformationConfig


class DataTransform:
      def __init__(self, config: DataTransformationConfig):
            self.config = config
      
      def load_data(self):
            """
                  It load the data from the configuration
                  
                  Return:
                        It return the text
            """
            try:
                  with open(self.config.data_file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        return text
            except FileNotFoundError as e:
                  raise e


      def clean_text(self, text: str) -> str:
            if isinstance(text, str):
                  text = re.sub(r'http\S+|www\.\S+', '', text)   # remove URLs
                  text = re.sub(r'[^a-zA-Z\s]', '', text)        # remove punctuation & numbers
                  text = re.sub(r'\n+', ' ', text)                # remove newlines → single space
                  text = re.sub(r'\s+', ' ', text)                # collapse multiple spaces
            return text

      
      def split_text(self, text):
            text = self.clean_text(text=text)
            n = len(text)

            train_text = text[:int(0.8 * n)]
            val_text   = text[int(0.8 * n):int(0.9 * n)]
            test_text  = text[int(0.9 * n):]

            for name, data in [("train", train_text), ("val", val_text), ("test", test_text)]:
                  with open(os.path.join(self.config.root_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                        f.write(data)
            return train_text
      
      
      def build_vocab(self):
            vocab = {
                  "<UNK>":0
            }
            text = self.load_data()
            train_text = self.split_text(text)
            for token in word_tokenize(train_text):
                  if token not in vocab:
                        vocab[token] = len(vocab)
            with open(os.path.join(self.config.root_dir, "tokenizer.pkl"), "wb") as f:
                  pickle.dump(vocab, f)
            logger.info("vocabolaries saved successfully")
            return vocab