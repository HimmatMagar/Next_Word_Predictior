import re
import os
import pickle
from nltk import word_tokenize
from collections import Counter
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
      
      
      def text_to_indices(self, sentence, vocab):
            numeric_sentence = []

            for token in sentence:
                  if token in vocab:
                        numeric_sentence.append(vocab[token])
                  else:
                        numeric_sentence.append(vocab['<UNK>'])
            return numeric_sentence
      

      def build_vocab(self, text):
            tokens = word_tokenize(text)
            vocab = {
                  "<UNK>":0
            }

            for token in Counter(tokens).keys():
                  if token not in vocab:
                        vocab[token] = len(vocab)
            return vocab
      

      def build_sequence(self, indices, seq_len):
            x = []
            y = []

            for i in range(len(indices) - seq_len):
                  input_seq = indices[i : i + seq_len]
                  target = indices[i + seq_len]

                  x.append(input_seq)
                  y.append(target)
            return x, y


      def final_output(self):
            try:
                  data = self.load_data()
                  text = self.clean_text(data)
                  vocab = self.build_vocab(text)

                  numeric_sentence = self.text_to_indices(word_tokenize(text), vocab)

                  x, y = self.build_sequence(numeric_sentence, self.config.seq_len)
                  with open(os.path.join(self.config.root_dir, "input.pkl"), "wb") as f:
                        pickle.dump(x, f)

                  with open(os.path.join(self.config.root_dir, "output.pkl"), "wb") as f:
                        pickle.dump(y, f)

                  with open(os.path.join(self.config.root_dir, "tokenizer.pkl"), "wb") as f:
                        pickle.dump(vocab, f)

                  logger.info("Data Transormation is completed")
            except Exception as e:
                  raise e