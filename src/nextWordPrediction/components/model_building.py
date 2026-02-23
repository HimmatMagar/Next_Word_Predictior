import os
from pathlib import Path
from src.nextWordPrediction import logger
from src.nextWordPrediction.utils import *
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM
from src.nextWordPrediction.entity import ModelBuildingConfig


class ModelBuilding:
      def __init__(self, config: ModelBuildingConfig):
            self.config = config
            
      
      def build_model(self) -> None:
            input_seq = load_file(Path(self.config.input_file))
            output = load_file(Path(self.config.output_file))

            model = Sequential([
                  Embedding(
                        input_dim=self.config.vocab_size,
                        output_dim=self.config.embedding_units,
                        input_length=self.config.seq_length
                  ),
                  LSTM(
                        units=self.config.lstm_unit
                  ),
                  Dense(
                        units=self.config.vocab_size,
                        activation=self.config.activation
                  )
            ])

            model.compile(
                  loss='categorical_crossentropy',
                  optimizer=self.config.optimizer,
                  metrics=['accuracy']
            )

            model.fit(
                  input_seq,
                  output,
                  epochs=self.config.epochs,
                  batch_size=self.config.batch_size,
                  validation_split=self.config.spliting
            )

            print("Model Summary \n", model.summary())

            model_path = os.path.join(self.config.root_dir, self.config.model)
            model.save(model_path)
            
            logger.info(f"Model building successfully in: {model_path}")
