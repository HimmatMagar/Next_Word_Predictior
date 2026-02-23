from src.nextWordPrediction.utils import *
from src.nextWordPrediction.constants import __CONFIG__, __PARAMS__
from src.nextWordPrediction.entity import DataIngestionConfig, DataTransformationConfig, ModelBuildingConfig


class ConfigManager:
      def __init__(self, config = __CONFIG__, params = __PARAMS__):
            self.config = load_yaml_file(config)
            self.params = load_yaml_file(params)

            create_directory([self.config.ariticate_root])
      

      def get_data_ingestion_config(self) -> DataIngestionConfig:
            """
                  Loads the data ingestion configuration from the YAML file and
                  prepares the required directories.

                  Returns:
                        DataIngestionConfig: Configuration object containing paths
                        and source URL for data ingestion.
            """
            config = self.config.data_ingestion

            create_directory([config.root_dir])
            data_ingestion_config = DataIngestionConfig(
                  root_dir=config.root_dir,
                  source_url=config.source_url,
                  ziped_data=config.ziped_data_path,
                  unzip_data=config.unziped_file
            )
            return data_ingestion_config


      def get_data_transform_config(self) -> DataTransformationConfig:
            """
                  Loads the data transformation configuration from the YAML file and
                  prepares the required directories.

                  Returns:
                        DataTransformationConfig: Configuration object containing paths
                        and Data path for data transformation.
            """
            config = self.config.data_transformation
            params = self.params.parameter

            create_directory([config.root_dir])
            data_transformation_config = DataTransformationConfig(
                  root_dir=config.root_dir,
                  vocab_size=params.max_vocab,
                  seq_len = params.seq_len,
                  data_file_path=config.data_file_path
            )
            return data_transformation_config


      def get_model_building_config(self) -> ModelBuildingConfig:
            """
                  Loads the model building configuration from the YAML file and
                  prepares the required directories.

                  Returns:
                        ModelBuildingConfig: Configuration object containing all the
                        parameter with data path for model train.
            """
            config = self.config.model_building
            params = self.params.parameter

            create_directory([config.root_dir])

            model_building_config = ModelBuildingConfig(
                  root_dir=config.root_dir,
                  input_file=config.input_file_path,
                  output_file=config.output_file_path,
                  model=config.model,
                  vocab_size=params.max_vocab,
                  seq_length=params.seq_len,
                  lstm_unit=params.lstm_unit,
                  embedding_units=params.embedding_units,
                  optimizer=params.optimizer,
                  epochs= params.epochs,
                  batch_size=params.batch_size,
                  spliting=params.spliting,
                  activation=params.activation
            )

            return model_building_config