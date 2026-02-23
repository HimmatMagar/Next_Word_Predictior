from src.nextWordPrediction.utils import *
from src.nextWordPrediction.entity import DataIngestionConfig
from src.nextWordPrediction.constants import __CONFIG__, __PARAMS__


class ConfigManager:
      def __init__(self, config = __CONFIG__, params = __PARAMS__):
            self.config = load_yaml_file(config)
            self.params = load_yaml_file(params)

            create_directory([self.config.ariticate_root])
      

      def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion

            create_directory([config.root_dir])
            data_ingestion_config = DataIngestionConfig(
                  root_dir=config.root_dir,
                  source_url=config.source_url,
                  ziped_data=config.ziped_data_path,
                  unzip_data=config.unziped_file
            )
            return data_ingestion_config