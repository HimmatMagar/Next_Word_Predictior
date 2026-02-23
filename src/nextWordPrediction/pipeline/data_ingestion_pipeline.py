from src.nextWordPrediction import logger
from src.nextWordPrediction.config import ConfigManager
from src.nextWordPrediction.components.data_ingestion import DataIngestion

stage_name = "Data Ingestion Stage"

class DataIngestionPipeline():
      def __init__(self):
            pass

      def main(self):
            config = ConfigManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_data()


if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = DataIngestionPipeline()
            obj.main()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e