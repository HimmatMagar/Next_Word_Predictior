from src.nextWordPrediction import logger
from src.nextWordPrediction.config import ConfigManager
from src.nextWordPrediction.components.data_transformation import DataTransform

stage_name = "Data Transformation Stage"

class DataTransformPipeline():
      def __init__(self):
            pass

      def main(self):
            config = ConfigManager()
            data_transform_config = config.get_data_transform_config()
            data_transform = DataTransform(data_transform_config)
            data_transform.cleaning_pipeline()


if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = DataTransformPipeline()
            obj.main()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e