from src.nextWordPrediction import logger
from src.nextWordPrediction.config import ConfigManager
from src.nextWordPrediction.components.model_building import ModelBuilding


stage_name = "Model Building Stage"

class ModelBuildingPipeline:
      def __init__(self):
            pass

      def main(self):
            config = ConfigManager()
            model_building_config = config.get_model_building_config()
            model_build = ModelBuilding(model_building_config)
            model_build.build_model()


if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = ModelBuildingPipeline()
            obj.main()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e