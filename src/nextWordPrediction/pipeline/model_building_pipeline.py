from nextWordPrediction import logger
from nextWordPrediction.config import ConfigManager
from nextWordPrediction.components.training_model import TrainModel


stage_name = "Model Building Stage"

class ModelBuildingPipeline:
      def __init__(self):
            pass

      def main(self):
            config = ConfigManager()
            model_building_config = config.get_model_building_config()
            model_build = TrainModel(model_building_config)
            model_build.train_model()

if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = ModelBuildingPipeline()
            obj.main()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e