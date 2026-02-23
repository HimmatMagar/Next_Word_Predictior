from src.nextWordPrediction import logger
from src.nextWordPrediction.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.nextWordPrediction.pipeline.data_transformation_pipeline import DataTransformPipeline

stage_name = "Data Ingestion Stage"
try:
      logger.info(f">>>>>> {stage_name} started <<<<<<")
      obj = DataIngestionPipeline()
      obj.main()
      logger.info(f">>>>>> {stage_name} completed <<<<<<")
except Exception as e:
      logger.exception(e)
      raise e


stage_name = "Data Transformation Stage"
try:
      logger.info(f">>>>>> {stage_name} started <<<<<<")
      obj = DataTransformPipeline()
      obj.main()
      logger.info(f">>>>>> {stage_name} completed <<<<<<")
except Exception as e:
      logger.exception(e)
      raise e