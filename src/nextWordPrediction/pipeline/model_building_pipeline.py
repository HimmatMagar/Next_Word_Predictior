import mlflow
from nextWordPrediction import logger
from nextWordPrediction.config import ConfigManager
from nextWordPrediction.utils.mlflow_config import configure_mlflow, save_run_id
from nextWordPrediction.components.training_model import TrainModel


stage_name = "Model Building Stage"

class ModelBuildingPipeline:
      def __init__(self):
            pass

      def main(self):
            config = ConfigManager()
            model_building_config = config.get_model_building_config()
            try:
                  configure_mlflow(experiment_name="LSTM-Model")

                  with mlflow.start_run(run_name="LSTM-retrained") as run:
                        mlflow.log_params({
                              "seq len": model_building_config.seq_length,
                              "lstm unit": model_building_config.lstm_unit,
                              "embedding_unit": model_building_config.embedding_units,
                              "learning rate": model_building_config.learning_rate,
                              "num_layer": model_building_config.num_layer,
                              "batch size": model_building_config.batch_size,
                              "epochs": model_building_config.epochs
                        })

                        vocab_path = "artifact/data_transformation/tokenizer.pkl"
                        mlflow.log_artifact(
                              local_path=vocab_path,
                              artifact_path="vecob"
                        )

                        model_build = TrainModel(model_building_config)
                        model, perplexity = model_build.train_model()
                        
                        mlflow.log_metric("val perplexity", perplexity)

                        mlflow.pytorch.log_model(
                              pytorch_model = model,
                              artifact_path="pytorch_model",
                              serialization_format="pickle",
                              registered_model_name="pytorch-model"
                        )
                        logger.info("Model logged successfully")
                        save_run_id(run.info.run_id)
            except Exception:
                  raise

if __name__ == "__main__":
      try:
            logger.info(f">>>>>> {stage_name} started <<<<<<")
            obj = ModelBuildingPipeline()
            obj.main()
            logger.info(f">>>>>> {stage_name} completed <<<<<<")
      except Exception as e:
            logger.exception(e)
            raise e