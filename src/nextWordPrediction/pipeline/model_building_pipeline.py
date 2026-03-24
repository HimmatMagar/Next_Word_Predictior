import mlflow
from nextWordPrediction import logger
from nextWordPrediction.config import ConfigManager
from nextWordPrediction.utils.mlflow_config import configure_mlflow
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

                  with mlflow.start_run(run_name="LSTM"):
                        mlflow.log_params({
                              "seq len": model_building_config.seq_length,
                              "lstm unit": model_building_config.lstm_unit,
                              "embedding_unit": model_building_config.embedding_units,
                              "learning rate": model_building_config.learning_rate,
                              "batch size": model_building_config.batch_size,
                              "epochs": model_building_config.epochs
                        })

                        vocab_path = "artifact/data_transformation/tokenizer.pkl"
                        mlflow.log_artifact(
                              local_path=vocab_path,
                              artifact_path="vecob"
                        )

                        model_build = TrainModel(model_building_config)
                        model, total_loss = model_build.train_model()
                        
                        mlflow.log_metric("loss", total_loss)

                        logged_model = mlflow.pytorch.log_model(
                              pytorch_model = model,
                              artifact_path="pytorch_model"
                        )

                        pyTorchModel = mlflow.register_model(
                              model_uri=f"models:/{logged_model.run_id}",
                              name="PyTorchLSTM"
                        )

                        client = mlflow.tracking.MlflowClient()
                        client.transition_model_version_stage(
                              name="PyTorchLSTM",
                              version=pyTorchModel.version,
                              stage="Staging"
                        )
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