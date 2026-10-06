from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
      root_dir: Path
      source_url: str
      ziped_data: Path
      unzip_data: Path


@dataclass(frozen=True)
class DataTransformationConfig:
      root_dir: Path
      data_file_path: Path

@dataclass(frozen=True)
class ModelBuildingConfig:
      root_dir: Path
      input_file: Path
      val_file: Path
      model: Path
      seq_length: int
      lstm_unit: int
      num_layer: int
      embedding_units: int
      epochs: int
      batch_size: int
      learning_rate: float

@dataclass(frozen=True)
class ModelEvalConfig:
      root_dir: Path
      test_file: Path
      model: Path
      metric: Path