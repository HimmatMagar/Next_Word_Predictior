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
      seq_len: int
      data_file_path: Path

@dataclass(frozen=True)
class ModelBuildingConfig:
      root_dir: Path
      input_file: Path
      output_file: Path
      model: Path
      seq_length: int
      lstm_unit: int
      embedding_units: int
      epochs: int
      batch_size: int
      learning_rate: float