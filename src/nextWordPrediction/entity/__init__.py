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
      vocab_size: int
      seq_len: int
      data_file_path: Path

@dataclass(frozen=True)
class ModelBuildingConfig:
      root_dir: Path
      input_file: Path
      output_file: Path
      model: Path
      vocab_size: int
      seq_length: int
      lstm_unit: int
      embedding_units: int
      optimizer: str
      epochs: int
      batch_size: int
      spliting: int
      activation: str