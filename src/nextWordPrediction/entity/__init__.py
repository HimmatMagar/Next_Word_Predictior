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