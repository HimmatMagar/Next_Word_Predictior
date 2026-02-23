from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
      root_dir: Path
      source_url: str
      ziped_data: Path
      unzip_data: Path