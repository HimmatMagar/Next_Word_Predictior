import os
import zipfile
import urllib.request as req
from src.nextWordPrediction import logger
from src.nextWordPrediction.entity import DataIngestionConfig


class DataIngestion:
      def __init__(self, config: DataIngestionConfig):
            self.config = config
      
      def download_file(self):
            if not os.path.exists(self.config.ziped_data):
                  filename, header = req.urlretrieve(
                        url = self.config.source_url,
                        filename=self.config.ziped_data
                  )
                  logger.info(f"{filename} download with following {header}")
            else:
                  logger.info(f"File already created")
      
      def extract_data(self):
            file = self.config.unzip_data
            os.makedirs(file, exist_ok=True)
            with zipfile.ZipFile(self.config.ziped_data, 'r') as f:
                  f.extractall(file)