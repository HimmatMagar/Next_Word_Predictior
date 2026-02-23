import os
import yaml
import json
import pickle
from pathlib import Path
from box.config_box import ConfigBox
from src.nextWordPrediction import logger
from ensure import ensure_annotations
from box.exceptions import BoxValueError


@ensure_annotations
def load_yaml_file(yaml_file: Path) -> ConfigBox:
      """
            load the yaml file from path

            Args:
                  yaml file: Path
            
            Return:
                  config box: it return the content of the yaml file
      """
      try:
            with open(yaml_file) as f:
                  content = yaml.safe_load(f)
                  logger.info(f"yaml file: {yaml_file} loaded successfully")
                  return ConfigBox(content)
      except BoxValueError:
            raise ValueError("Yaml file is empty")
      except Exception:
            raise Exception


@ensure_annotations
def create_directory(list_directory: list, verbose=True):
      """
            Create a directory
            Args:
                  directory name: list
      """
      for filename in list_directory:
            os.makedirs(filename, exist_ok=True)

            if verbose:
                  logger.info(f"File directory: {filename} created successfully")


@ensure_annotations
def save_json(path: Path, data: dict):
      """
            Save the json file in path

            Args:
                  file path: (Path) It store the json in path
                  data: (dict) It store into the json file
      """
      try:
            with open(path, 'w') as f:
                  json.dump(data, f, indent=4)
            logger.info(f"json file: {path} saved successfully")
      except Exception as e:
            raise e


@ensure_annotations
def load_file(file:Path):
      """
            load the pickle file from path

            Args:
                  json file: Path
            
            Return:
                 data: it return the content of the pickle file for the model
      """
      try:
            with open(file, 'rb') as f:
                  data = pickle.load(f)
            return data
      except Exception as e:
            raise e