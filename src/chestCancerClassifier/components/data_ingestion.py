import os
import zipfile
import gdown
from pathlib import Path
from chestCancerClassifier import logger
from chestCancerClassifier.utils.common import get_size
from chestCancerClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

             # Check if file already exists and has valid size
            if os.path.exists(zip_download_dir):
                file_size = get_size(Path(zip_download_dir))
                logger.info(f"File {zip_download_dir} already exists with size: {file_size}")
                if os.path.getsize(zip_download_dir) > 0:  # Check if file is not empty
                    logger.info(f"File {zip_download_dir} already downloaded. Skipping download.")
                    return str(zip_download_dir)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
