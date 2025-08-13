from src.data.data_loader import DataLoader
from src.utils.logger import default_logger as logger
from src.data.data_processor import DataProcessor

if __name__ == "__main__":
    try:
        logger.info("inisialisasi class")
        data_loader = DataLoader()
        data_processing = DataProcessor()
        
        logger.info('Star Download dataset')
        df = data_loader.load_data()

        logger.info("Data loaded")
        logger.info("Start processing")
        X,y = data_processing.fit_transform(df)
    except:
        logger.info("gagal load data")
