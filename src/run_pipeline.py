import os
import sys
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split

# Add src to path for imports
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.trainer import ModelTrainer
from src.utils.logger import default_logger as logger
from src.utils.config import config


def setup_mlflow():
    """Setup MLflow configuration"""
    try:
        # Set MLflow tracking URI explicitly
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Set experiment
        experiment_name = "House Price Prediction"
        try:
            mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists, just proceed
            pass
        
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow setup completed. Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment name: {experiment_name}")
        
        return experiment_name
        
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise

def run_pipeline():
    """
    Run the complete training pipeline
    """
    try:
        logger.info("Starting pipeline execution")
        
        # Setup MLflow first
        experiment_name = setup_mlflow()
        
        # Start MLflow run for the entire pipeline
        with mlflow.start_run(run_name="full_pipeline_run") as run:
            
            # 1. Data Loading
            logger.info("Step 1: Loading data")
            data_loader = DataLoader()
            df = data_loader.load_data()
            data_loader.validate_data(df)
            
            X, y = data_loader.split_features_target(df)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.get('test_size', 0.2), random_state=config.get('random_state', 42)
            )
            
            logger.info("Data loaded and split into training and testing sets")
            
            # 2. Preprocessing data
            logger.info("Step 2: Preprocessing data")
            preprocessor_handler = DataProcessor()
            # Fit the preprocessor on training data only
            preprocessor_handler.fit(X_train)
            
            # Now, transform both training and testing data
            X_train_processed = preprocessor_handler.transform(X_train)
            X_test_processed = preprocessor_handler.transform(X_test)
            
            preprocessor_handler.save_preprocessors()
            
            logger.info("Preprocessor fitted, data transformed, and preprocessor saved successfully.")
            
            # 3. Model Training and Evaluation
            # Pass the already processed data to the trainer
            logger.info("Step 3: Training and evaluating models")
            trainer = ModelTrainer(experiment_name)
            
            # Train all models (will create nested runs)
            trainer.train_all_models(
                X_train=X_train_processed,
                y_train=y_train,
                X_test=X_test_processed,
                y_test=y_test
            )
            
            # 4. Log best model info
            best_model = trainer.get_best_model()
            mlflow.log_params({
                "best_model_type": best_model['model'].__class__.__name__,
                "best_model_params": str(best_model['model'].get_params())
            })
            mlflow.log_metrics({
                f"best_model_{k}": v for k, v in best_model['metrics'].items()
            })
            
            logger.info("Pipeline execution completed successfully")
            return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Delete existing MLflow database if exists
    mlflow_db = Path("mlflow.db")
    if mlflow_db.exists():
        try:
            mlflow_db.unlink()
            logger.info("Deleted existing MLflow database")
        except PermissionError:
            logger.warning("Could not delete mlflow.db. It might be in use by another process. Proceeding anyway.")

    run_pipeline()
