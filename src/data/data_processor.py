import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataProcessor:
    """Data preprocessing pipeline for House Price Prediction"""
    
    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize data processor
        
        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path', 'models/preprocessing')
        self.preprocessor = None
        self.trained = False
        logger.info("Initialized DataProcessor")
    
    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)
    
    def _define_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Define the column transformer based on data types"""
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from features
        if 'SalePrice' in numerical_features:
            numerical_features.remove('SalePrice')
        
        # Create pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Fills missing numerical values with the median
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing categorical values with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create a preprocessor pipeline using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessor on the input DataFrame.
        
        Args:
            df: Input DataFrame to fit the preprocessor on.
        """
        try:
            logger.info("Starting fit process")
            self.preprocessor = self._define_preprocessor(df)
            self.preprocessor.fit(df)
            self.trained = True
            logger.info("Fit process completed successfully")
            return self
        except Exception as e:
            logger.error(f"Error in fit: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            df: Input DataFrame to transform.
            
        Returns:
            np.ndarray: Transformed features.
        """
        try:
            if not self.trained:
                raise ValueError("Preprocessor not fitted. Call .fit() first.")
                
            logger.info("Starting transform process")
            # If the dataframe still contains the target column, drop it before transforming
            if 'SalePrice' in df.columns:
                df = df.drop('SalePrice', axis=1)

            X_transformed = self.preprocessor.transform(df)
            
            logger.info("Transform completed successfully")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of transformed features and target
        """
        try:
            logger.info("Starting fit_transform process")
            # Split features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Fit the preprocessor
            self.fit(X)
            
            # Transform the features
            X_transformed = self.transform(X)
            
            logger.info("Fit_transform completed successfully")
            return X_transformed, y
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def save_preprocessors(self) -> None:
        """Save preprocessor objects"""
        try:
            logger.info(f"Saving preprocessor to {self.preprocessing_path}")
            self._prepare_preprocessing_path()
            
            joblib.dump(
                self.preprocessor,
                Path(self.preprocessing_path) / 'preprocessor.joblib'
            )
            
            logger.info("Preprocessors saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
    
    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")
            
            preprocessor_path = Path(self.preprocessing_path) / 'preprocessor.joblib'
            self.preprocessor = joblib.load(preprocessor_path)
            
            self.trained = True
            logger.info("Preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
