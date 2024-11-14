import os 
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.pipeline.logger import logging

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_transformer_object(self):
        try:
            categorical_features = [
            'person_home_ownership',
            'loan_intent',
            'loan_grade',
            'cb_person_default_on_file']      
            numerical_features = [
            'person_age',
            'person_income',
            'person_emp_length',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length'
    ]
            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent") ),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                ]
            )
            logging.info("num cols missing values handeled ")
            logging.info("cat clos encoding completed ")
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
            pass
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessor obj")
            
            preprocessor_obj = self.get_transformer_object()
            target_column_name = "loan_status"
            
            # Separating features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing on training and test datasets")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            