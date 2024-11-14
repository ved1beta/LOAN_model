import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.pipeline.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [8, 16, 32, 64, 128],
                    'subsample': [0.8, 0.9]
                },
                "Logistic Regression": {
                    'solver': ['lbfgs', 'liblinear'],
                    'C': [0.001, 0.01, 0.1, 1, 10]
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [8, 16, 32, 64, 128]
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [8, 16, 32, 64, 128]
                }
            }


            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = accuracy_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)