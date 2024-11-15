import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print("Features before transform:", features)
            data_scaled = preprocessor.transform(features)
            print("Features after transform:", data_scaled)
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        person_age: float,
        person_income: float,
        person_home_ownership: str,
        person_emp_length: float,
        loan_intent: str,
        loan_grade: str,
        loan_amnt: float,
        loan_int_rate: float,
        loan_percent_income: float,
        cb_person_default_on_file: str,
        cb_person_cred_hist_length: float
    ):
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_default_on_file = cb_person_default_on_file
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "person_age": [self.person_age],
                "person_income": [self.person_income],
                "person_home_ownership": [self.person_home_ownership],
                "person_emp_length": [self.person_emp_length],
                "loan_intent": [self.loan_intent],
                "loan_grade": [self.loan_grade],
                "loan_amnt": [self.loan_amnt],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_default_on_file": [self.cb_person_default_on_file],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) 