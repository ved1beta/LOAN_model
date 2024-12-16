from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                person_age=float(request.form.get('person_age')),
                person_income=float(request.form.get('person_income')),
                person_home_ownership=request.form.get('person_home_ownership'),
                person_emp_length=float(request.form.get('person_emp_length')),
                loan_intent=request.form.get('loan_intent'),
                loan_grade=request.form.get('loan_grade'),
                loan_amnt=float(request.form.get('loan_amnt')),
                loan_int_rate=float(request.form.get('loan_int_rate')),
                loan_percent_income=float(request.form.get('loan_percent_income')),
                cb_person_default_on_file=request.form.get('cb_person_default_on_file'),
                cb_person_cred_hist_length=float(request.form.get('cb_person_cred_hist_length'))
            )
            
            pred_df = data.get_data_as_data_frame()
            print("Prediction DataFrame:", pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("Prediction Results:", results)
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template('home.html', error="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port = 8000)
