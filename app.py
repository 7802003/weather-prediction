from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import joblib

app=Flask(__name__)
@app.route('/',methods=["GET"])
def home():
    return render_template('result.html')

@app.route('/predict_weather',methods=['POST'])
def predict():
    if request.method=="POST":    
        precipitation=float(request.form["precipitation"])
        temp_max=float(request.form["temp_max"])
        temp_min=float(request.form["temp_min"])
        wind=float(request.form["wind"])
        try:
            prediction=preprocessDataAndPredict(precipitation,temp_max,temp_min,wind)
            if prediction==0:
                prediction='Dizzle'
            elif prediction==1:
                prediction='Fog'
            elif prediction==2:
                prediction='Rain'
            elif prediction==3:
                prediction='Snow'
            elif prediction==4:
                prediction='Sun'
            return render_template('result1.html',prediction=prediction)
        except ValueError:
            return "Please Enter valid input/Values"
    

def preprocessDataAndPredict(precipitation,temp_max,temp_min,wind):
    test_data=[[precipitation,temp_max,temp_min,wind]]
    test_data=np.array(test_data)
    test_data=pd.DataFrame(test_data)
    try:
        with open("C:/Users/Ankita/predict.pkl", "rb") as file:
            train_model = joblib.load(file)
            prediction = train_model.predict(test_data)
            return prediction
    except FileNotFoundError:
        return "Model file not found"

