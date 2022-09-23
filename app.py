import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print("Predict_api_data:",data)
    output=regmodel.predict(np.array(list(data.values())).reshape(1,-1))
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    output=regmodel.predict(data)
    return render_template("home.html",prediction_text="The prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)