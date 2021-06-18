
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy
app = Flask(__name__)

model = pickle.load(open("Final_model.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
   
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7])])
    print(row_df)
    prediction=model.predict(row_df)
    output=prediction
    output = str(float(output))
    return  render_template('result.html',pred=f'MPG (miles per gallon) of your car is {output}')
  



if __name__ == '__main__':
    app.run(debug=True)