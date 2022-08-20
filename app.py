import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle


app = Flask(__name__)
dataset= pd.read_csv('Churn_Modelling.csv')
model1            = pickle.load(open('projecy7_DecisionTree.pkl','rb'))
model2            = pickle.load(open('project7_KNN.pkl','rb'))
model3            = pickle.load(open('project7_kernalsvm.pkl','rb'))
model4            = pickle.load(open('project7_randomforest.pkl','rb'))
model5            = pickle.load(open('project7_nb.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    CreditScore = int(request.args.get('CreditScore'))
    Age = int(request.args.get('Age'))
    Tenure = int(request.args.get('Tenure'))
    Balance = float(request.args.get('Balance'))
    NumOfProducts = int(request.args.get('NumOfProducts'))
    EstimatedSalary = float(request.args.get('EstimatedSalary'))
    
    Geography = (request.args.get('Geography'))
    if Geography=="France":
      Geography = 0
    elif Geography=="Germany":
      Geography = 1
    else:
      Geography = 2

    Gender = (request.args.get('Gender'))
    if Gender=="Male":
      Gender = 1
    else:
      Gender = 0

    HasCrCard = (request.args.get('HasCrCard'))
    if HasCrCard=="Yes":
      HasCrCard = 1
    else:
      HasCrCard = 0

    IsActiveMember = (request.args.get('IsActiveMember'))
    if IsActiveMember=="Male":
      IsActiveMember = 1
    else:
      IsActiveMember = 0

   
    Model = (request.args.get('Model'))
    if Model=="Decision Tree Classifier":
      prediction = model1.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])
    
    if Model=="KNN Classifier":
      prediction = model2.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])

    if Model=="SVM Classifier":
      prediction = model3.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])

    if Model=="Random Forest Classifier":
      prediction = model4.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])

    if Model=="Naive Bayes Classifier":
      prediction = model5.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])


    if prediction == [1]:
      return render_template('index.html', prediction_text='Customer Exited Bank', modeltype_text ="-> Prediction by " + Model)    
    else:
      return render_template('index.html', prediction_text='Customer Not Exited Bank', modeltype_text ="-> Prediction by " + Model)


if __name__=="__main__":
  app.run(debug=True)
