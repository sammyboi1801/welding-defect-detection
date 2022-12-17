from flask import Flask,render_template,request
import pickle
import numpy as np


model = pickle.load(open('xgboost-350.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html',message='')

@app.route('/predict',methods=['POST'])
def predict():
    str_result = ''

    current = float(request.form.get('current'))
    humidity = float(request.form.get('humidity'))
    temperature = float(request.form.get('temperature'))
    flow = float(request.form.get('flow'))
    job_temp = float(request.form.get('job_temp'))
    voltage = float(request.form.get('voltage'))

    #predict

    result = model.predict(np.array([current,humidity,temperature,flow,job_temp,voltage]).reshape(1,6))
    temp = model.predict_proba(np.array([current,humidity,temperature,flow,job_temp,voltage]).reshape(1,6))

    prob = np.max(temp[0])

    print(prob,temp)
    if result[0] == 0:
        str_result='No defect    '+str(round(prob, 4)*100)[:4]+'%'
    elif result[0] == 1:
        str_result='Porosity    '+str(round(prob, 4)*100)[:4]+'%'
    else:
        str_result='Tungsten Inclusion    '+str(round(prob, 4)*100)[:4]+'%'
    return render_template('home.html',message=str_result)



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)