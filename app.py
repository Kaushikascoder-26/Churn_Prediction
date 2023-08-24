from flask import Flask , render_template , request
import numpy as np 
import pickle
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing  import StandardScaler

model = pickle.load(open('Churn_project.pkl','rb'))
standard_scalar = pickle.load(open('stadardscalar.pkl','rb'))



app = Flask(__name__)

@app.route('/')
def fun():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET','POST'])
def fun1():
    a = [i for i in request.form.values()]

    a = [int(j) if j.isdigit() else float(j) for j in a]

    a = np.array([a])

    res = standard_scalar.transform(a)

    sol = model.predict(res)[0]

    if sol == 1:
        return render_template('index.html' , value = 'Customer Churned')
    else:
        return render_template('index.html' , value = 'Customer not churned')



if __name__ == '__main__':
    app.run()