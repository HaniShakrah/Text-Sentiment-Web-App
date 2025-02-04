# Importing necessary libraries
from flask import Flask,render_template,request
import pickle

# Loading both pickle files
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
mymodel=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['tweet']
        data = [message]
        vect = vectorizer.transform(data)
        my_prediction=mymodel.predict(vect)
        
        if my_prediction == 0:
            return render_template('index.html',prediction="Negative 😡")
        else:
            return render_template('index.html',prediction="Positive 😊")
if __name__=='__main__':
    app.run(debug=True)