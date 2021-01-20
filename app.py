import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__,template_folder='template')
model = pickle.load(open('clf_model.pkl', 'rb'))
port = int(os.environ.get('PORT', 5000))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)
    output = '{0:.{1}f}'.format(prediction[0][1],2)


    if output > str(0.5):
        return render_template('index.html',prediction_text='Your heart is in danger.\nProbability of having a cardiovascular disease is {}'.format(output))
    else:
        return render_template('index.html',prediction_text='Your heart is healthy.\nProbability of having a  Cardiovascular disease is {}'.format(output))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port,debug=True)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0',port=8080)
