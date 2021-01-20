# Heart_Disease_Predictor

This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

Prerequisites

You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Project Structure

This project has four major parts :
Cardiovascular Disease Project.py - This contains code for our Machine Learning model to predict heart disease based on training data in 'cardio_train.csv' file.
app.py - This contains Flask APIs that receives users input through GUI or API calls, computes the precited value based on our model and returns it.
templates - This folder contains the HTML template to allow user to enter health details and displays the predicted probability of having a cardiovascular disease.

Running the project

Ensure that you are in the project home directory. Create the machine learning model by running below command -
python Cardiovascular Disease Project.py
This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API
python app.py
By default, flask will run on port 5000.

Navigate to URL http://localhost:5000
You should be able to view the homepage as below : alt text

Enter valid numerical values in all 5 input boxes and hit Predict.

If everything goes well, you should be able to see the predcited vaule on the HTML page! alt text
