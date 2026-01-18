from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

app = Flask(__name__)

# Load datasets
training = pd.read_csv("Training.csv")
description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")
workout = pd.read_csv("workout_df.csv")

# Prepare data
X = training.drop("prognosis", axis=1)
y = training["prognosis"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = SVC(kernel="linear")
model.fit(X, y_encoded)

symptoms_dict = {symptom: i for i, symptom in enumerate(X.columns)}

def helper(disease):
    dis_des = " ".join(description[description["Disease"] == disease]["Description"])
    my_precautions = precautions[precautions["Disease"] == disease].iloc[:, 1:].values.flatten().tolist()
    medications_list = medications[medications["Disease"] == disease]["Medication"].tolist()
    my_diet = diets[diets["Disease"] == disease]["Diet"].tolist()
    workout_list = workout[workout["disease"] == disease]["workout"].tolist()
    return dis_des, my_precautions, medications_list, my_diet, workout_list

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms")

    if not symptoms:
        return render_template("index.html", message="Please enter symptoms")

    user_symptoms = [s.strip().lower() for s in symptoms.split(",")]
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in user_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    prediction = model.predict([input_vector])
    predicted_disease = le.inverse_transform(prediction)[0]

    dis_des, my_precautions, medications_list, my_diet, workout_list = helper(predicted_disease)

    return render_template(
        "index.html",
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        my_precautions=my_precautions,
        medications=medications_list,
        my_diet=my_diet,
        workout=workout_list
    )

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

if __name__ == "__main__":
    app.run(debug=True)
