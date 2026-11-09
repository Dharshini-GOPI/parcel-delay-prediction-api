
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained AI model
model = joblib.load("parcel_delay_model.pkl")


@app.route("/")
def home():
    return "Parcel Delay Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Convert arrival date into required features
    arrival_date = pd.to_datetime(data["arrival_date"])

    new_parcel = pd.DataFrame([{
        "Courier Company": data["courier_company"],
        "Parcel Description": data["parcel_description"],
        "Department": data["department"],
         year_text = str(data["year_of_study"]).strip().upper()
         year_mapping = {
             "FIRST YEAR": 1,
             "SECOND YEAR": 2,
             "THIRD YEAR": 3,
             "FOURTH YEAR": 4,
             "1": 1,
             "2": 2,
             "3": 3,
             "4": 4
         }

         year_number = year_mapping.get(year_text)

         if year_number is None:
                    raise ValueError("Invalid Year of Study: " + year_text)
        "Arrival Month": arrival_date.month,
        "Arrival Day": arrival_date.day,
        "Arrival Weekday": arrival_date.dayofweek
    }])

    # Predict parcel delay
    prediction = model.predict(new_parcel)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(new_parcel)[0]

    confidence = max(probabilities) * 100

    return jsonify({
        "prediction": str(prediction),
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
