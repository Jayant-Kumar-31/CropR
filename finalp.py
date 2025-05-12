from flask import Flask, render_template, request, jsonify
import requests
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("trained_random_forest_model.joblib")

# ThingSpeak API details
THINGSPEAK_CHANNEL_ID = "2781281"
THINGSPEAK_API_KEY = "OOFGKWCQJBIKZWWG"
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch-data', methods=['GET'])
def fetch_data():
    response = requests.get(THINGSPEAK_URL)
    if response.status_code == 200:
        data = response.json()
        if "feeds" in data and len(data["feeds"]) > 0:
            latest_entry = data["feeds"][0]
            temperature = float(latest_entry["field1"])
            humidity = float(latest_entry["field2"])
            ph = float(latest_entry["field3"])
            soil_moisture = float(latest_entry["field4"])
            
            # Predict using the model
            input_features = np.array([[temperature, humidity, ph, soil_moisture]])
            predicted_crop = model.predict(input_features)[0]
            
            return jsonify({
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "soil_moisture": soil_moisture,
                "predicted_crop": predicted_crop
            })
        else:
            return jsonify({"error": "No data found in ThingSpeak"}), 404
    else:
        return jsonify({"error": "Failed to fetch data from ThingSpeak"}), 500

if __name__ == '__main__':
    app.run(debug=True)

# Create the HTML file (index.html)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Crop Prediction</title>
    <script>
        function fetchData() {
            fetch('/fetch-data')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('result').innerHTML = "Error: " + data.error;
                    } else {
                        document.getElementById('temperature').innerText = data.temperature;
                        document.getElementById('humidity').innerText = data.humidity;
                        document.getElementById('ph').innerText = data.ph;
                        document.getElementById('soil_moisture').innerText = data.soil_moisture;
                        document.getElementById('predicted_crop').innerText = data.predicted_crop;
                    }
                })
                .catch(error => console.error('Error:', error));
        }
    </script>
</head>
<body>
    <h1>Crop Prediction System</h1>
    <button onclick="fetchData()">Fetch Data & Predict</button>
    <h3>Sensor Data:</h3>
    <p>Temperature: <span id="temperature">-</span></p>
    <p>Humidity: <span id="humidity">-</span></p>
    <p>PH: <span id="ph">-</span></p>
    <p>Soil Moisture: <span id="soil_moisture">-</span></p>
    <h3>Predicted Crop:</h3>
    <p><span id="predicted_crop">-</span></p>
</body>
</html>
"""

with open("templates/index.html", "w") as f:
    f.write(html_content)
