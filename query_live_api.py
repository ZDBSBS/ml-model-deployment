import requests

# Live API URL (Render)
url = "https://ml-model-deployment-8mh2.onrender.com/model"

# Example payload
payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Prediction: {response.json()['prediction']}")