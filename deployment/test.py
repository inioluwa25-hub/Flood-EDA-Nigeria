import requests

response = requests.post(
    "https://90c1-197-211-57-0.ngrok-free.app/predict",
    json={"rainfall": 1850, "river_level": 4.2, "drainage": 0.6, "urbanization": 45},
)
print(response.json())
