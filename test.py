import requests
import json

url = "http://localhost:8000/query"

query_text = 'What is fat-tailedness?'


response = requests.get(url, params={"query_text": query_text, "top_k": 2})

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
