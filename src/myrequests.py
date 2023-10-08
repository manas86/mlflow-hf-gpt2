import requests

url = 'http://localhost:7000/predict'
data = {'prompt': 'Once upon a time,'}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(result['prediction'])
else:
    print('Prediction request failed')
