import requests

url = "https://deep-translate1.p.rapidapi.com/language/translate/v2"

payload = {
	"q": "ans",
	"source": "en",
	"target": "te"
}
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "2761bb9559msh64aaf6be6191b4cp191e83jsn51da63e6da75",
	"X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
}

response = requests.post(url, json=payload, headers=headers)
data=response.json()
print(data['data']['translations']['translatedText'])