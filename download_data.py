import requests
import zipfile
import io

url = "https://github.com/PuruSinghvi/Spam-Email-Classifier/tree/main/Datasets/combined_data.csv"
r = requests.get(url)
with open("combined_data.csv", "wb") as f:
    f.write(r.content)

print("Dataset downloaded!")