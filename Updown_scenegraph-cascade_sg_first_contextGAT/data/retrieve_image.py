import urllib.request
import json

with open('nocaps_test_image_info.json') as json_data:
    d = json.load(json_data)
    images = d["images"]
    for obj in images:
        urllib.request.urlretrieve(obj['coco_url'], "/home/ubuntu/jeff/dataset/nocaps/" + str(obj["id"]).zfill(6)+".jpg")
