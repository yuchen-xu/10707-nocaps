import json
from tqdm import tqdm
# import requests
import os

with open('/home/ubuntu/updown-baseline/data/nocaps/nocaps_test_image_info.json') as f:
    val = json.loads(f.read())

os.chdir('/home/ubuntu/updown-baseline/data/nocaps/test')

for image in tqdm(val['images']):
    os.system(f"wget {image['coco_url']}")
    # response = requests.get(image['coco_url'])
    # open(f"data/nocaps/val/{image['file_name']}", "wb").write(response.content)