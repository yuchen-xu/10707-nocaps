# average: 5 captions per image
# 2 captions per image contains detected object name (and was replaced by zzzzzzzz)

import json
import numpy as np
from tqdm import tqdm
import h5py

with open('data/coco/captions_train2017.json') as f:
    b = json.loads(f.read())

with open('data/coco/train_detectron.json') as f:
    detect = json.loads(f.read())

out = {'info' : b['info'], 'licenses' : b['licenses'], 'images' : [], 'annotations' : []}
img_id_to_name = {}
img_id_to_box = {}

for image in b['images']:
    if image['file_name'] in detect:
        img_id_to_name[image['id']] = detect[image['file_name']]['name']
        img_id_to_box[image['id']] = detect[image['file_name']]['box']
    out['images'].append(image)

count = 0
for annotation in b['annotations']:
    if annotation['image_id'] in img_id_to_name:
        annotation['caption'] = annotation['caption'].replace(img_id_to_name[annotation['image_id']], 'zzzzzzzz')
    out['annotations'].append(annotation)
    if 'zzzzzzzz' in annotation['caption']:
        count += 1

with open('data/coco/captions_train2017_mod.json', 'w') as f:
    f.write(json.dumps(out))

# Average captions per image: 5.0027
# Average captions modified per image: 1.9987
# 117432 / 118287 modified
print(f'Average captions per image: {len(out["annotations"]) / len(out["images"]):.4f}')
print(f'Average captions modified per image: {count / len(img_id_to_name):.4f}')
print()

# features are 2048d
# bounding boxes are (x1, y1, x2, y2) for both h5 and Detectron outputs

hf = h5py.File('data/coco_train2017_vg_detector_features_adaptive.h5', 'r+')

# method adapted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    bounding box format (x1, y1, x2, y2), x1 < x2, y1 < y2

    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    x_right = min(bb1[2], bb2[2])
    y_top = max(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

avg_boxes_before, avg_boxes_after = 0, 0
# keep_inds = []

for i, image_id in enumerate(tqdm(hf['image_id'])):
    num_boxes = hf['num_boxes'][i]
    avg_boxes_before += num_boxes

    if image_id not in img_id_to_box:
        avg_boxes_after += num_boxes
        continue

#     keep_inds.append(i)

    # easier to create a larger thing then trim down, compared to append (not in-place for numpy)
    new_boxes, new_features = np.zeros(num_boxes * 4), np.zeros(num_boxes * 2048)
    new_count = 0

    for j in range(num_boxes):
        if iou(hf['boxes'][i][j*4:j*4+4], img_id_to_box[image_id]) < 1/3:
            new_boxes[new_count*4:new_count*4+4] = hf['boxes'][i][j*4:j*4+4]
            new_features[new_count*2048:new_count*2048+2048] = hf['features'][i][j*2048:j*2048+2048]
            new_count += 1

    hf['boxes'][i] = new_boxes[:4*new_count]
    hf['features'][i] = new_features[:2048*new_count]
    hf['num_boxes'][i] = new_count

    avg_boxes_after += new_count

# for key in tqdm(hf.keys()):
#     tmp = hf[key][keep_inds]
#     del hf[key]
#     hf.create_dataset(key, data=tmp)

# coco train
# -2.5
print(f'Average # boxes before: {avg_boxes_before / len(hf["num_boxes"]):.4f}')
print(f'Average # boxes after: {avg_boxes_after / len(hf["num_boxes"]):.4f}')

hf.close()