DATA_PATH = 'train2017'
OUT = 'train_detectron1.json'

import torch, torchvision

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
# cfg.MODEL.DEVICE='cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

with open('data/coco_classnames.txt') as f:
    import ast
    coco_labels = ast.literal_eval(f.read())

img_to_boxes = dict()

file_list = os.listdir(DATA_PATH)
file_list.sort()

qtr = len(file_list) // 4
file_list = file_list[:qtr]

for i, file in enumerate(tqdm(file_list)):
    im = cv2.imread(os.path.join(DATA_PATH, file))
    outputs = predictor(im)

    # pick the highest confidence non-background object
    if len(outputs['instances'].pred_boxes) == 0:
        continue

    # plus one because Detectron doesn't have background (0 in coco)
    img_to_boxes[file] = {'box' : outputs['instances'].pred_boxes[0].tensor.cpu().numpy().tolist()[0], 
                          'name' : coco_labels[outputs['instances'].pred_classes[0].item()+1]}

    if i % 1000 == 0:
        with open(OUT, 'w') as f:
            f.write(json.dumps(img_to_boxes))

with open(OUT, 'w') as f:
    f.write(json.dumps(img_to_boxes))
