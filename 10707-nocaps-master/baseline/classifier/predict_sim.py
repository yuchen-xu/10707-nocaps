import torch
from transformers import BertTokenizer, VisualBertForVisualReasoning, BertForSequenceClassification
from updown.data.readers import CocoCaptionsReader, ConstraintBoxesReader, ImageFeaturesReader
import numpy as np
import json
import csv
# from train_sim_from_pretrained import VisualBertClassifier
# pretrained_path = "/data/10707-nocaps/baseline/checkpoints/similarity/"
pretrained_path = "/data/10707-nocaps/baseline/checkpoints/sim_v2/"
class VisualBertPrediction():
    def __init__(self, image_features_h5path, pretrained_path, in_memory=False):
        self.image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self.model = VisualBertForVisualReasoning.from_pretrained(
            #"uclanlp/visualbert-vqa-coco-pre")
            pretrained_path + "pytorch_model.bin", config=pretrained_path + "config.json")
        # self.model = torch.load(pretrained_path + 'pytorch_model.bin')
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def predict(self, image_id, text):

        image_features = torch.from_numpy(self.image_features_reader[image_id]).unsqueeze(0)
        visual_token_type_ids = torch.ones(image_features.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(image_features.shape[:-1], dtype=torch.long)

        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        inputs.update({
            "visual_embeds": image_features,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        })
        f = open("correlation_out.csv", "a+")
        csv_writer = csv.writer(f)
        with torch.no_grad():
            # shape: (batch_size, max_caption_length)
            # Pass finite state machine and number of constraints if using CBS.
            outputs = self.model(**inputs)# ["logits"]# .argmax(-1)
            
            logits = outputs.logits[0]
            prediction = (torch.exp(logits)/torch.sum(torch.exp(logits))).numpy()[1]
            print(prediction)
            csv_writer.writerow([image_id, text, prediction])



class BertSequencePrediction():
    def __init__(self, pretrained_path):
        # self.image_features_reader = ImageFeaturesReader(image_features_h5path, in_memory)
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_path + "pytorch_model.bin", config=pretrained_path + "config.json")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def predict(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", 
            padding="max_length", max_length=512, truncation=True)
        f = open("fluency_out.csv", "a+")
        csv_writer = csv.writer(f)
        with torch.no_grad():
            # shape: (batch_size, max_caption_length)
            # Pass finite state machine and number of constraints if using CBS.
            outputs = self.model(**inputs)# ["logits"]# .argmax(-1)
            logits = outputs.logits[0]
            #print(logits)
            prediction = (torch.exp(logits)/torch.sum(torch.exp(logits))).numpy()[1]
            # print(prediction)
            #prediction = outputs.logits.argmax(dim=-1)
            csv_writer.writerow([text, prediction])

def predict_fluency():
    model = BertSequencePrediction(pretrained_path)
    f = open("results.json")
    data = json.load(f)

    for img in data:
        res = model.predict(img['caption'])
def predict_correlation():
    path = "/data/10707-nocaps/data/nocaps_test_vg_detector_features_adaptive.h5"
    model = VisualBertPrediction(path, pretrained_path)
    cap = "a person riding a bike on a street"
    model.predict(4500, cap)


    f = open("results.json")
    data = json.load(f)

    for img in data:
        print(img)
        res = model.predict(img['image_id'], img['caption'])

predict_correlation()
