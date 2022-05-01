import json
import csv
import numpy as np
def write_training_file(path, out_path):
    f = open(path)
    data = json.load(f)
    csv_file = open(out_path, "w+")
    csv_writer = csv.writer(csv_file)
    annotations = data['annotations']
    for caption in annotations:
        pos = caption['caption'].strip(".")

        sent = np.array(pos.split(" "))
        sent_len = len(sent)
        neg = np.random.permutation(sent_len)
        neg = sent[neg]
        neg_sent = " ".join(neg)
        csv_writer.writerow([pos, 1])
        csv_writer.writerow([neg_sent, 0])


path = "/data/10707-nocaps/data/coco/captions_train2017.json"
out_path = "/data/10707-nocaps/data/coco/natural_captions.csv"
write_training_file(path, out_path)