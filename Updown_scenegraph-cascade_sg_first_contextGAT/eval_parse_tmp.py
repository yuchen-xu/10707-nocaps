# tmp file since eval_nocaps failed
# due to some problem with evalai, the evaluation is not running that well

import pickle
import json
from updown.utils.evalai import NocapsEvaluator

submit = True

with open("tmp_sgg_prediction", "rb") as fp:  # Pickling
    predictions = pickle.load(fp)
with open("tmp_imageid", "rb") as fp:  # Pickling
    img_ids = pickle.load(fp)

print(len(predictions), len(img_ids))

predictions_dict = []

for image_id, caption in zip(img_ids, predictions):
    predictions_dict.append({"image_id": int(image_id), "caption": " ".join(caption)})

json.dump(predictions_dict, open("predictions.json", "w"))

if submit:
    evaluator = NocapsEvaluator("val")
    evaluation_metrics = evaluator.evaluate(predictions_dict)

    print(f"Evaluation metrics:")
    for metric_name in evaluation_metrics:
        print(f"\t{metric_name}:")
        for domain in evaluation_metrics[metric_name]:
            print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])