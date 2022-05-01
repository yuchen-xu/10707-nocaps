# Discriminative Methods using Language Modeling

## Code Structure
Modifies from [updown-baseline](https://nocaps.org/updown-baseline), by adding files

- [baseline/classifier](baseline/classifier)
- [baseline/updown/data/datasets_for_classifier.py](baseline/updown/data/datasets_for_classifier.py)


## Preparation
Download data from nocaps.org, modify the paths in [baseline/configs/updown_nocaps_val.yaml](baseline/configs/updown_nocaps_val.yaml).
To get data for fluency classifier, run
<pre><code> python classifier/prepare_captions.py
</code></pre>

Install required packages in [baseline/requirements.txt](baseline/requirements.txt)

## Training

Run the following code to run the correlation classifier
<pre><code> python classifier/train_sim.py
--config configs/updown_nocaps_val.yaml --output_dir checkpoints/sim
</code></pre>

Run the following code to run the fluency classifier
<pre><code>python classifier/train_natural.py
--config configs/updown_nocaps_val.yaml --output_dir checkpoints/nat
</code></pre>

## Prediction
Run ``python predict_sim.py`` with corresponding functions to get output csv files

## Reference
[Huggingface Transformers Library](https://github.com/huggingface/transformers/tree/main/examples/pytorch)
