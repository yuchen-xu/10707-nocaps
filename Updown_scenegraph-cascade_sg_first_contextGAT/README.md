<h1> Image captioning with scene graph </h1>
Deep learning project by Jiaheng Hu

<h2> Data preparation </h2>

Create a folder called 'data'

Unzip all files (captions and images) and place the folders in 'data' folder.

<br>

Set up the graph-rcnn.pytorch repository at https://github.com/jwyang/graph-rcnn.pytorch.
Download the imp_relpn weight and place it in ./weights

<br>

Next type this command in a python environment: 
```bash
cd bottom-up_features
python create_sg_h5.py
```
This will create the scene graph features.
<br>


<h2> Training </h2>

To train the model, type:
```bash
python train.py
```

<h2> Evaluation </h2>

To evaluate the model on the coco eval dataset, edit the eval.py file to include the model checkpoint location and then type:
```bash
python eval.py
```

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR and ROUGE-L. Official MSCOCO evaluation scripts are used for measuring these scores.
  
To evaluate the mode on the nocaps datset, setup the updown baseline repo, then type:
```bash
python eval_nocaps.py
python eval_parse_tmp.py
```

The first script generates the prediction, while the second script reformat and submit the result
  
<h2>References</h2>

Code adapted with thanks from https://github.com/poojahira/image-captioning-bottom-up-top-down
