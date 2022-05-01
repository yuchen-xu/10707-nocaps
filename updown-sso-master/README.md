Separate Representations for Scene and Object (SSO) modification of UpDown
=====================================

Author: Yuchen Xu (@yuchen-xu), as part of a group project for 10707 Advanced Deep Learning (Spring 2022) at Carnegie Mellon University.

The SSO modification to the UpDown Captioner Baseline for `nocaps`. Original codebase at https://github.com/nocaps-org/updown-baseline.

Use the instructions from the original authors to set up: https://nocaps.org/updown-baseline/setup_dependencies.html
Note that requirements.txt has been modified, as well as some scripts from the original codebase.

For training, you need to do a few things:
1. Download the 2017 Train dataset from COCO.
2. Run auxiliary_scripts/detectron_proc.py to extract bounding boxes and object names. There are four identical scripts to allow for faster processing. This takes approximately 3.5 hours on a GPU, but could take days on a CPU (uncomment the CPU line in the Detectron setup part).
3. Download the h5 file from the setup link above, for COCO train images. Warning: this file is extremely large (~30G).
4. Run auxiliary_scripts/modify_coco_feat_capt.py. This will give you the appropriate training data (both caption and image).
5. Otherwise, follow the setup instructions, including building vocabulary. Use the yaml file given here for settings. You also need to run the following steps on the validation data.

For inference (validation and test), follow this:
1. Run auxiliary_scripts/download_data.py to download validation and/or test images of nocaps.
2. Download the h5 files from the setup link.
3. Run auxiliary_scripts/modify_nocaps_feat_capt.py.
4. Note that this is markedly different because validation and test captions are not provided; instead, the predictions are uploaded to the EvalAI server for evaluation.
