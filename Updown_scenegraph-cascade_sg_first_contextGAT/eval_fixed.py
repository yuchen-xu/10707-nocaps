# this is just a ad-hoc fix since our previous eval fails

import os
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='data', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--dataset', default='TEST', type=str, help='which split to use')
    parser.add_argument('--outdir', default='/home/ubuntu/jeff/outputs', type=str,
                        help='path to location where the outputs are saved, so the checkpoint')
    parser.add_argument('--checkpoint_file', type=str,
                        default="BEST_9_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar",
                        help="Checkpoint to use for beam search.")
    parser.add_argument('--beam_size', type=int, default=5,
            help="Beam size to use with beam search. If set to one we run greedy search.")
    parser.add_argument('--graph_feature_dim', type=int, default=512,
                        help="depends on which scene graph generator is used")
    args = parser.parse_args()

    # Calculate scores
    hypotheses_file = os.path.join(args.outdir, 'hypotheses', '{}.{}.Hypotheses.json'.format(args.dataset,
                                                                                        args.data_name.split('_')[0]))
    references_file = os.path.join(args.outdir, 'references', '{}.{}.References.json'.format(args.dataset,
                                                                                        args.data_name.split('_')[0]))
    coco = COCO(references_file)
    # add the predicted results to the object
    coco_results = coco.loadRes(hypotheses_file)
    # create the evaluation object with both the ground-truth and the predictions
    coco_eval = COCOEvalCap(coco, coco_results)
    # change to use the image ids in the results object, not those from the ground-truth
    coco_eval.params['image_id'] = coco_results.getImgIds()
    # run the evaluation
    coco_eval.evaluate()
    # Results contains: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"
    results = coco_eval.eval
    print(results)

