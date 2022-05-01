import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import collate_fn, create_captions_file, create_batched_graphs
import torch.nn.functional as F
from tqdm import tqdm
import dgl
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from allennlp.data import Vocabulary
from datasets import TrainingDataset, ValidationDataset


def beam_evaluate(data_name, checkpoint_file, data_folder, beam_size, outdir, graph_feature_dim=512, dataset='TEST'):
    """
    Evaluation
    :param data_name: name of the data files
    :param checkpoint_file: which checkpoint file to use
    :param data_folder: folder where data is stored
    :param beam_size: beam size at which to generate captions for evaluation
    :param outdir: place where the outputs are stored, so the checkpoint file
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model():
        # Load model using checkpoint file provided
        torch.nn.Module.dump_patches = True
        checkpoint = torch.load(os.path.join(outdir, checkpoint_file), map_location=device)
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        return decoder


    decoder = load_model()
    global vocabulary
    vocabulary = Vocabulary.from_files("data/vocabulary")
    vocab_size = vocabulary.get_vocab_size()
    boundary_index = vocabulary.get_token_index("@@BOUNDARY@@")
    pad_index = vocabulary.get_token_index("@@UNKNOWN@@")

    # DataLoader
    # TODO: later test with nocaps, first use the validation dataset
    val_image_features_h5path = "/home/ubuntu/jeff/dataset/coco_val2017_vg_detector_features_adaptive.h5"
    val_captions_jsonpath = "data/coco/captions_val2017.json"
    loader = torch.utils.data.DataLoader(
        ValidationDataset(args.data_folder, vocabulary,
                          val_captions_jsonpath, val_image_features_h5path),
        batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available())



    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for caption_idx, (image_features, obj, rel, obj_mask, rel_mask, pair_ids, caps, caplens, orig_caps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        if caption_idx % 5 != 0:
            continue

        k = beam_size


        # Move to GPU device, if available
        image_features = image_features.to(device)  # (1, 36, 2048)
        obj = obj.to(device)
        rel = rel.to(device)
        obj_mask = obj_mask.to(device)
        rel_mask = rel_mask.to(device)
        pair_ids = pair_ids.to(device)
        image_features_mean = image_features.mean(1)
        image_features_mean = image_features_mean.expand(k, 2048)
        graph_features_mean = torch.cat([obj, rel], dim=1).sum(dim=1) / \
                              torch.cat([obj_mask, rel_mask], dim=1).sum(dim=1, keepdim=True)
        graph_features_mean = graph_features_mean.to(device)
        graph_features_mean = graph_features_mean.expand(k, graph_feature_dim)

        # initialize the graphs
        g = create_batched_graphs(obj, obj_mask, rel, rel_mask, pair_ids, beam_size=k)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.tensor([[boundary_index]] * k, dtype=torch.long).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            h1, c1 = decoder.top_down_attention(torch.cat([h2, image_features_mean, graph_features_mean, embeddings], dim=1),
                                                (h1, c1))  # (batch_size_t, decoder_dim)
            cgat_out, cgat_mask_out = decoder.context_gat(h1, g, batch_num_nodes=g.batch_num_nodes)
            # make sure the size doesn't decrease
            of = obj.repeat(cgat_out.size(0), 1, 1)
            om = obj_mask.repeat(cgat_mask_out.size(0), 1)
            cgat_obj = torch.zeros_like(of)  # size of number of objects
            cgat_obj[:, :cgat_out.size(1)] = cgat_out  # fill with output of io attention
            cgat_mask = torch.zeros_like(om)  # mask shaped like original objects
            cgat_mask[:, :cgat_mask_out.size(1)] = cgat_mask_out  # copy over mask from io attention
            cgat_obj[~cgat_mask & om] = of[~cgat_mask & om]  # fill the no in_degree nodes with the original state
            # we pass the object mask. We used the cgat_mask only to determine which io's where filled and which not.
            graph_weighted_enc = decoder.cascade1_attention(cgat_obj, h1, mask=om)
            img_weighted_enc = decoder.cascade2_attention(image_features, torch.cat([h1, graph_weighted_enc], dim=1))
            h2, c2 = decoder.language_model(torch.cat([graph_weighted_enc, img_weighted_enc, h1], dim=1), (h2, c2))
            scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != boundary_index]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            graph_features_mean = graph_features_mean[prev_word_inds[incomplete_inds]]
            gs = dgl.unbatch(g)
            g = dgl.batch([gs[incomp_i] for incomp_i in incomplete_inds])
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                if len(complete_seqs) == 0:
                    # if we have to terminate, but none of the sequences are complete,
                    # recreate the complete inds without removing the incomplete ones: so everything.
                    complete_inds = list(set(range(len(next_word_inds))))
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        # img_caps = [' '.join(c) for c in orig_caps]
        img_caps = [c for c in orig_caps]
        references.append(img_caps)

        # Hypotheses
        hypothesis = ([vocabulary.get_token_from_index(w) for w in seq if w not in {boundary_index, pad_index}])
        # hypothesis = ' '.join(hypothesis)
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

    # Calculate scores
    # metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    hypotheses_file = os.path.join(outdir, 'hypotheses', '{}.{}.Hypotheses.json'.format(dataset,
                                                                                        data_name.split('_')[0]))
    references_file = os.path.join(outdir, 'references', '{}.{}.References.json'.format(dataset,
                                                                                        data_name.split('_')[0]))
    create_captions_file(range(len(hypotheses)), hypotheses, hypotheses_file)
    create_captions_file(range(len(references)), references, references_file)
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
    return results


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
    cudnn.benchmark = True  # True only if inputs to model are fixed size, otherwise lot of computational overhead

    metrics_dict = beam_evaluate(args.data_name, args.checkpoint_file, args.data_folder, args.beam_size, args.outdir,
                                 graph_feature_dim=args.graph_feature_dim, dataset=args.dataset)
    print(metrics_dict)
