from allennlp.data import Vocabulary
from datasets import TrainingDataset, ValidationDataset
import argparse
import shutil
import time
import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pickle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Decoder
from utils import collate_fn, save_checkpoint, AverageMeter, adjust_learning_rate, accuracy, create_captions_file


def main():
    """
    Training and validation.
    """
    # read vocabulary
    global vocabulary
    vocabulary = Vocabulary.from_files("data/vocabulary")
    vocab_size = vocabulary.get_vocab_size()
    # Initialize / load checkpoint
    if args.checkpoint is None:
        decoder = Decoder(attention_dim=args.attention_dim,
                          embed_dim=args.emb_dim,
                          decoder_dim=args.decoder_dim,
                          graph_features_dim=args.graph_features_dim,
                          vocab_size=vocab_size,
                          dropout=args.dropout,
                          cgat_obj_info=args.cgat_obj_info,
                          cgat_rel_info=args.cgat_rel_info,
                          cgat_k_steps=args.cgat_k_steps,
                          cgat_update_rel=args.cgat_update_rel,
                          vocabulary=vocabulary
                          )
        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))
        tracking = {'eval': [], 'test': None}
        start_epoch = 0
        best_epoch = -1
        epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation
        best_stopping_score = 0.  # stopping_score right now
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        args.stopping_metric = checkpoint['stopping_metric']
        best_stopping_score = checkpoint['metric_score']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        tracking = checkpoint['tracking']
        best_epoch = checkpoint['best_epoch']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.MultiLabelMarginLoss().to(device)

    train_image_features_h5path = "/home/ubuntu/jeff/dataset/coco_train2017_vg_detector_features_adaptive.h5"
    train_captions_jsonpath = "data/coco/captions_train2017.json"

    val_image_features_h5path = "/home/ubuntu/jeff/dataset/coco_val2017_vg_detector_features_adaptive.h5"
    val_captions_jsonpath = "data/coco/captions_val2017.json"

    if not args.test_val:
        # Custom dataloaders
        train_loader = torch.utils.data.DataLoader(TrainingDataset(args.data_folder, vocabulary,
                                                                   train_captions_jsonpath, train_image_features_h5path),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(ValidationDataset(args.data_folder, vocabulary,
                                                               val_captions_jsonpath, val_image_features_h5path),
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)



    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == args.patience:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        if not args.test_val:
            # One epoch's training
            train(train_loader=train_loader,
                  decoder=decoder,
                  criterion_ce=criterion_ce,
                  criterion_dis=criterion_dis,
                  decoder_optimizer=decoder_optimizer,
                  epoch=epoch)

        # One epoch's validation
        validate(val_loader=val_loader,
                      decoder=decoder,
                      criterion_ce=criterion_ce,
                      criterion_dis=criterion_dis,
                      epoch=epoch)


        is_best = True

        # Save checkpoint
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                        args.stopping_metric, best_stopping_score, tracking, is_best, args.outdir, best_epoch)

    with open(os.path.join(args.outdir, 'TRACKING.'+args.data_name+'.pkl'), 'wb') as f:
        pickle.dump(tracking, f)

def train(train_loader, decoder, criterion_ce, criterion_dis, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - start)

        (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens) = sample
        # Move to GPU, if available
        imgs = imgs.to(device)
        obj = obj.to(device)
        obj_mask = obj_mask.to(device)
        rel = rel.to(device)
        rel_mask = rel_mask.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, obj, rel, obj_mask, rel_mask,
                                                                          pair_idx, caps, caplens)
        # Max-pooling across predicted words across time steps for discriminative supervision
        scores_d = scores_d.max(1)[0]


        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0), scores_d.size(1)).to(device)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:, :length - 1] = targets[:, :length - 1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data
        # Calculate loss
        loss_d = criterion_dis(scores_d, targets_d.long())
        loss_g = criterion_ce(scores, targets)
        loss = loss_g + (10 * loss_d)


        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

def validate(val_loader, decoder, criterion_ce, criterion_dis, epoch):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param epoch: for which epoch is validated
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        # for i, (imgs, caps, caplens,allcaps) in enumerate(val_loader):
        for i, sample in enumerate(val_loader):
            if i % 5 != 0:
                # only decode every 5th caption, starting from idx 0.
                # this is because the iterator iterates over all captions in the dataset, not all images.
                if i % args.print_freq_val == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                    batch_time=batch_time,
                                                                                    loss=losses, top5=top5accs))
                continue

            (imgs, obj, rel, obj_mask, rel_mask, pair_idx, caps, caplens, _) = sample
            # Move to GPU, if available
            imgs = imgs.to(device)
            obj = obj.to(device)
            obj_mask = obj_mask.to(device)
            rel = rel.to(device)
            rel_mask = rel_mask.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, scores_d, caps_sorted, decode_lengths, sort_ind = decoder(imgs, obj, rel, obj_mask, rel_mask,
                                                                              pair_idx, caps, caplens)

            # Max-pooling across predicted words across time steps for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(scores_d.size(0), scores_d.size(1)).to(device)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:, :length - 1] = targets[:, :length - 1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data

            # Calculate loss
            loss_d = criterion_dis(scores_d, targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss = loss_g + (10 * loss_d)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq_val == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

                _, preds = torch.max(scores_copy, dim=2)
                # print(preds)
                instance_predictions = preds[0].tolist()

                # De-tokenize caption tokens and trim until first "@@BOUNDARY@@".
                caption = [
                    vocabulary.get_token_from_index(p) for p in instance_predictions
                ]
                eos_occurences = [
                    j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"
                ]
                caption = (
                    caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption
                )
                print({"caption": " ".join(caption)})

    decoder.train()


if __name__ == '__main__':
    metrics = ["CIDEr", "SPICE", "loss", "top5"]
    parser = argparse.ArgumentParser('Nocap')
    # Add config file arguments
    parser.add_argument('--data_folder', default='data', type=str,
                        help='folder with data files saved by create_input_files.py')
    parser.add_argument('--data_name', default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help='base name shared by data files')
    parser.add_argument('--print_freq', default=100, type=int, help='print training stats every __ batches')
    parser.add_argument('--print_freq_val', default=50, type=int, help='print validation stats every __ batches')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint, None if none')
    parser.add_argument('--outdir', default='/home/ubuntu/jeff/outputs', type=str,
                        help='path to location where to save outputs. Empty for current working dir')
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--seed', default=42, type=int, help='The random seed that will be used.')
    parser.add_argument('--emb_dim', default=1024, type=int, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', default=1024, type=int, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', default=1024, type=int, help='dimension of decoder lstm layers')
    parser.add_argument('--graph_features_dim', default=512, type=int, help='dimension of graph features')
    parser.add_argument('--cgat_obj_info', default=True, type=bool, help='whether to use object info in CGAT')
    parser.add_argument('--cgat_rel_info', default=True, type=bool, help='whether to use relation info in CGAT')
    parser.add_argument('--cgat_k_steps', default=1, type=int, help='how many CGAT steps to do')
    parser.add_argument('--cgat_update_rel', default=True, type=bool, help='whether to update relation states '
                                                                           'for k CGAT steps')
    parser.add_argument('--dropout', default=0.5, type=float, help='dimension of decoder RNN')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--patience', default=20, type=int,
                        help='stop training when metric doesnt improve for this many epochs')
    parser.add_argument('--stopping_metric', default='Bleu_4', type=str, choices=metrics,
                        help='which metric to use for early stopping')
    parser.add_argument('--test_val', default=False, action="store_true",)

    # Parse the arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    # setup initial stuff for reproducability
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size otherwise lot of computational overhead
    cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)

    # args.outdir = os.path.join(args.outdir,
    #                            'cascade_sg_first_contextGAT',
    #                            'batch_size-{bs}_epochs-{ep}_dropout-{drop}_patience-{pat}_stop-metric-{met}'.format(
    #                                bs=args.batch_size, ep=args.epochs, drop=args.dropout,
    #                                pat=args.patience, met=args.stopping_metric),
    #                            'emb-{emb}_att-{att}_dec-{dec}'.format(emb=args.emb_dim, att=args.attention_dim,
    #                                                                   dec=args.decoder_dim),
    #                            'cgat_useobj-{o}_userel-{r}_ksteps-{k}_updaterel-{u}'.format(
    #                                o=args.cgat_obj_info, r=args.cgat_rel_info, k=args.cgat_k_steps,
    #                                u=args.cgat_update_rel),
    #                            'seed-{}'.format(args.seed))
    if os.path.exists(args.outdir) and args.checkpoint is None:
        # answer = input("\n\t!! WARNING !! \nthe specified --outdir already exists, "
        #                "probably from previous experiments: \n\t{}\n"
        #                "Ist it okay to delete it and all its content for current experiment? "
        #                "(Yes/No) .. ".format(args.outdir))
        # if answer.lower() == "yes":
        print('SAVE_DIR will be deleted ...')
        shutil.rmtree(args.outdir)
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
        # else:
        #     print('To run this experiment and preserve the other one, change some settings, like the --seed.\n'
        #           '\tExiting Program...')
        #     exit(0)
    elif os.path.exists(args.outdir) and args.checkpoint is not None:
        print('continueing from checkpoint {} in {}...'.format(args.checkpoint, args.outdir))
    elif not os.path.exists(args.outdir) and args.checkpoint is not None:
        print('set a checkpoint to continue from, but the save directory from --outdir {} does not exist. '
              'creating directory...')
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.outdir, 'hypotheses'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'references'), exist_ok=True)
    main()