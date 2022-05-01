import datasets
import argparse
import logging
import math
import os
import random
from pathlib import Path
import csv
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import VisualBertForVisualReasoning, VisualBertConfig, VisualBertModel
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
from updown.data.datasets import (
    TrainingDataset,
    EvaluationDataset,
    EvaluationDatasetWithConstraints,
)
from updown.config import Config

class VisualBertClassifier(nn.Module):
  def __init__(self, checkpoint, num_labels=2): 
    super(VisualBertClassifier, self).__init__() 
    self.num_labels = num_labels 

    # Load Model with given checkpoint and extract its body
    self.model = VisualBertModel.from_pretrained(checkpoint)
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, inputs):
    # Extract outputs from the body

    labels = inputs['labels']
    inputs.pop('labels')
    outputs = self.model(**inputs)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a VisualBERT model on a text classification task")

    parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
    )
    parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
    "--in-memory", action="store_true", help="Whether to load image features in memory."
    ) 
    parser.add_argument(
    "--cpu-workers", type=int, default=8, help="Number of CPU workers to use for data loading."
    )     
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="uclanlp/visualbert-nlvr2",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/similarity", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        required=False,
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to run.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    _C = Config(args.config, args.config_override)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", 
        use_fast=not args.use_slow_tokenizer,
        padding="max_length", 
        max_length=args.max_length, 
        truncation=True)
    pretrained_model = VisualBertModel.from_pretrained(
        "uclanlp/visualbert-vqa-coco-pre",
        from_tf=False,
        # config=config,
    )
    
    model = VisualBertClassifier(checkpoint="uclanlp/visualbert-vqa-coco-pre")

    dataset = TrainingDataset.from_config(_C, in_memory=False)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    dataset = dataset
    train_dataset, eval_dataset = random_split(dataset, [train_size,test_size]) #or=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.collate_fn,
        # train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=dataset.collate_fn, batch_size=args.per_device_eval_batch_size)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    num_training_steps = args.max_train_steps
    ## Start Training ##
    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(args.num_epochs * len(eval_dataloader)))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    log_file = open("log_file.csv", "w+")
    csv_writer = csv.writer(log_file)
    completed_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v for k, v in batch.items()}
            outputs = model(inputs=batch)
            loss = outputs.loss
            csv_writer.writerow([float(loss)])
            accelerator.backward(loss)
            completed_steps += 1
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)
            
            if completed_steps >= args.max_train_steps:
                break

        '''
        model.eval()
        num_steps = 100
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(inputs=batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)
        '''
        print(loss)
        # print(metric.compute())

    if args.output_dir is not None:
        tokenizer.save_pretrained(args.output_dir)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model, args.output_dir + "/pytorch_model.bin")# unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()