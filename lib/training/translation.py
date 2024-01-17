
import sys

from tqdm import tqdm
sys.path.append(".")
sys.path.append("../..")

import os
import torch
from lib.config import *
from datasets import load_dataset
from lib.eval.translation import *
from torch.utils.data import DataLoader
from dataset.dataloader import wrapper_collator_function
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator

from model.translation import Sign2Text


class Seq2SeqTrainerCustom:
    def __init__(self,
                 model,
                 dataset,
                 device=DEVICE,
                 learning_rate=LEARNING_RATE):


        print('Loading model...')
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

        print('Model loaded.')
        
        print('Loading datasets..')
        self.dataset = dataset
        self.device = device
        
        print('Loading optimizer...')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    def train(self):
        
        collator_function = wrapper_collator_function(tokenizer=self.tokenizer)
        
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            output_dir=LOG_DIR,
            remove_unused_columns=False,
            num_train_epochs=NUM_EPOCHS,
            deepspeed=False,
            load_best_model_at_end=True,
        )
        
        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['val'],
            data_collator=collator_function,
        )
        
        trainer.train()

