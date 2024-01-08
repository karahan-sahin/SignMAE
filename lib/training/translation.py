
import sys

from tqdm import tqdm
sys.path.append(".")
sys.path.append("../..")

import os
import torch
from lib.config import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset.dataloader import wrapper_collator_function
from transformers import AutoTokenizer

from model.translation import Sign2Text


class Seq2SeqTrainer:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 device=DEVICE,
                 learning_rate=LEARNING_RATE):


        print('Loading model...')
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)

        print('Model loaded.')
        
        print('Loading datasets..')
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        print('Loading optimizer...')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    def train(self, epochs):
        
        collator_function = wrapper_collator_function(tokenizer=self.tokenizer)
        
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=BATCH_SIZE, 
            collate_fn=collator_function, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=BATCH_SIZE, 
            collate_fn=collator_function
        )

        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (batch) in tqdm(enumerate(train_loader)):
                
                print(batch)
                
                self.optimizer.zero_grad()

                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if batch_idx % 10 == 9:  # Print every 10 batches
                    with open(LOG_DIR, 'a+', encoding='utf-8') as f_out:
                        f_out.write(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx + 1}/{len(self.train_dataloader)}] Loss: {running_loss / 10:.4f}\n")

            print(f'Epoch {epoch+1}/{epochs} - Training Loss: {loss.item()}')
            self.evaluate(val_loader)

    def evaluate(self, 
                 val_loader):
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_loss}')

