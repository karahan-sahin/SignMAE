
import os

import torch
from config import *
from transformers import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import collator_function

from model.translation import Sign2Text


class Seq2SeqTrainer:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 device='cpu',
                 learning_rate=LEARNING_RATE):


        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate
        )

    def train(self, epochs, batch_size):
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            collate_fn=collator_function, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            collate_fn=collator_function
        )

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad()

                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

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


if __name__ == '__main__':
    
    # LOAD DATASET
    ds_sign = load_dataset('bsign', split='train[:10%]')
    
    # LOAD MODEL
    model = Sign2Text()
    model.load_image_encoder()
    model.load_text_decoder()
    model.freeze_layers()
    
    # TRAIN MODEL
    trainer_torch = Seq2SeqTrainer(
        model,
        train_dataset=ds_sign['train'],
        val_dataset=ds_sign['val'],
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    trainer_torch.train(
        epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE
    )