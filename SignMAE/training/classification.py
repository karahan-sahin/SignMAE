

import torch
from tqdm import tqdm
from SignMAE.config import *
# from transformers import AdamW


class ClassificationTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 device=DEVICE,
                 learning_rate=LEARNING_RATE):


        self.model = model.to(device)
        model = model.float()
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train(self):
        # Set optimizer and loss function
        optimizer = self.optimizer
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        # Training loop
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch_idx, (inputs, labels) in tqdm(enumerate(self.train_dataloader)):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                running_loss += loss.item()
                if batch_idx % 10 == 9:  # Print every 10 batches
                    with open(LOG_DIR, 'a+', encoding='utf-8') as f_out:
                        f_out.write(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx + 1}/{len(self.train_dataloader)}] Loss: {running_loss / 10:.4f} Accuracy: {(correct_predictions / total_predictions) * 100:.2f}%\n")
                    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx + 1}/{len(self.train_dataloader)}] Loss: {running_loss / 10:.4f} Accuracy: {(correct_predictions / total_predictions) * 100:.2f}%\n")
                    running_loss = 0.0
                    
            self.evaluate(epoch)

    
    def evaluate(self, epoch):   
        
        self.model.eval()
        
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        y_pred_5 = []

        with torch.no_grad():
            for val_inputs, val_labels in tqdm(self.val_dataloader):
                val_inputs, val_labels = val_inputs.to(DEVICE), val_labels.to(DEVICE)
                val_outputs = self.model(val_inputs)

                _, val_predicted = torch.topk(val_outputs.logits, k=5, dim=1)
                top_5_pred = torch.topk(val_outputs.logits, k=5, dim=1).indices.tolist()
                for pred_idxs, true_idx in zip(top_5_pred, val_labels.tolist()):
                    y_pred_5.extend(top_5_pred)
                
                total_samples += val_labels.size(0)
                top1_correct += (val_predicted[:, 0] == val_labels).sum().item()

                for i in range(val_labels.size(0)):
                    if val_labels[i] in val_predicted[i]:
                        top5_correct += 1
                        break

        top1_accuracy = (top1_correct / total_samples) * 100
        top5_accuracy = (top5_correct / total_samples) * 100
        
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation Top-5 Accuracy: {top5_accuracy:.2f}%")

