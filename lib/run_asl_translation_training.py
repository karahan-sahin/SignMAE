from config import *
from model.translation import Sign2Text
from training.translation import Seq2SeqTrainer
from dataset.dataloader import ASLFingerSpellingDataset

if __name__ == '__main__':
    
    ASL = ASLFingerSpellingDataset()
    data = ASL.dataset
    

    model = Sign2Text()
    translation = model.model
    
    trainer = Seq2SeqTrainer(translation,
                            train_dataset=data['train'],
                            val_dataset=data['val'])
    
    trainer.train(epochs=NUM_EPOCHS)
    