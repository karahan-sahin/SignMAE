import os
from config import *
from model.translation import Sign2Text
from training.translation import Seq2SeqTrainerCustom
from dataset.dataloader import ASLFingerSpellingDataset

os.environ["WANDB_DISABLED"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' 
os.environ['CUDA_VISIBLE_DEVICES']='0' 
os.environ['CUDA_LAUNCH_BLOCKING']='1'

if __name__ == '__main__':
    
    ASL = ASLFingerSpellingDataset()
    data = ASL.dataset
    

    model = Sign2Text()
    translation = model.model
    
    trainer = Seq2SeqTrainerCustom(translation, dataset=data)
    
    trainer.train()
    