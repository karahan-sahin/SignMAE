from config import *
from model.posemae import Sign2Gloss

from training.classification import ClassificationTrainer
from dataset.dataloader import PoseTopologyDataset, BosphorusSign22kDataset, DataLoader, collate_fn_pose_enc


def main():

    print('Loading dataset...')
    BSIGN = BosphorusSign22kDataset(DATASET_FILE_PATH='data/bsign22k/BosphorusSign22k.csv')
    print('Loaded dataset...')

    training_labels = BSIGN.train_labels.tolist()
    training_data = PoseTopologyDataset(BSIGN.train_videos, training_labels, 'datasets/train')
    train_dataloader = DataLoader(
        training_data,
        collate_fn=collate_fn_pose_enc, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    testing_labels = BSIGN.test_labels.tolist()
    testing_data = PoseTopologyDataset(BSIGN.test_videos, testing_labels, 'datasets/test')
    test_dataloader = DataLoader(
        testing_data,
        collate_fn=collate_fn_pose_enc, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    model = Sign2Gloss()
    gloss_model = model.model
    
    trainer = ClassificationTrainer(gloss_model,
                                    train_dataloader,
                                    test_dataloader)
    
    trainer.train()
    
    trainer.model.save_pretrained(f'models/posemae_base-bsign22k-{DATASET_TYPE}-{TASK_TYPE}-{NUM_EPOCHS}-{BATCH_SIZE}-{TUBELET_SIZE}-{NUM_FRAMES}-{PATCH_SIZE}')
    


if __name__ == "__main__":
    main()
