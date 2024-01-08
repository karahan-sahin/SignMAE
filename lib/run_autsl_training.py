from config import *
from model.posemae import Sign2Gloss

from training.classification import ClassificationTrainer
from dataset.dataloader import PoseTopologyDataset, AUTSLDataset, DataLoader, collate_fn_pose_enc


def main():

    AUTSL = AUTSLDataset(POSE_DATA_PATH='data/autsl/mmpose-full')
    print('Loaded dataset...')

    training_labels = AUTSL.train_labels.tolist()
    training_data = PoseTopologyDataset(AUTSL.train_videos, training_labels, 'datasets/train')
    train_dataloader = DataLoader(
        training_data,
        collate_fn=collate_fn_pose_enc, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    testing_labels = AUTSL.test_labels.tolist()
    testing_data = PoseTopologyDataset(AUTSL.test_videos, testing_labels, 'datasets/test')
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
    
    trainer.model.save_pretrained(f'models/posemae_base-{DATASET_TYPE}-{TASK_TYPE}-{NUM_EPOCHS}-{BATCH_SIZE}-{TUBELET_SIZE}-{NUM_FRAMES}-{PATCH_SIZE}')
    


if __name__ == "__main__":
    main()
