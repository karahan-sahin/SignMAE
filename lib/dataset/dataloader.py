import sys
sys.path.append(".")
sys.path.append("../..")

import os
import math
import torch
import numpy as np
import pandas as pd

from lib.config import *
from lib.utils.math import extractPoseGraphs

from datasets import DatasetDict
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as TransformerDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence as torch_pad


# BosphorusSign22k Dataset

def collate_fn_pose_enc(batch):
    
    if NUM_CHANNELS == 1:
    
        videos = [item['image'][0] for item in batch]
        
        videos = torch_pad(videos, batch_first=True, padding_value=0.00).float()

        return videos[:,:,None,:,:], torch.tensor([item['label'] for item in batch])
    
    elif NUM_CHANNELS == 2:
        
        videos = [item['image'][0].transpose(1,3) for item in batch]
        
        videos = torch_pad(videos, batch_first=True, padding_value=0.00).float()

        return videos, torch.tensor([item['label'] for item in batch])

class PoseTopologyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, videos, labels, dataset_type='train'):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
        """
        self.dataset_type = dataset_type
        self.dataset = videos
        self.labels = labels


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        img_src = self.dataset[idx]
        targets = self.labels[idx]

        pose_embed = extractPoseGraphs([img_src], self.dataset_type, distF=DISTANCE, dim=NUM_CHANNELS)
        sample = {
            'image': pose_embed, 
            'label': targets
        }

        return sample

class BosphorusSign22kDataset(Dataset):
    
    def __init__(self, DATASET_FILE_PATH) -> None:
        super().__init__()
        
        dataset_df = pd.read_csv(DATASET_FILE_PATH)

        self.train_labels = dataset_df[dataset_df['UserID'] != 'User_4']['ClassID'].to_numpy()
        self.test_labels  = dataset_df[dataset_df['UserID'] == 'User_4']['ClassID'].to_numpy()

        train_labels_tr = dataset_df[dataset_df['UserID'] != 'User_4']['ClassName_tr'].to_numpy()

        self.label2id = {tr: int(idx) for idx, tr in zip(self.train_labels, train_labels_tr)}
        self.id2label = {int(idx): tr for idx, tr in zip(self.train_labels, train_labels_tr)}

        def getPath(x):
            class_id = ('0' * (3-math.floor(math.log10(x[0])))) + str(x[0])
            repeat_id = ('0' * (2-math.floor(math.log10(x[2])))) + str(x[2])
            return f"data/bsign22k/mmpose-full/{class_id}/{x[1]}_{repeat_id}.pickle"

        train_file_pths = dataset_df[dataset_df['UserID'] != 'User_4'][['ClassID', 'UserID', 'RepeatID']].apply(lambda x: getPath(x), axis=1).to_list()
        test_file_pths = dataset_df[dataset_df['UserID'] == 'User_4'][['ClassID', 'UserID', 'RepeatID']].apply(lambda x: getPath(x), axis=1).to_list()

        self.train_videos = pd.Series(train_file_pths)
        self.test_videos = pd.Series(test_file_pths)


# AUTSL Dataset

class AUTSLDataset(Dataset):

    def __init__(self, POSE_DATA_PATH) -> None:
        super().__init__()
        
        train_df = pd.read_csv('data/autsl/train_labels.csv', header=None, names=['path', 'labels'])
        val_df = pd.read_csv('data/autsl/val_labels.csv', header=None, names=['path', 'labels'])
        test_df = pd.read_csv('data/autsl/test_labels.csv', header=None, names=['path', 'labels'])
        
        
        self.train_labels = train_df['labels'].to_numpy()
        self.val_labels  = val_df['labels'].to_numpy()
        self.test_labels  = test_df['labels'].to_numpy()

        label_df = pd.read_csv('data/autsl/classes.csv')
        
        train_labels_tr = label_df['ClassName_tr'].to_list()
        train_labels_id = label_df['ClassID'].to_list()

        self.label2id = {tr: int(idx) for idx, tr in zip(train_labels_id, train_labels_tr)}
        self.id2label = {int(idx): tr for idx, tr in zip(train_labels_id, train_labels_tr)}

        train_file_pths = train_df['path'].apply(lambda x: f'{POSE_DATA_PATH}/train/{x}.pickle').to_numpy()
        val_file_pths  = val_df['path'].apply(lambda x: f'{POSE_DATA_PATH}/val/{x}.pickle').to_numpy()
        test_file_pths  = test_df['path'].apply(lambda x: f'{POSE_DATA_PATH}/test/{x}.pickle').to_numpy()
        
        self.train_videos = pd.Series(train_file_pths)
        self.val_videos = pd.Series(val_file_pths)
        self.test_videos = pd.Series(test_file_pths)




# ASL-FINGERSPELLING Dataset

class ASLFingerSpellingDataset():
    
    def __init__(self) -> None:
        
        asl_df = pd.read_csv('data/asl_fingerspelling/train.csv')
        
        SAMPLE = pd.read_parquet('data/asl_fingerspelling/mmpose-full/1021040628.parquet')
        SAMPLE = SAMPLE.reset_index()
        SAMPLE = SAMPLE[SAMPLE.columns[~SAMPLE.columns.str.contains('face')]]
        SAMPLE.replace(np.NaN, 0, inplace=True)
        
        x_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^x_', regex=True)]
        y_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^y_', regex=True)]
        z_cols = SAMPLE.columns[SAMPLE.columns.str.contains('^z_', regex=True)]
                
        dataset = asl_df[asl_df.sequence_id.isin(SAMPLE.sequence_id.unique())]
        dataset.path = dataset.path.apply(lambda x: x.split('/')[1])
        
        X_train, X_validation = train_test_split(
            dataset, test_size=0.3
        )

        X_val, X_test = train_test_split(
            X_validation, test_size=0.5
        )
        
        train_labels = X_train['phrase'].to_numpy()
        val_labels = X_val['phrase'].to_numpy()
        test_labels  = X_test['phrase'].to_numpy()

        train_file_paths = X_train[['path', 'sequence_id']].to_records(index = False)
        val_file_paths = X_val[['path', 'sequence_id']].to_records(index = False)
        test_file_paths = X_test[['path', 'sequence_id']].to_records(index = False)

        self.dataset = DatasetDict({
            'train':TransformerDataset.from_dict({
                'phrase': list(train_labels),
                'file_name': list([i[0] for i in train_file_paths]),
                'sequence_id': list([i[1] for i in train_file_paths]),
            }),
            'val': TransformerDataset.from_dict({
                'phrase': list(val_labels),
                'file_name': list([i[0] for i in val_file_paths]),
                'sequence_id': list([i[1] for i in val_file_paths]),
            }),
            'test': TransformerDataset.from_dict({
                'phrase': list(test_labels),
                'file_name': list([i[0] for i in test_file_paths]),
                'sequence_id': list([i[1] for i in test_file_paths]),
            })
        })


def collate_fn(batch):
    
    videos = torch_pad(batch, batch_first=True, padding_value=0.00).float()

    return torch.permute(videos,(0,1,4,2,3))

def wrapper_collator_function(tokenizer):
    
    from utils.util_translation import extractPoseGraphs
    
    def collator_function(batch):
        
        labels = [example['phrase'] for example in batch]
        pths = [(example['file_name'], example['sequence_id']) for example in batch]
        
        batch_dict = {
            'labels': torch.tensor(tokenization_fn(
                labels, MAX_TOKENS, tokenizer=tokenizer
            )),
            'pixel_values': collate_fn(
                extractPoseGraphs(
                    pths, 
                )
            )
        }
        
        return batch_dict

    return collator_function

def tokenization_fn(captions, max_target_length, tokenizer):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                      padding="max_length", 
                      max_length=max_target_length).input_ids

    return labels






