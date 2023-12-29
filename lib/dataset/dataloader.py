import torch
from config import *
from torch.utils.data import Dataset
from utils.math import extractPoseGraphs
from torch.nn.utils.rnn import pad_sequence as torch_pad


# BosphorusSign22k Dataset
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

        pose_embed = extractPoseGraphs([img_src], self.dataset_type, distF=DISTANCE, dim=NUM_CHANNELS )
        sample = {
            'image': pose_embed, 
            'label': targets
        }

        return sample

class BosphorusSign22kDataset(Dataset):
    pass

dataset_df = pd.read_csv(DATASET_FILE_PATH)

train_labels = dataset_df[dataset_df['UserID'] != 'User_4']['ClassID'].to_numpy()
test_labels  = dataset_df[dataset_df['UserID'] == 'User_4']['ClassID'].to_numpy()

train_labels_tr = dataset_df[dataset_df['UserID'] != 'User_4']['ClassName_tr'].to_numpy()
test_labels_tr  = dataset_df[dataset_df['UserID'] == 'User_4']['ClassName_tr'].to_numpy()

label2id = {tr: int(idx) for idx, tr in zip(train_labels, train_labels_tr)}
id2label = {int(idx): tr for idx, tr in zip(train_labels, train_labels_tr)}

train_file_pths = dataset_df[dataset_df['UserID'] != 'User_4'][['ClassID', 'UserID', 'RepeatID']].apply(lambda x: getPath(x), axis=1).to_list()
test_file_pths = dataset_df[dataset_df['UserID'] == 'User_4'][['ClassID', 'UserID', 'RepeatID']].apply(lambda x: getPath(x), axis=1).to_list()

train_videos = pd.Series(train_file_pths)
test_videos = pd.Series(test_file_pths)


for param in model.parameters():
    param.requires_grad = False

training_labels = train_labels.tolist()
training_data = PoseTopologyDataset(train_videos, training_labels, 'datasets/train')
train_dataloader = DataLoader(
    training_data,
    collate_fn=collate_fn, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

testing_labels = test_labels.tolist()
testing_data = PoseTopologyDataset(test_videos, testing_labels, 'datasets/test')
test_dataloader = DataLoader(
    testing_data,
    collate_fn=collate_fn, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)


def collate_fn(batch):
    
    if NUM_CHANNELS == 1:
    
        videos = [item['image'][0] for item in batch]
        
        videos = torch_pad(videos, batch_first=True, padding_value=0.00).float()

        return videos[:,:,None,:,:], torch.tensor([item['label'] for item in batch])
    
    elif NUM_CHANNELS == 2:
        
        videos = [item['image'][0].transpose(1,3) for item in batch]
        
        videos = torch_pad(videos, batch_first=True, padding_value=0.00).float()

        return videos, torch.tensor([item['label'] for item in batch])
    
    
    
    
    

# AUTSL Dataset

class AUTSLDataset(Dataset):
    pass







# ASL-FINGERSPELLING Dataset

class ASLFingerSpellingDataset(Dataset):
    pass

def collator_function(batch):
    
    labels = [example['phrase'] for example in batch]
    pths = [(example['file_name'], example['sequence_id']) for example in batch]
    
    
    return {
        'labels': torch.tensor(tokenization_fn(
           labels, 128
        )),
        'pixel_values':  collate_fn(
            extractPoseGraphs(
                pths, 
                distF='euc', 
                dim=2, 
                max_length=50 
            )
        )
    }

def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                      padding="max_length", 
                      max_length=max_target_length).input_ids

    return labels



