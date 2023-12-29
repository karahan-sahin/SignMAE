
import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoModelForVideoClassification, VideoMAEModel, VideoMAEForVideoClassification, VivitForVideoClassification
from config import *
from torch.nn.utils.rnn import pad_sequence as torch_pad
from torch.utils.data import DataLoader, Dataset


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Sign2Gloss(AutoModelForVideoClassification):

    def __init__(self, config):
        super().__init__(config)
        
        self.init()

    def init(self):

        if POSE_ENCODER_TYPE == 'vivit':

            model = VivitForVideoClassification.from_pretrained(
                POSE_ENCODER_MODEL,
                num_labels=NUM_CLASSES + 1,
                ignore_mismatched_sizes=True
                # label2id=label2id,
                # id2label=id2label,
            )

            configuration = model.config

            configuration.num_channels = NUM_CHANNELS
            configuration.image_size = 65
            configuration.num_hidden_layers = NUM_HIDDEN_LAYERS
            configuration.num_attention_heads = NUM_ATTENTION_HEADS
            configuration.num_frames = NUM_FRAMES
            configuration.tubelet_size = [
                PATCH_SIZE,
                TUBELET_SIZE,
                TUBELET_SIZE
            ]

            model = VivitForVideoClassification(
                config=model.config
            )


        if POSE_ENCODER_TYPE == 'videomae':

            model = VideoMAEForVideoClassification.from_pretrained(
                'MCG-NJU/videomae-base',
                num_labels=NUM_CLASSES + 1,
                # label2id=label2id,
                # id2label=id2label,
            )

            model.config.patch_size = PATCH_SIZE
            model.config.num_channels = NUM_CHANNELS
            model.config.image_size = 65
            model.config.num_hidden_layers = NUM_HIDDEN_LAYERS
            model.config.num_attention_heads = NUM_ATTENTION_HEADS
            model.config.num_frames = NUM_FRAMES
            model.config.tubelet_size = TUBELET_SIZE

            model = VideoMAEForVideoClassification(
                config=model.config
            )
