

import re
from SignMAE.config import *
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import VivitModel, VisionEncoderDecoderModel, AutoTokenizer


class Sign2Text():
    
    def __init__(self):
        
        self.VIDEO_ENCODER = VIDEO_ENCODER_MODEL_VIDEOMAE if POSE_ENCODER_TYPE == 'videomae' else VIDEO_ENCODER_MODEL_VIVIT
        
        self.load_image_encoder()        
        
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            'posemae_base', TEXT_DECODER_MODEL
        )

        self.load_text_decoder()

        self.freeze_layers()
        
        self.model.save_pretrained(TRANSLATION_MODEL)
        
    
    def load_image_encoder(self):
    
    
        # if POSE_ENCODER_TYPE == 'videomae':

        #     model = VivitModel.from_pretrained(
        #         self.VIDEO_ENCODER,
        #     )

        #     model.config.patch_size = PATCH_SIZE
        #     model.config.num_channels = NUM_CHANNELS
        #     model.config.image_size = IMAGE_SIZE
        #     model.config.num_hidden_layers = NUM_HIDDEN_LAYERS
        #     model.config.num_attention_heads = NUM_ATTENTION_HEADS
        #     model.config.num_frames = NUM_FRAMES
        #     model.config.tubelet_size = TUBELET_SIZE
            

        if POSE_ENCODER_TYPE == 'videomae':

            print('Importing google/vivit-b-16x2')
            # Get a pretrained vivit model config
            vivit = VivitModel.from_pretrained(
                'google/vivit-b-16x2'
            )

            configuration = vivit.config
            
            # Set custom params for new vivit model
            configuration.num_channels = NUM_CHANNELS
            configuration.image_size = IMAGE_SIZE
            configuration.num_hidden_layers = NUM_HIDDEN_LAYERS
            configuration.num_attention_heads = NUM_ATTENTION_HEADS
            configuration.num_frames = NUM_FRAMES
            configuration.tubelet_size = [
                PATCH_SIZE,
                TUBELET_SIZE,
                TUBELET_SIZE
            ]

            vivit = VivitModel(
                config=configuration
            )

            vivit.save_pretrained('posemae_base')        
            
    def load_text_decoder(self):        
        
        tokenizer = AutoTokenizer.from_pretrained(TEXT_DECODER_MODEL)
        # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
        tokenizer.pad_token = tokenizer.eos_token
        # update the model config
        self.model.config.eos_token_id = tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = tokenizer.bos_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        
        tokenizer.save_pretrained(TRANSLATION_MODEL)
        
    
    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            # Open params of vivit encoder
            if name.startswith('encoder'): 
                param.requires_grad = True
                #print(name, param.requires_grad)
            if re.findall('decoder\.transformer\.[hw]\.[0-9]\.', name): 
                param.requires_grad = False    
                