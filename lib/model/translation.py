from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from config import *
from transformers import VivitModel, VisionEncoderDecoderModel, AutoTokenizer


class Sign2Text(VisionEncoderDecoderModel):
    
    def __init__(self, 
                 config: PretrainedConfig | None = None, 
                 encoder: PreTrainedModel | None = None, 
                 decoder: PreTrainedModel | None = None):
        super().__init__(config, encoder, decoder)

        
        self.from_encoder_decoder_pretrained(
            POSE_ENCODER_MODEL, TEXT_DECODER_MODEL
        )

        self.save_pretrained(TRANSLATION_MODEL)
        self.tokenizer.save_pretrained(TRANSLATION_MODEL)
        
    
    def load_image_encoder(self):
    
        # Get a pretrained vivit model config
        vivit = VivitModel.from_pretrained(
            POSE_ENCODER_MODEL,
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
        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.decoder_start_token_id = tokenizer.bos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id
    
    def freeze_layers(self):
        for name, param in self.named_parameters():
            # Open params of vivit encoder
            if name.startswith('encoder'): param.requires_grad = True
            #if name.startswith('decoder'): param.requires_grad = False  