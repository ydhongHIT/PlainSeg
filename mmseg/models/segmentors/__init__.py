# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .encoder_decoder_mask2former import EncoderDecoderMask2Former
from .encoder_decoder_mask2former_aug import EncoderDecoderMask2FormerAug


__all__ = ['BaseSegmentor',  'EncoderDecoderMask2Former', 'EncoderDecoderMask2FormerAug']
