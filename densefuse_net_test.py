# DenseFuse Network
# Encoder -> Addition/L1-norm -> Decoder

import tensorflow as tf

from encoder_test import Encoder
from decoder import Decoder
from fusion_addition import Strategy

class DenseFuseNet(object):

    def __init__(self, model_pre_path):
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)

    def transform_addition(self, img1, img2):
        # encode image
        enc_1_1,enc_2_1, enc_3_1 = self.encoder.encode(img1)
        enc_1_2, enc_2_2, enc_3_2 = self.encoder.encode(img2)
        target_features1 = Strategy(enc_1_1, enc_1_2)
        target_features2 = Strategy(enc_2_1, enc_2_2)
        target_features3 = Strategy(enc_3_1, enc_3_2)
        # target_features = enc_c
        temp_add = tf.concat([target_features1, target_features2], 3)
        target_features = tf.concat([temp_add, target_features3], 3)
        self.target_features = target_features
        print('target_features:', target_features.shape)
        # decode target features back to image
        generated_img = self.decoder.decode(target_features)
        return generated_img

    def transform_recons(self, img):
        # encode image
        enc = self.encoder.encode(img)
        target_features = enc
        self.target_features = target_features
        generated_img = self.decoder.decode(target_features)
        return generated_img


    def transform_encoder(self, img):
        # encode image
        enc = self.encoder.encode(img)
        return enc

    def transform_decoder(self, feature):
        # decode image
        generated_img = self.decoder.decode(feature)
        return generated_img

