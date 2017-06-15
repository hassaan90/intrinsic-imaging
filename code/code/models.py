# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:12:01 2016

@author: sergi
"""
from __future__ import absolute_import

import os
import h5py
import numpy as np 
import keras.models as models
import cv2
import time
import threading

import theano
import theano.tensor as T

from keras.layers.advanced_activations import PReLU, LeakyReLU

from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution3D, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Layer, Permute
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from theano.compile.nanguardmode import NanGuardMode
from keras.preprocessing.image import ImageDataGenerator
THEANO_FLAGS=mode=NanGuardMode
np.random.seed(1337)
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential

from keras.optimizers import Adam
from theano.tensor.signal.conv import conv2d

#RESNET    
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot

##############################################
# BEGIN OBJECTIVE FUNCTIONS
##############################################

from keras import backend as K

smooth = 1.0
def mean_length_error(y_true, y_pred):
    y_true_f = K.sum(K.round(K.flatten(y_true)))
    y_pred_f = K.sum(K.round(K.flatten(y_pred)))
    delta = (y_pred_f - y_true_f)
    return K.mean(K.tanh(delta))

    
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
 
def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)

    
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#Scale invariant L2 loss
def scale_loss(y_true, y_pred):
    y = (y_true - y_pred)
    l = 0.5
    term1 = K.mean(K.square(y))
    term2 = K.square(K.mean(y))
    return (term1-l*term2)

    
delta = 0.1
def huber(target, output):
    d = target - output
    a = .5 * d**2
    b = delta * (abs(d) - delta / 2.)
    l = T.switch(abs(d) <= delta, a, b)
    return l.sum()

    
#https://github.com/titu1994/Neural-Style-Transfer/blob/master/Network.py

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    channel_dim = 0 if K.image_dim_ordering() == "th" else -1

    channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    #if args.content_loss_type == 1:
    multiplier = 1 / (2. * channels ** 0.5 * size ** 0.5)
    #elif args.content_loss_type == 2:
    #multiplier = 1 / (channels * size)
    #else:
    #multiplier = 1.

    return multiplier * K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def custom_loss(y_true, y_pred):
    '''
    loss = K.variable(0.)
    layer_features = outputs_dict[args.content_layer]  # 'conv5_2' or 'conv4_2'
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)
    '''
    y = (y_true - y_pred)
    return total_variation_loss(y)
    

# https://github.com/fchollet/keras/issues/4292
# https://github.com/farizrahman4u/keras-contrib/tree/master/keras_contrib
# DSSIMObjective
def loss_DSSIM_theano(y_true, y_pred):
    # expected net output is of shape [batch_size, row, col, image_channels]
    # e.g. [10, 480, 640, 3] for a batch of 10 640x480 RGB images
    # We need to shuffle this to [Batch_size, image_channels, row, col]
    y_true = y_true.dimshuffle([0, 3, 1, 2])
    y_pred = y_pred.dimshuffle([0, 3, 1, 2])
    
    
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    #   and cannot be used for learning
    patches_true = T.nnet.neighbours.images2neibs(y_true, [4, 4])
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4, 4])

    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    
    ssim /= K.clip(denom, K.epsilon(), np.inf)
    #ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    
    return K.mean((1.0 - ssim) / 2.0)
    
'''
def loss_DSSIM_theano(y_true, y_pred):
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    #   and cannot be used for learning
    
    patches_true = T.nnet.neighbours.images2neibs(y_true, [4,4])
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4,4])
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    eps = 1e-9
    std_true = K.sqrt(var_true+eps)
    std_pred = K.sqrt(var_pred+eps)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom #no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)
'''  
    
##############################################
# END CUSTOM OBJECTIVE FUNCTIONS
##############################################


##############################################
# BEGIN UNET
##############################################

#CURRENT UNET 
def unet(shp=(3,224,224)):
    inputs = Input((shp))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(3, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss='mse') 
    
    return model

    
##############################################
# END UNET
##############################################


##############################################
# BEGIN INCEPTION UNET
##############################################

# https://github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/u_model.py
from keras.layers import Lambda   
    
def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(_input)

    return merge([shortcut, residual], mode="sum")


def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    
    c1_1 = Convolution2D(depth/4, 1, 1, init='he_normal', border_mode='same')(inputs)
    
    c2_1 = Convolution2D(depth/8*3, 1, 1, init='he_normal', border_mode='same')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Convolution2D(depth/2, 1, 3, init='he_normal', border_mode='same')(c2_1)
        c2_2 = BatchNormalization(mode=batch_mode, axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Convolution2D(depth/2, 3, 1, init='he_normal', border_mode='same')(c2_2)
    else:
        c2_3 = Convolution2D(depth/2, 3, 3, init='he_normal', border_mode='same')(c2_1)
    
    c3_1 = Convolution2D(depth/16, 1, 1, init='he_normal', border_mode='same')(inputs)
    #missed batch norm
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Convolution2D(depth/8, 1, 5, init='he_normal', border_mode='same')(c3_1)
        c3_2 = BatchNormalization(mode=batch_mode, axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Convolution2D(depth/8, 5, 1, init='he_normal', border_mode='same')(c3_2)
    else:
        c3_3 = Convolution2D(depth/8, 5, 5, init='he_normal', border_mode='same')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode='same')(inputs)
    c4_2 = Convolution2D(depth/8, 1, 1, init='he_normal', border_mode='same')(p4_1)
    
    res = merge([c1_1, c2_3, c3_3, c4_2], mode='concat', concat_axis=1)
    res = BatchNormalization(mode=batch_mode, axis=1)(res)
    res = actv()(res)
    return res
    

def rblock(inputs, num, depth, scale=0.1):    
    residual = Convolution2D(depth, num, num, border_mode='same')(inputs)
    residual = BatchNormalization(mode=2, axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res) 
    

def NConvolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    def f(_input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              border_mode=border_mode)(_input)
        norm = BatchNormalization(mode=2, axis=1)(conv)
        return ELU()(norm)

    return f
    
#IMG_ROWS, IMG_COLS = 80, 112 
def get_unet_inception_2head(shp=(3,80,112), weights_path=''):
    splitted = True
    act = 'elu'
    
    inputs = Input(shp, name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
    #conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)
    
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    #conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)
    
    #
    #pre = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid')(conv5)
    #pre = Flatten()(pre)
    #aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre) 
    #
    
    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)
    
    after_conv3 = rblock(conv3, 1, 128)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)
    
    after_conv2 = rblock(conv2, 1, 64)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1) # Peta aqui amb 109x256 [(None, 128, 56, 128), (None, 64, 55, 128)]
    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)
    
    after_conv1 = rblock(conv1, 1, 32)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
    #conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid', name='main_output')(conv9)
    #print conv10._keras_shape

    model = Model(input=inputs, output=[conv10]) #output=[conv10, aux_out]
    
    '''
    model.compile(optimizer=Adam(lr=1e-5),
                  loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})

    '''
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(loss=new_loss, optimizer=Adam(lr=1e-5))    
    # mean_length_error

    return model
    
##############################################
# END INCEPTION UNET
##############################################


##############################################
# BEGIN HYPERCOLUMNS UNET
##############################################

def myhypercolumn(shp=(3,224,224), weights_path=''):
    splitted = True
    act = 'relu' #'elu'
    
    inputs = Input(shp, name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
    
    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)
    
    #Hypercolumns
    hc_conv5 = UpSampling2D(size=(16, 16))(conv5) #8x8, f = 16
    hc_conv4 = UpSampling2D(size=(8, 8))(conv4) #16x16, f = 8
    hc_conv3 = UpSampling2D(size=(4, 4))(conv3) #32x32, f = 4
    hc_conv2 = UpSampling2D(size=(2, 2))(conv2) #64x64, f = 2
    
    hc = merge([conv1, hc_conv2, hc_conv3, hc_conv4, hc_conv5], mode='concat', concat_axis=1) #(None, 992, 128, 128)

    #From (None, 992, 128, 128) to 3x128x128
    #hc_red_conv1 = inception_block(hc, 128, batch_mode=2, splitted=splitted, activation=act)
    #hc_red_conv2 = inception_block(hc_red_conv1, 64, batch_mode=2, splitted=splitted, activation=act)
    hc_red_zpad1 = ZeroPadding2D((1,1))(hc)
    hc_red_conv1 = Convolution2D(128, 3, 3, init='he_normal', activation=act)(hc_red_zpad1)
    hc_red_zpad2 = ZeroPadding2D((1,1))(hc_red_conv1)
    hc_red_conv2 = Convolution2D(64, 3, 3, init='he_normal', activation=act)(hc_red_zpad2)
    hc_red_conv3 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid', name='aux_output')(hc_red_conv2)
    
    #Deconvolution process
    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)
    
    after_conv3 = rblock(conv3, 1, 128)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)
    
    after_conv2 = rblock(conv2, 1, 64)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)
    
    after_conv1 = rblock(conv1, 1, 32)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)
    conv10 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid', name='main_output')(conv9)

    model = Model(input=inputs, output=[conv10, hc_red_conv3]) #output=[conv10, aux_out]
    
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    model.compile(optimizer='Adam', loss={'main_output': huber, 'aux_output': huber}, loss_weights={'main_output': 1., 'aux_output': 0.2})
    
    return model

##############################################
# END HYPERCOLUMNS UNET
##############################################