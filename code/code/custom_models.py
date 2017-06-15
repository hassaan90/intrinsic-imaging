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

img_width = 112*2
img_height = 80*2


#BEGIN VGG
def VGG_16(shp=(3,128,128), weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=shp))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    # PART 2
    
    import h5py
    
    weights_path = 'vgg16_weights.h5'
    
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
    model.add(Activation('sigmoid'))    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    '''
    model.add(Flatten())
    model.add(Dense(3*64*64, activation='relu')) #4096
    model.add(Dropout(0.5))
    model.add(Dense(3*64*64, activation='relu')) #4096
    model.add(Dropout(0.5))
    model.add(Dense(3*64*64, activation='softmax'))
    '''
    return model
#END VGG
    
#SEGNET
class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]
    
def segnet(shp=(3,224,224), weights_path=''):
    model = models.Sequential()
    model.add(Layer(input_shape=shp))
    #model.add(GaussianNoise(sigma=0.3))
    model.encoding_layers = create_encoding_layers()
    model.decoding_layers = create_decoding_layers()
    for l in model.encoding_layers:
        model.add(l)
    for l in model.decoding_layers:
        model.add(l) #64 x 224 x 224
    model.add(Convolution2D(3, 1, 1, border_mode='valid',)) # 3 x 224 x 224
    #model.add(Reshape((3, img_rows, img_cols)))
        
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    model.compile(loss='mse', optimizer='adam')
    #model.compile(loss=custom_objective, optimizer='adam')
    return model    
#END SEGNET

#BEGIN RESNET
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)
    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)
    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
                input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input
    return f


# CUSTOM OBJECTIVE FUNCTIONS
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



def old_custom_objective(y_true, y_pred):
    #Scale invariant L2 loss
    y = y_pred - y_true # difference
    h = 0.5 # lambda
    term1 = K.mean(K.square(y), axis=-1)
    term2 = K.square(K.mean(y, axis=-1))
    sca = term1-h*term2
    #return sca
    #Gradient L2 loss
    gra = K.mean(K.square(K.gradients(K.sum(y[:,1]), y)) + K.square(K.gradients(K.sum(y[1,:]), y)))
    return (sca + gra)

 # END CUSTOM OBJECTIVE FUNCTIONS   


# http://arxiv.org/pdf/1512.03385v1.pdf
def resnet(shp=(3,224,224), layers=18, weights_path=''):
    print ('-- ResNet, shp: ' + str(shp) +', layers: ' + str(layers) + '.')
    input = Input(shp)

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    if layers == 18:
        block_fn = _basic_block # Build residual blocks..
        block1 = _residual_block(block_fn, nb_filters=64, repetations=2, is_first_layer=True)(pool1)
        block2 = _residual_block(block_fn, nb_filters=128, repetations=2)(block1)
        block3 = _residual_block(block_fn, nb_filters=256, repetations=2)(block2)
        block4 = _residual_block(block_fn, nb_filters=512, repetations=2)(block3)

    elif layers == 34:
        block_fn = _basic_block # Build residual blocks..
        block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1) #64,32,32
        block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1) # 64,16,16
        block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2) # 64,8,8
        block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3) # 64,2,2
    
    elif layers == 50: #72 real layers
        block_fn = _bottleneck # Build residual blocks..
        block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
        block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
        block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
        block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)

    elif layers == 101:
        block_fn = _bottleneck # Build residual blocks..
        block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
        block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
        block3 = _residual_block(block_fn, nb_filters=256, repetations=23)(block2)
        block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)
        
    elif layers == 152: #512 real layers
        block_fn = _bottleneck # Build residual blocks..
        block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
        block2 = _residual_block(block_fn, nb_filters=128, repetations=8)(block1)
        block3 = _residual_block(block_fn, nb_filters=256, repetations=36)(block2)
        block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)
    
    else:
        print('-- Please, choose between [18, 34, 50, 101, 152] layers.')
    
    
    # Classifier block
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
    
    flatten1 = Flatten()(pool2)
    dense = Dense(output_dim=shp[0]*shp[1]*shp[2], init="he_normal", activation="relu")(flatten1)
    
    '''
    deconv1 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/16, factor/16), subsample=(2,2), border_mode='same')(pool2)
    level4 = merge([block3, deconv1], mode='sum')
    
    deconv2 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/8, factor/8), subsample=(2,2), border_mode='same')(level4)
    level3 = merge([block2, deconv2], mode='sum') 
    
    deconv3 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/4, factor/4), subsample=(2,2), border_mode='same')(level3)
    level2 = merge([block1, deconv3], mode='sum')
    
    deconv4 = Deconvolution2D(64, 7, 7, activation='relu', output_shape=(None, 64, factor/2, factor/2), subsample=(2,2), border_mode='same')(level2)
    level1 = merge([conv1, deconv4], mode='sum')

    deconv5 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor, factor), subsample=(2,2), border_mode='same')(level1)
    
    conv2 = Convolution2D(nb_filter=3, nb_row=1, nb_col=1, init="he_normal", border_mode="valid")(deconv5)
    '''
    #(None, 3, 2, 2)
    '''
    factor = int(shp[1])
   
    deconv1 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/16, factor/16), subsample=(2,2), border_mode='same')(pool2)    
    deconv2 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/8, factor/8), subsample=(2,2), border_mode='same')(deconv1)
    level3 = merge([block2, deconv2], mode='sum') 
    
    deconv3 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/4, factor/4), subsample=(2,2), border_mode='same')(level3)
    deconv4 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor/2, factor/2), subsample=(2,2), border_mode='same')(deconv3)
    level1 = merge([conv1, deconv4], mode='sum')

    deconv5 = Deconvolution2D(64, 3, 3, activation='relu', output_shape=(None, 64, factor, factor), subsample=(2,2), border_mode='same')(level1)
    
    conv2 = Convolution2D(nb_filter=3, nb_row=1, nb_col=1, init="he_normal", border_mode="valid")(deconv5)
    '''
    model = Model(input=input, output=dense)
    #block 4 >> (None, 512, 2, 2)
    
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    model.compile(loss='mse', optimizer='adam')    
    return model
#END RESNET
    
#BEGIN UNET  
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
#END UNET
    
#CHORRINET
def custom_Deconvolution2D(prevlayer, nb_filter=64, nb_row=3, nb_col=3, subsample=2):
    # Custom convolution to solve 'Deconvolution2D stack with different nb_filter' issue
    upsampling  = UpSampling2D(size=(subsample,subsample))(prevlayer)
    convolution = Convolution2D(nb_filter, nb_row, nb_col, activation='relu', border_mode='same')(upsampling)
    return convolution
    
def chorrinet(shp=(3,224,224)):
    
    channels = 3
    height = 224
    width = 224
    
    n1 = 64
    n2 = 64 # Compiles and fits properly if n1 == n2
    
    init = Input(shape=(channels, height, width))
    
    level1_1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(init)
    level2_1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(level1_1)
    
    level3_1 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(level2_1)
    level4_1 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(level3_1)
    
    
    level4_2 = Deconvolution2D(n2, 3, 3, activation='relu', output_shape=(None, n2, height, width), border_mode='same')(level4_1)
    level3_2 = Deconvolution2D(n2, 3, 3, activation='relu', output_shape=(None, n2, height, width), border_mode='same')(level4_2)
    level3 = merge([level3_1, level3_2], mode='sum')
    
    level2_2 = Deconvolution2D(n1, 3, 3, activation='relu', output_shape=(None, n1, height, width), border_mode='same')(level3)
    level1_2 = Deconvolution2D(n1, 3, 3, activation='relu', output_shape=(None, n1, height, width), border_mode='same')(level2_2)
    level1 = merge([level1_1, level1_2], mode='sum')
    #NOT ORIGINAL

    
    #With custom
    '''
    level4_2 = custom_Deconvolution2D(level4_1, n2, 3, 3, 2)
    level3_2 = custom_Deconvolution2D(level4_2, n2, 3, 3, 2)
    level3 = merge([level3_1, level3_2], mode='sum')
    
    level2_2 = custom_Deconvolution2D(level3, n1, 3, 3, 2)
    level1_2 = custom_Deconvolution2D(level2_2, n1, 3, 3, 2)
    level1 = merge([level1_1, level1_2], mode='sum')
    '''
    
    
    #Deconvolution2D(n2, 3, 3, activation='relu', output_shape=(None, n2, height, width), border_mode='same')(level4_1)


    
    #NOT ORIGINAL
    
    decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)
    
    model = Model(init, decoded)
    #model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss='mse')
    return model


#BEGIN INPAINTING
def inpainting(shp=(3,224,224)):
    # https://github.com/fchollet/keras/issues/4050
    from keras.models import Model
    from keras.layers import Input, merge

    first_input = Input(batch_shape=(None,3,224,224))
    
    conv0_1_3 = Convolution2D(3, 3, 3, activation='relu', name='conv0_1_3', border_mode='same')(first_input)
    
    conv1_1_64 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1', border_mode='same')(conv0_1_3)
    conv1_2_64 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same')(conv1_1_64)
    conv1_2_64 = MaxPooling2D((2, 2))(conv1_2_64)
    
    conv2_1_128 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same')(conv1_2_64)
    conv2_2_128 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same')(conv2_1_128)
    conv2_2_128 = MaxPooling2D((2, 2))(conv2_2_128)
    
    conv3_1_256 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same')(conv2_2_128)
    conv3_2_256 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same')(conv3_1_256)
    conv3_3_256 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same')(conv3_2_256)
    conv3_3_256 = MaxPooling2D((2, 2))(conv3_3_256)
    
    conv4_1_512 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same')(conv3_3_256)
    conv4_2_512 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same')(conv4_1_512)
    conv4_3_512 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same')(conv4_2_512)
    conv4_3_512 = MaxPooling2D((2, 2))(conv4_3_512)
    
    residual1 = BatchNormalization(axis=1, name='batch1')(conv4_3_512)
    residual1 = Convolution2D(256, 3, 3, activation='relu', name='residual1', border_mode='same')(residual1)
    residual1 = UpSampling2D(name='upsample1')(residual1)
    
    conv3_3_256_batch_norm = BatchNormalization(axis=1, name='batch2')(conv3_3_256)
    merge1 = merge((conv3_3_256_batch_norm, residual1), mode='concat', name='merge1', concat_axis=1)
    residual2 = Convolution2D(128, 3, 3, activation='relu', name='residual2', border_mode='same')(merge1)
    residual2 = UpSampling2D(name='upsample2')(residual2)
    
    conv2_2_128_batch_norm = BatchNormalization(axis=1, name='batch3')(conv2_2_128)
    merge2 = merge((conv2_2_128_batch_norm, residual2), mode='concat', name='merge2', concat_axis=1)
    residual3 = Convolution2D(64, 3, 3, activation='relu', name='residual3', border_mode='same')(merge2)
    residual3 = UpSampling2D(name='upsample3')(residual3)
    
    conv1_2_64_batch_norm = BatchNormalization(axis=1, name='batch4')(conv1_2_64)
    merge3 = merge((conv1_2_64_batch_norm, residual3), mode='concat', name='merge3', concat_axis=1)
    residual4 = Convolution2D(3, 3, 3, activation='relu', name='residual4', border_mode='same')(merge3)
    residual4 = UpSampling2D(name='upsample4')(residual4)
    
    conv0_1_3_batch_norm = BatchNormalization(axis=1, name='batch5')(conv0_1_3)
    merge4 = merge((conv0_1_3_batch_norm, residual4), mode='concat', name='merge4', concat_axis=1)
    residual5 = Convolution2D(3, 1, 1, activation='relu', name='residual5', border_mode='same')(merge4)
    
    model = Model(input=first_input, output=residual5)
    
    model.compile(loss='mean_squared_error', optimizer='adam')    
    
    return model
#END INPAINTING

# SEE WHAT CNN IS FOCUSED ON
# https://github.com/tdeboissiere/VGG16CAM-keras/blob/master/VGGCAM-keras.py
import matplotlib.pylab as plt
import theano.tensor.nnet.abstract_conv as absconv
from keras.layers.advanced_activations import ELU

def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):

    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 224 * 224))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])
    


def show_cnn_focus(im, model, label, nb_classes, batch_size, num_input_channels, ratio):
    # Get a copy of the original image
    im_ori = im.copy().astype(np.uint8)
    classmap = get_classmap(model,
                            im.reshape(1, 3, 224, 224),
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    plt.imshow(im_ori)
    plt.imshow(classmap[0, label, :, :],
               cmap="jet",
               alpha=0.5,
               interpolation='nearest')
    plt.show()
# END
    
    
# PRO UNET!
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

def BNA(_input):
    inputs_norm = BatchNormalization(mode=2, axis=1)(_input)
    return ELU()(inputs_norm)

def reduction_a(inputs, k=64, l=64, m=96, n=96):
    "35x35 -> 17x17"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
    
    conv2 = Convolution2D(n, 3, 3, subsample=(2,2), border_mode='same')(inputs_norm)
    
    conv3_1 = NConvolution2D(k, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv3_2 = NConvolution2D(l, 3, 3, subsample=(1,1), border_mode='same')(conv3_1)
    conv3_2 = Convolution2D(m, 3, 3, subsample=(2,2), border_mode='same')(conv3_2)
    
    res = merge([pool1, conv2, conv3_2], mode='concat', concat_axis=1)
    return res


def reduction_b(inputs):
    "17x17 -> 8x8"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
    #
    conv2_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv2_2 = Convolution2D(96, 3, 3, subsample=(2,2), border_mode='same')(conv2_1)
    #
    conv3_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv3_2 = Convolution2D(72, 3, 3, subsample=(2,2), border_mode='same')(conv3_1)
    #
    conv4_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv4_2 = NConvolution2D(72, 3, 3, subsample=(1,1), border_mode='same')(conv4_1)
    conv4_3 = Convolution2D(80, 3, 3, subsample=(2,2), border_mode='same')(conv4_2)
    #
    res = merge([pool1, conv2_2, conv3_2, conv4_3], mode='concat', concat_axis=1)
    return res

  


def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

def make_kernel(sigma):
    # kernel radius = 2*sigma, but minimum 3x3 matrix
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
    # make 2D kernel
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize kernel by sum of elements
    kernel = np_kernel / np.sum(np_kernel)
    return kernel

def conv_loss(y_true, y_pred):
    '''
    from theano.tensor.signal.conv import conv2d
    image = K.square(y_true - y_pred)
    if image.ndim > 3:
        image = K.squeeze(image, axis=1) #Remove 1 dim
    c = conv2d(input=image, filters=make_kernel(sigma=1.0), border_mode='full')
    return K.mean(c, axis=-1)
    '''
    image = K.square(y_true - y_pred)
    
    if image.ndim > 3:
        image = K.squeeze(image, axis=0) #Remove 1 dim
        
    rng = np.random.RandomState(42)
    #conv_filter = make_kernel(1.0) # 
    conv_filter = rng.randn(3).astype(np.float64)
    
    img = T.dtensor4()
    fil = T.dtensor4()
    
    for border_mode in ["full", "valid"]:
        theano_convolve2d = theano.function([img, fil], T.nnet.conv2d(img, fil,
                                            border_mode=border_mode))
        theano_convolved = theano_convolve2d(image.reshape(3,128, 128),
                                             conv_filter.reshape(3,1,len(conv_filter)))
    return K.mean(theano_convolved)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def kullback_leibler2(y_pred,y_true):
    eps = 0.0001
    results, updates = theano.scan(lambda y_true,y_pred: (y_true+eps)*(T.log(y_true+eps)-T.log(y_pred+eps)), 
    sequences = [y_true,y_pred])
    return T.sum(results, axis= - 1)


def cconv(image, kernel, batch_size, osize):
    g_kernel = theano.shared(kernel)
    '''
    M = T.dtensor3()
    conv = theano.function(
        inputs=[M],
        outputs=conv2d(M, g_kernel, border_mode='full'),
    )
    #conv2d(image[0],g_kernel)
    '''
    accum = 0
    for curr_batch in range (10): #0 a 9
        accum = accum + conv2d(image[curr_batch], g_kernel, border_mode='full')
    accum = accum/batch_size
    return accum[:, :osize, :osize]
    
    
def scale_gradient(y_true, y_pred):
    image_with_and_height = 128
    batch_size = 40
    
    #Approach
    log = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1], [0,1,2,1,0], [0,0,1,0,0]]).astype(np.float32)
    g_true = cconv(y_true, log, batch_size, image_with_and_height)
    g_pred = cconv(y_pred, log, batch_size, image_with_and_height)
    
    return huber(g_true, g_pred)
    
    '''
    y = (y_true - y_pred)
    
    #Scale invariant L2 loss
    l = 0.5
    term1 = K.mean(K.square(y))
    term2 = K.square(K.mean(y))
    sca_inv_l2_loss = term1-l*term2
    
    # Gradient l2 loss
    # Direction x
    #pw_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).astype(np.float32)
    so_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).astype(np.float32)
    g_x = cconv(y, so_x, batch_size, image_with_and_height)
    # Direction y
    #pw_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]).astype(np.float32)
    so_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).astype(np.float32)
    g_y = cconv(y, so_y, batch_size, image_with_and_height)
    
    gra_l2_loss = K.mean(K.square(g_x) + K.square(g_y))
    
    return (sca_inv_l2_loss + gra_l2_loss)
    '''

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
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
    
    
def new_loss(y_true, y_pred):
    return contrastive_loss(y_true, y_pred)/2 + huber(y_true, y_pred)/2
    
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
#END PRO UNET

#myhypercolumn
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
#END myhypercolumn
    
#myhypercolumn2
def weighted_hypercolumn(shp=(3,224,224), weights_path=''):
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
    
    out = merge([conv10, hc_red_conv3], mode='ave', concat_axis=1)

    model = Model(input=inputs, output=[out])
    
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    model.compile(optimizer='Adam', loss='mse')
    
    return model
#END myhypercolumn