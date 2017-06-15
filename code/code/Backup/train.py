# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 18:52:08 2016

@author: sergi
"""

#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from __future__ import absolute_import

import os
import numpy as np 
import cv2

import sys
import glob

import theano
import theano.tensor as T


from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from theano.compile.nanguardmode import NanGuardMode
from keras.preprocessing.image import ImageDataGenerator
THEANO_FLAGS=mode=NanGuardMode
np.random.seed(1337)

from keras import backend as K
from theano import tensor  
from theano.tensor.signal.conv import conv2d
#from keras_plus import LearningRateDecay


import custom_models as cmodel
import data as d
import augmentation as aug

'''
def closing_app():
    # delete global vars
    all = [var for var in globals() if (var[:2], var[-2:]) != ('__', '__') and var != "gc"]
    for var in all:
        del globals()[var]
    # garbage collector
    gc.collect()
    del gc.garbage[:]
'''
'''
class EarlyStoppingByLossVal(callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
'''

from skimage.measure import structural_similarity as ssim
def compute_mse(predicted_img, gt_img, ax=None):
# with ax=0 the average is performed along the row, for each column, returning an array
# with ax=1 the average is performed along the column, for each row, returning an array
# with ax=None the average is performed element-wise along the array, returning a single value    
    value = ((predicted_img - gt_img) ** 2).mean(axis=ax)
    normalized = np.divide(value, (gt_img.shape[0]*gt_img.shape[1]*gt_img.shape[2]), dtype='float32')
    return normalized
    

def compute_lmse(predicted_img, gt_img, ax=None):
# LMSE is the local mean-squared error, which is the average
# of the scale-invariant MSE errors computed on
# overlapping square windows of size 10% of the image
# along its larger dimension.
# Look for the largest dimension (80L, 112L, 3L)
    largest_dim = max(gt_img.shape[0], gt_img.shape[1], gt_img.shape[2])
    
    WINDOW_PERCENTAGE = 10
    # Computed on overlapping square windows of size 10% of the image along its larger dimension
    accum_lmse = np.zeros([int((gt_img.shape[0] - WINDOW_PERCENTAGE) * (gt_img.shape[1] - WINDOW_PERCENTAGE))])
    accum_idx = 0
    for rows in range(int(gt_img.shape[0] - WINDOW_PERCENTAGE)):
        for cols in range(int(gt_img.shape[1] - WINDOW_PERCENTAGE)):
            curr_pred_crop = predicted_img[rows:rows+WINDOW_PERCENTAGE, cols:cols+WINDOW_PERCENTAGE]
            curr_gt_crop = gt_img[rows:rows+WINDOW_PERCENTAGE, cols:cols+WINDOW_PERCENTAGE]
            accum_lmse[accum_idx] = compute_mse(curr_pred_crop, curr_gt_crop)
            accum_idx = accum_idx + 1
    return accum_lmse.mean()


def compute_dssim(predicted_img, gt_img):
# DSSIM is the dissimilarity version of the structural similarity
# index (SSIM), defined as 1−SSIM2. SSIM characterizes
# image similarity as perceived by human observers.
# It combines errors from independent aspects
# of luminance, contrast, and structure, which are captured
# by mean, variance, and covariance of patches.
    gt_bw = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)
    predicted_bw = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    
    dynamic_range_min = min(gt_bw.min(),predicted_bw.min())
    dynamic_range_max = max(gt_bw.max(),predicted_bw.max())
    s = ssim(gt_bw, predicted_bw,
                      dynamic_range=dynamic_range_max - dynamic_range_min)
                      
    dssim = np.divide(1-s,2)    
    return dssim


    

if __name__ == '__main__':
    img_rows = 80 #80*2 #436/4 #128 #224 #109
    img_cols = 112 #112*2 #1024/4 #128 #224 #256
    color_type = 3
    
    gen_data = 1
    unidimensional = 0
    
    fit = 0
    augmentation = 0
    
    predicting = 0
    
    if gen_data:
        # Generate data    
        X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = d.prepare_data(img_rows, img_cols, color_type, False, False, True)
        '''        
        del X
        del y
        '''
        del X_val
        del y_val

        del Xa
        del ya
        del Xa_val
        del ya_val
        
        if unidimensional:
            #Transform y to a 1D vector
            y = np.array(np.reshape(y,(y.shape[0],color_type*img_rows*img_cols)))
            y_val = np.array(np.reshape(y_val,(y_val.shape[0],color_type*img_rows*img_cols)))
            
    # img = 440, val= 450
    #print ('-- Images: '+str(y.shape[0])+', val. Images: '+str(y_val.shape[0]) +'.')
    
    
    print ('-- Building the model...') 
    # VGG    
    #model = cmodel.VGG_16(X[0].shape)
    model = cmodel.myhypercolumn(X[0].shape, '../output/weights/checkpoint_263it_w1-02.hdf5')
    
    # Inpainting
    #model = cmodel.inpainting(X[0].shape)
    
    # RESNET 'weights/relu_res50_ep3000_best.hdf5')#, 'weights/sigmoid_rmse_01_1000.hdf5')
    #model = cmodel.resnet(X[0].shape, 18)
    
    #model = cmodel.chorrinet(X[0].shape)
    #model = cmodel.unet(X[0].shape)
      
    # SEGNET , 'weights/first_checkpoint681.hdf5')
    #model = cmodel.segnet(X[0].shape)
      
    # INCEPTION UNET
    #model = cmodel.get_unet_inception_2head(X[0].shape, 'weights/checkpoint_data-aug_unet_huber_286.hdf5') #'weights/unet_dice_1000.hdf5') #2100 unet_ep500.hdf5')
    #model = cmodel.get_unet_inception_2head(Xa[0].shape)#, 'weights/checkpoint_scaleloss_aug.hdf5' )
        
    if fit:
        print ('-- Fitting the model...')
        
        if not os.path.isfile('../output/weights') and not os.path.isdir('../output/weights'):
            os.mkdir('../output/weights')
        kfold_weights_path = os.path.join('../output/weights', 'checkpoint' + '.hdf5' )
        callbacks = [
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]
        
        if augmentation:
            print ('-- Using real-time data augmentation...')
            # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
            # this will do preprocessing and realtime data augmentation
            datagen = aug.ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.8,
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,  # randomly flip images
                fill_mode='nearest')
                
            # Make the model learn using the image generator
            model.fit_generator(datagen.flow(Xa, ya, batch_size=11),
                                samples_per_epoch=len(Xa),
                                nb_epoch=10, 
                                validation_data=(Xa_val, ya_val),
                                callbacks=callbacks,
                                verbose=1)
        else:  
            #model.fit(Xa, ya, batch_size=40, nb_epoch=50, verbose=1, validation_data=(Xa_val, ya_val), callbacks=callbacks)
            #model.fit(X, y, batch_size=4, nb_epoch=100, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)
        
            # 1120/2 = 560
            # 10,14,16,20,28,35,40,56,70,80,112,140,280,560,
            model.fit({'main_input': X, 'aux_input': X},
                      {'main_output': y, 'aux_output': y}, validation_split=0.5,
                      nb_epoch=500, batch_size=10, callbacks=callbacks)
        
        print ('-- Saving weights...')
        oname = os.path.join('../output/weights', 'weights.hdf5') #nb_epoch
        model.save_weights(oname)
        
    if predicting:
        del X
        del y
        X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = d.prepare_data(img_rows, img_cols, color_type, True, False, True)
        del X_val
        del y_val
        del Xa
        del ya
        del Xa_val
        del ya_val
        
        num_outputs = len(X)
        print ('-- Predicting...')
        general_predictions = model.predict(np.array(X[0:num_outputs]))
        predictions = general_predictions[0]
        
        result = (predictions - predictions.min()) / (predictions.max() - predictions.min()) #Cal?
                     
        #print ('-- Saving predictions...')
        if unidimensional:
            result = np.array(np.reshape(predictions,(y.shape[0],color_type,img_rows,img_cols)))
        
        for idx in range(num_outputs): #y_val.shape[0]
            #idx = 0
                        
            output_img = result[idx]
            if unidimensional:
                ib = np.array(output_img[0], dtype=np.float64)
                ig = np.array(output_img[1], dtype=np.float64)
                ir = np.array(output_img[2], dtype=np.float64)
                pack = np.array([ib,ig,ir]).transpose(1,2,0)
            else:
                pack = predictions[idx].transpose(1,2,0)
            oname = os.path.join('../output/predictions', str(idx)+'_reg.png')
            cv2.imwrite(oname, pack*255)
            
            resulty = np.array(np.reshape(y,(y.shape[0],color_type,img_rows,img_cols)))
            output_img = resulty[idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join('../output/predictions', str(idx)+'_y.png')
            cv2.imwrite(oname, pack*255) 
            
            output_img = X[idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join('../output/predictions', str(idx)+'_oX.png')
            cv2.imwrite(oname, pack*255)
            
        
        folders = {}
        folders["predictions"]   = '../output/predictions/'
        prefixes = ['oX','y','reg']
        
        folder = folders["predictions"]
        framesFiles = sorted(glob.glob(folder + '*reg.png'))
        nFrames = len(framesFiles)
        mseList = np.zeros([nFrames])
        lmseList = np.zeros([nFrames])
        dssimList = np.zeros([nFrames])
        
        for file_number in range (nFrames):
            # For each frame, get the current prefix
            regFile = framesFiles[file_number]
            xFile = regFile.replace(prefixes[2], prefixes[0])
            yFile = regFile.replace(prefixes[2], prefixes[1])
            
            # Load the images
            predicted_img = np.array(cv2.imread(regFile), dtype='uint8')
            gt_img = np.array(cv2.imread(yFile), dtype='uint8')
            
            # Compute MSE
            mse = compute_mse(predicted_img, gt_img)
            
            # Compute LMSE
            # lmse = compute_lmse(predicted_img, gt_img)
            
            # Compute DSSIM
            dssim = compute_dssim(predicted_img, gt_img)
            
            # Store it
            mseList[file_number] = mse
            # lmseList[file_number] = lmse
            dssimList[file_number] = dssim
            
        print(mseList.mean())
        # print(lmseList.mean())
        print(dssimList.mean())
            
            
            
            