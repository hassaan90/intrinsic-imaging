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
import time
import sys
import glob

import theano
import theano.tensor as T

import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, Progbar, ProgbarLogger, EarlyStopping
from keras.callbacks import LambdaCallback
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from theano.compile.nanguardmode import NanGuardMode
from keras.preprocessing.image import ImageDataGenerator
THEANO_FLAGS=mode=NanGuardMode
np.random.seed(1337)
from tempfile import TemporaryFile
from keras import backend as K
from theano import tensor  

from theano.tensor.signal.conv import conv2d
#from keras_plus import LearningRateDecay

import time
import segnet as smodel
import data as d
import augmentation as aug
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools as it

outfile = TemporaryFile()
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


class Histories(keras.callbacks.Callback):

    def __init__(self):
        super(Histories, self).__init__()
        self.directory = '../output/prediction'
        self.count = 0
        #self.count1 = 0
        self.nepoch = 0
    


    def on_batch_end(self, batch, logs={}):
            
            self.losses.append(logs.get('loss'))
            #print self.losses[0]
            if self.count >49:
                self.count = 0
            y_pred = self.model.predict(self.model.validation_data[0][self.count:self.count+1,:,:,:])
            #print y_pred.shape
            #print type(y_pred)
            y_pred = (y_pred - np.min(y_pred))
            
            y_pred = y_pred / np.max(y_pred)
            y_pred = y_pred[0]
            y_gt = model.validation_data[1][self.count]
            y_pred = y_pred.transpose(1,2,0)
            y_gt = y_gt.transpose(1,2,0)
            result = np.concatenate((y_gt,y_pred ), axis=1)
            #print np.max(y_pred), np.min(y_pred), np.max(y_gt), np.min(y_gt)
            #plt.imshow(result)
            #plt.show()
            #pause(0.01)
            oname = os.path.join(self.directory, str(self.count)+'_'+str(self.nepoch)+'_'+str(batch)+'.jpg')
            #print oname
            time.sleep(5)
            self.count += 1
            cv2.imwrite(oname, result*255)

            #plt.imshow(result)
            #cv2.imwrite('gt.jpg'+str(Histories.count)+'.jpg', y_gt)
            return
    def on_epoch_begin(self, epoch, logs={}):
            self.nepoch = epoch
            return

    def on_train_begin(self, logs={}):
            self.losses = []
            return

            
            
            
    
    
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

def unison_shuffled_copies(a, b):
    '''
    X_val,y_val = unison_shuffled_copies(X_val,y_val)
    img_x = X_val[10].transpose(1,2,0)
    cv2.imwrite('X.jpg',img_x)
    img_y = y_val[10].transpose(1,2,0)
    cv2.imwrite('y.jpg',img_y)
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #352 -> array([1, 5, 8, 6])
    return a[p], b[p]
  
    

if __name__ == '__main__':
    # MemoryError: Error allocating 599851008 bytes 
    # AR: 1024/436 = 2.3486
    # AR: 224/96 = 2.3333 -> 896 / 384
    # P. Values: 16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784    
    img_rows = 160
    img_cols = 224
    color_type = 3
    
    # Batch size 32 i res:80x144
    
    gen_data = 1
    
    fit = 1
    augmentation = 0
    manual_augmentation = True
    
    predicting = 0
    
    if gen_data:
        # Generate data    
        X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = d.new_prepare_data(img_rows, img_cols, color_type, False, manual_augmentation, False)
        '''        
    del X
    del y
        
    del X_val
    del y_val
    '''
        
        del Xa_val
        del ya_val
        if manual_augmentation:
            #Add Xa to X and ya to y
            X = np.append(X,Xa, axis=0)
            y = np.append(y,ya, axis=0)
            # Randomize
            X, y = unison_shuffled_copies(X, y)
        
        del Xa
        del ya
        

            
    # img = 440, val= 450
    #print ('-- Images: '+str(y.shape[0])+', val. Images: '+str(y_val.shape[0]) +'.')
    
    
    print ('-- Building the model...') 
    #model = cmodel.myhypercolumn(X[0].shape, '../output/weights/checkpoint_263it_w1-02.hdf5')
    #model = cmodel.weighted_hypercolumn(X[0].shape, '') # '../output/weights/normal_checkpoint.hdf5')
    model = smodel.build_segnet(img_shape=(3, 160, 224), n_classes=3, l2_reg=0.,
                 init='he_normal', path_weights=None,
                 freeze_layers_from=None, use_unpool=False, basic=True)
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    #model.compile(loss=lambda y,f: tilted_loss(0.5,y,f), optimizer='adagrad')
    if fit:
        if not os.path.isfile('../output/weights') and not os.path.isdir('../output/weights'):
            os.mkdir('../output/weights')
        
        kfold_weights_path = os.path.join('../output/weights', 'checkpoint' + '.hdf5' )
        plot_loss_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))
        my_callback = Histories()
        #Early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        callbacks_list = [
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0), my_callback 
        ]
        if augmentation:
            print ('-- Using real-time data augmentation...')
            print ('samples: '+ str(len(X)))
            '''
            # https://keras.io/preprocessing/image/
            # we create two instances with the same arguments
            data_gen_args = dict(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=15.0,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rescale=1./255, #If data is normalized
                                 zoom_range=0.6,
                                 horizontal_flip=True)
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            
            # Provide the same seed and keyword arguments to the fit and flow methods
            seed = 1
            image_datagen.fit(X, augment=True, seed=seed)
            mask_datagen.fit(y, augment=True, seed=seed)
            
            image_generator = image_datagen.flow(X) batch_size=16)
            mask_generator = mask_datagen.flow(y) batch_size=16)
            
            # combine generators into one which yields image and masks
            train_generator = it.izip(image_generator, mask_generator)
            
            print ('-- Fitting the model...')
            # Exception: The model expects 2 input arrays, but only received one array. Found: array with shape (32, 3, 80, 112)
            # TO DO: ADD CROSS VALIDATION METHOD  
            model.fit_generator(
                train_generator,
                samples_per_epoch=len(X),
                validation_data = (X_val, y_val),
                nb_epoch=1000, callbacks=callbacks)
            '''
            # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
            # this will do preprocessing and realtime data augmentation
            datagen = aug.ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15.0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.8,
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,  # randomly flip images
                fill_mode='nearest')
                
            # Make the model learn using the image generator
            history = model.fit_generator(datagen.flow(X, y, batch_size=10),
                                samples_per_epoch=len(X),
                                nb_epoch=1000, 
                                validation_data=(X_val, y_val),
                                callbacks=callbacks,
                                verbose=1)
            
        else:
            print ('-- Fitting the model...')
            '''
            model.fit({'main_input': X, 'aux_input': X},
                      {'main_output': y, 'aux_output': y}, validation_split=0.5,
                      nb_epoch=10000, batch_size=10, callbacks=callbacks)
            '''
            history = model.fit(X, y, batch_size=30, nb_epoch=300, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks_list)
            
        print ('-- Saving weights...')
        oname = os.path.join('../output/weights', 'weights_ep10000.hdf5')
        model.save_weights(oname)
	
        #plt.ioff()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim(0 , 5)
        plt.draw()
        plt.pause(0.05)
        plt.savefig('loss.png')
        
    if predicting:
        #del X
        #del y
        X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = d.prepare_data(img_rows, img_cols, color_type, True, False, True)
        del X_val
        del y_val
        del Xa
        del ya
        del Xa_val
        del ya_val
        model = smodel.build_segnet(img_shape=(3, 160, 224), n_classes=3, l2_reg=0.,
                init='glorot_uniform', path_weights=None,
                freeze_layers_from=None, use_unpool=False, basic=True)
        oname = os.path.join('../output/weights', 'weights_ep10000.hdf5')
        print oname
        model.load_weights(oname)
        
        num_outputs = len(X)
        print num_outputs
        print ('-- Predicting...')
        general_predictions = model.predict(np.array(X[0:num_outputs]))
        predictions = general_predictions
        print predictions.shape
        
        result = (predictions - predictions.min()) / (predictions.max() - predictions.min()) #Cal?
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.set_title('prediction')
        ax2 = fig.add_subplot(312)
        ax2.set_title('GT')
        ax3 = fig.add_subplot(313)
        ax3.set_title('input')
        
        plt.draw()
        for idx in range(num_outputs): #y_val.shape[0]
            #idx = 0
                        
            output_img = result[idx]
            pack = predictions[idx].transpose(1,2,0)
            oname = os.path.join('../output/predictions', str(idx)+'_reg.png')
           
            
            ax1.imshow(pack)
            #ax1.pause(0.2)
            #cv2.imwrite(oname, pack*255)
            
            resulty = np.array(np.reshape(y,(y.shape[0],color_type,img_rows,img_cols)))
            output_img = resulty[idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join('../output/predictions', str(idx)+'_y.png')
            
            ax2.imshow(pack)
            #ax2.pause(0.2)
            #cv2.imwrite(oname, pack*255) 
            
            output_img = X[idx]
            pack = output_img.transpose(1,2,0) # img_rows x img_cols x 3
            oname = os.path.join('../output/predictions', str(idx)+'_oX.png')
           
            ax3.imshow(pack)
            #ax3.pause(0.2)
            #cv2.imwrite(oname, pack*255)
            #plt.draw()
            plt.pause(0.2)
            
        '''
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
            
        '''    
            
            
