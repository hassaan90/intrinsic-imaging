# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:09:23 2017

@author: hassan
"""

from __future__ import print_function
from keras import backend as K
from keras.engine.training import GeneratorEnqueuer
from pprint import pprint
import os
import numpy as np
import cv2
import pickle
import segnet as smodel
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import itertools
import keras
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint, Progbar, ProgbarLogger, EarlyStopping
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import time
from sys import argv
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam 
import fnmatch
from itertools import chain
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
class Histories(keras.callbacks.Callback):

    def __init__(self, val_gen, directory):
        super(Histories, self).__init__()
        self.directory = directory
        self.count = 0
        #self.count1 = 0
        self.nepoch = 0
        self.losses = []
        self.val_losses = []
        self.val_gen = val_gen
    


    def on_batch_end(self, batch, logs={}):
            '''
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
            '''
            return
            
    def on_epoch_end(self, epoch, logs={}):
            self.nepoch = epoch
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            enqueuer = GeneratorEnqueuer(self.val_gen, pickle_safe=False)
            enqueuer.start(nb_worker=1, max_q_size=5, wait_time=0.05)
	    plt.ioff()
            plt.plot(self.losses,'b-')
            plt.plot(self.val_losses,'c-')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.ylim(0 , 0.2)
            #plt.draw()
            plt.pause(0.02)
            plt.savefig(self.directory+'_loss.png')
            #print self.losses[0]
            #if self.count >100:
            #    self.count = 0
            #pprint(self.model, indent=2)

            # Get data for this minibatch
            data = None
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    data = enqueuer.queue.get()
                    break
                else:
                    plt.pause(0.05)
            #data = data_gen_queue.get()
            x_true = data[0]
            #print (x_true.shape)
            img = x_true[0].transpose(1,2,0)
#            plt.imshow(img)
#            plt.show()
#            plt.pause(0.01)
#            y_true = data[1]
#            img = y_true[0].transpose(1,2,0)
#            plt.imshow(img)
#            plt.show()
#            plt.pause(0.01)
            #print (y_true.shape)

            # Get prediction for this minibatch
            y_pred = self.model.predict(x_true)

            # Reshape y_true and compute the y_pred argmax
#            if K.image_dim_ordering() == 'th':
#                y_pred = np.argmax(y_pred, axis=1)
#                y_true = np.array(y_true).transpose(0,3,1,2)
#            else:
#                y_pred = np.argmax(y_pred, axis=3)
            for i in range(y_pred.shape[0]):
                img = (y_pred[i] - np.min(y_pred[i]))
            
                img = (img / np.max(img)).transpose(1,2,0)
                #print (img.shape)
                #y_pred = y_pred[0]
                gt = np.array(x_true[i]).transpose(1,2,0)
                #print (gt.shape)
                #y_pred = y_pred.transpose(1,2,0)
                #y_gt = y_gt.transpose(1,2,0)
                result = np.concatenate((gt,img ), axis=1)
            #print np.max(y_pred), np.min(y_pred), np.max(y_gt), np.min(y_gt)
            #plt.imshow(result)
            #plt.show()
            #pause(0.01)
                oname = os.path.join(self.directory, str(self.nepoch)+'_'+str(i)+'.jpg')
            #print oname
            #time.sleep(5)
            #self.count += 1
                cv2.imwrite(oname, cv2.cvtColor(result*255, cv2.COLOR_RGB2BGR))
#                    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1],
#                                                 y_true.shape[2]))
            # Save output images


        # Stop data generator
            if enqueuer is not None:
                enqueuer.stop()
            
            
            
            
            
            
            '''
            val_samples =20
            pred = self.model.predict_generator(self.val_gen, val_samples)
            #plt.imshow(y_pred[0,:,:,:].transpose(1,2,0))
            #plt.show()
            #plt.pause(0.01)
            #print (y_pred.shape)
            #y_pred = self.model.predict(self.model.validation_data[0][self.count:self.count+1,:,:,:])
            #print y_pred.shape
            #print type(y_pred)
            for i in range(val_samples):
                y_pred = (pred[i,:,:,:] - np.min(pred[i,:,:,:]))
            
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
            oname = os.path.join(self.directory, str(self.count)+'_'+str(self.nepoch)+'_'+'.jpg')
            #print oname
            #time.sleep(5)
            self.count += 1
            cv2.imwrite(oname, result*255)
            '''
            return

    def on_train_begin(self, logs={}):
            self.losses = []
            return
def format_gen_outputs(gen1,gen2):
    x1 = gen1[0]
    y1 = gen2[0]
    y2 = gen2[1]
    return zip(x1, [y1, y2])
def generator(gen1 , gen2, gen3):
    return itertools.izip(gen1, [gen2, gen3])


def counter(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:    
            if file.endswith('.png'):
            	count += 1
    return count

if __name__ == '__main__':
    #script, experiment,augmentation,split   = argv
    #augmentation = str2bool(augmentation)
    #print (script)
    experiment = 'Exp_10'
    augmentation = False
    #augmentation = str2bool(augmentation)
    split = 'scene'
    script = 'data_hassan.py'
    
    #experiment = ('exp_'+str(1))
    save_folder = os.path.join('../output', experiment)
    weight_folder = os.path.join(save_folder, 'weights')
    prediction_folder = os .path.join(save_folder, 'prediction')
    if (split == 'scene'):
        if (augmentation == True):
            gen_path = '/home/hassan/Intrinsic/code/input/data/sintelGenerated/'
	#print (counter(os.path.join(gen_path,'albedo/train')))
        else:
            gen_path = '/home/hassan/Intrinsic/code/input/data/sintel/'
	#print (counter(os.path.join(gen_path,'albedo/train')))
    elif( split == 'image'):
        if (augmentation == True):
            gen_path = '/home/hassan/Intrinsic/code/input/data/sintelimageGenerated/'
	#print (counter(os.path.join(gen_path,'albedo/train')))
        else:
            gen_path = '/home/hassan/Intrinsic/code/input/data/sintelimage/'
            
	#print (counter(os.path.join(gen_path,'albedo/train')))
    print (gen_path)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(weight_folder):
        os.mkdir(weight_folder)
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)
    file_r = open(script, 'r')
    indata = file_r.read()
    file_w = open(os.path.join(save_folder,'script.py'), 'w')
    file_w.write(indata)
    file_r.close()
    file_w.close()
    image_datagen = ImageDataGenerator(
            rotation_range=15.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='nearest')
    
    mask_datagen = ImageDataGenerator(
            rotation_range=15.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='nearest')
    shading_datagen = ImageDataGenerator(
            rotation_range=15.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    seed_validation = 1
    validation_image_generator = test_datagen.flow_from_directory(
            os.path.join(gen_path,'clean/test'),
            target_size=(128, 256),class_mode=None,classes=None,
            batch_size=20,seed=seed_validation)
    validation_mask_generator = test_datagen.flow_from_directory(
            os.path.join(gen_path,'albedo/test'),
            target_size=(128, 256),class_mode=None,classes=None,
            batch_size=20,seed=seed_validation)
    validation_shading_generator = test_datagen.flow_from_directory(
            os.path.join(gen_path,'gray_shading/test'),
            target_size=(128, 256),class_mode=None,classes=None,
            batch_size=20,seed=seed_validation)
    #testDataGen = ImageDataGenerator(rescale=1./255)
    seed = 5
    #image_datagen.fit(image_datagen, augment=True, seed=seed)
    #mask_datagen.fit(mask_datagen, augment=True, seed=seed)
    
    image_generator = image_datagen.flow_from_directory(
        os.path.join(gen_path,'clean/train'),
        class_mode=None,target_size=(128,256),batch_size=20,classes=None,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        os.path.join(gen_path,'albedo/train'),
        class_mode=None,target_size=(128,256),batch_size=20,classes = None,
        seed=seed)
    shading_generator = shading_datagen.flow_from_directory(
        os.path.join(gen_path,'gray_shading/train'),
        class_mode=None,target_size=(128,256),batch_size=20,classes = None,
        seed=seed)
    
    input_generator =  generator(image_generator, image_generator, image_generator)
    output_generator = itertools.izip(mask_generator, shading_generator)
    print(next(input_generator)[1].shape)
#    output = itertools.izip(mask_generator, shading_generator)
#    train_shading_generator = itertools.izip(image_generator, shading_generator)
#
#    #train = map(format_gen_outputs, train_generator, train_shading_generator)
#    validation_generator = itertools.izip(validation_image_generator, validation_mask_generator)
#    validation_shading_generator = itertools.izip(validation_image_generator, validation_shading_generator)
#    #test = map(format_gen_outputs, validation_generator, validation_shading_generator)
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#              patience=5, min_lr=0.00001)
##    model = smodel.build_segnet(img_shape=(3, 128, 256), n_classes=3, l2_reg=0.,
##                     init='he_normal', path_weights=None,
##                     freeze_layers_from=None, use_unpool=False, basic=True)
#    model = smodel.direct_intrinsic(img_shape=(3, 128, 256), weight_path='', activation= 'relu', init = 'he_normal' )
#    #train=(image_generator,[mask_generator,shading_generator])
#    plot(model, to_file='model.png', show_shapes = True, show_layer_names = True)
#    model.summary()
#    kfold_weights_path = os.path.join(weight_folder, 'checkpoint' + '.hdf5' )
#    plot_loss_callback = LambdaCallback(
#    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
#    logs['loss']))
#    my_callback = Histories(validation_generator,prediction_folder)
#    #Early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
#    
#    callbacks_list = [
#    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0), my_callback, reduce_lr 
#            ]
#    adam_rl = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#    model.compile(loss="mse", optimizer=adam_rl)
#    model.fit_generator(
#        input_generator,
#        samples_per_epoch=image_generator.nb_sample*2,
#        nb_epoch=500,
#        callbacks=callbacks_list)
#    oname = os.path.join(weight_folder, 'weights_final.hdf5')
#    model.save_weights(oname)
#
#    
#    '''
#    img = load_img('/home/hassan/Intrinsic/code/input/data/segnet/albedo/ambush_5/frame_0002.png')  # this is a PIL image
#    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#    
#    # the .flow() command below generates batches of randomly transformed images
#    # and saves the results to the `preview/` directory
#    i = 0
#    for batch in datagen.flow(x, batch_size=1,
#                              save_to_dir='/home/hassan/Intrinsic/code/output/augmentation', save_prefix='cat', save_format='png', seed= 0):
#        i += 1
#        if i > 40:
#            break  # otherwise the generator would loop indefinitely
#    img = load_img('/home/hassan/Intrinsic/code/input/data/segnet/albedo/ambush_5/frame_0005.png')  # this is a PIL image
#    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#    i = 0
#    for batch in datagen.flow(x, batch_size=1,
#                              save_to_dir='/home/hassan/Intrinsic/code/output/augmentation/test', save_prefix='cat', save_format='png', seed= 0):
#        i += 1
#        if i > 40:
#            break  # otherwise the generator would loop indefinitely '''