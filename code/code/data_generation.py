#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:21:30 2017

@author: hassan
"""
import os
import numpy as np
import cv2
#import random

path = os.path.join(os.getcwd(),'../input/data','sintel')
save_path = os.path.join(os.getcwd(),'../input/data','sintelGenerated')
img_extension = '.png'
parent_list = os.listdir(path)
#folders = ['albedo','clean']
img_extension = '.png'
w_factor = 4
#o_width = 1024
#o_height = 436
for experiment_folder in parent_list: # ['albedo','clean']     
        
        current_folder = os.path.join(path, experiment_folder)
        folder_list = os.listdir(current_folder)
        save_experiment_folder= os.path.join(save_path, experiment_folder)
        if not os.path.exists(save_experiment_folder):
            os.mkdir(save_experiment_folder)
        for folder in folder_list:
            current_folder_list = os.path.join(current_folder, folder)
            save_folder_list= os.path.join(save_experiment_folder, folder)
            if not os.path.exists(save_folder_list):
                os.mkdir(save_folder_list)
            fileList = os.listdir(current_folder_list)
            imagesList = filter(lambda element: img_extension in element, fileList)
             
            for filename in imagesList:
                    current_image = os.path.join(current_folder_list, filename)
                    img = cv2.imread(current_image)
                    o_height, o_width = img.shape[:2]
                    window = np.array([o_height/w_factor, o_width/w_factor, 3])
                    imgResize = cv2.resize(img, (window[1], window[0]))
                    save_image=os.path.splitext(filename)[0]
                    save_img = os.path.join(save_folder_list,save_image+'_'+str(0)+'.png')
                    cv2.imwrite(save_img,imgResize)
                    k=1
                    for widy in range(4):
                                for widx in range(4):
                                    curr_window = img[widy*window[0]:(widy+1)*window[0], widx*window[1]:(widx+1)*window[1], 0:window[2]]                                    
                                    save_img = os.path.join(save_folder_list,save_image+'_'+str(k)+'.png')
                                    cv2.imwrite(save_img,curr_window)
                                    k=k+1
 
 