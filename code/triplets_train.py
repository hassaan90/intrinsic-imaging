'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU

Along the lines of BPR [1].

[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." 
Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1337)  # for reproducibility
import random
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, merge, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, SGD, Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import glob
from keras.applications import ResNet50, VGG16
from scipy.misc import imread, imresize
import TripletsLoader as tl
import tensorflow as tf

from skimage import io
from skimage.transform import rescale
import matplotlib.pyplot as plt

def L2_loss(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def L2_triplet_loss(X):
    user, positive, negative = X
    m = 0.1
#    d = K.sum(K.abs(x - y), axis=1, keepdims=True) - K.sum(K.abs(x - z), axis=1, keepdims=True)
    loss = 0.5 * K.maximum(K.variable(0), K.variable(m) + K.sum(K.square(user - positive), axis=1, keepdims=True) 
                                                        - K.sum(K.square(user - negative), axis=1, keepdims=True))
    return loss

def L1_triplet_loss(X):
    user, positive, negative = X
    x, y, z = X
    m = 0.1
#    d = K.sum(K.abs(x - y), axis=1, keepdims=True) - K.sum(K.abs(x - z), axis=1, keepdims=True)
    loss = 0.5 * K.maximum(K.variable(0), K.variable(m) + K.sum(K.abs(user - positive), axis=1, keepdims=True) - 
                                                          K.sum(K.abs(user - negative), axis=1, keepdims=True))
    return loss

def BPR_triplet_loss(X):
    '''
    [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." 
    Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.
    '''
    user, positive, negative = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user * positive, axis=-1, keepdims=True) -
        K.sum(user * negative, axis=-1, keepdims=True))

    return loss


def triplet_loss_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def distance(x, y=None):
    
    if y is None:
        y = x
    distance_m = np.zeros((len(x), len(y)))
    
    for i in range(len(x)):
        distance_m[i] = np.sum((np.subtract(y, x[i])**2), axis=1)

    return np.sqrt(distance_m)


def similarity(x, y=None):
    if y is None:
        y = x

    norm_x = np.linalg.norm(x,axis=1)
    norm_y = np.linalg.norm(y,axis=1)
    return np.dot(x, y.transpose())/np.dot(norm_x.reshape((-1,1)),norm_y.reshape((1,-1)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.2].mean()


def MAPatN(X_test, Y_test, X_recover=None, Y_recover=None, N=5, simililarity = False):
#https://www.kaggle.com/wiki/MeanAveragePrecision   
    if X_recover is None:
        X_recover = X_test
        Y_recover = Y_test
        
    if similarity:
        distance = -similarity(X_test, X_recover)
    else:
        distance = distance(X_test, X_recover)

    N = min(N, distance.shape[1])
    similar = np.zeros((X_test.shape[0],N+1), dtype=np.int) #    similar = np.zeros((X_test.shape[0],N), dtype=np.int)
    values = np.zeros((X_test.shape[0],N+1), dtype=np.float) #    similar = np.zeros((X_test.shape[0],N), dtype=np.int)
    error = np.zeros((X_test.shape[0],))
    for i in range(X_test.shape[0]):
        ind = np.argsort(distance[i])[:N+1]#[1:N+1]
        
        similar[i] = ind
        values[i] = -np.sort(distance[i])[:N+1]#[1:N+1]

        hit = ((Y_test[i]==Y_recover[ind])*1).astype(float)[1:]#remove [1:]
        error[i] = sum(np.cumsum(hit)/np.arange(1,N+1))/N

    return np.mean(error), error, similar, values
    
def topN_error(X_test, Y_test, X_recover=None, Y_recover=None, N=5, simililarity = False):
    weights = [0.4,0.30,0.25,0.20,0.18,0.12,0.1,0.08,0.07,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        
    if X_recover is None:
        X_recover = X_test
        Y_recover = Y_test
        
    if similarity:
        distance = -similarity(X_test, X_recover)
    else:
        distance = distance(X_test, X_recover)
    
    similar = np.zeros((X_test.shape[0],N), dtype=np.int)
        
    N = min(N, distance.shape[1])
    error = np.zeros((X_test.shape[0],))
    for i in range(X_test.shape[0]):
        ind = np.argsort(distance[i])[:N]
        similar[i] = ind
        error[i] = 1-(sum((Y_test[i]==Y_recover[ind])*1)>0)

    return np.mean(error), error, similar

    
def image_preprocessing(img):
    img=img.astype(np.float32)
    img = img[:, :, ::-1]
    # Zero-center by mean pixel
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    return img/255
    
def get_images(file_name, images_dir, class_dict={}):
    train_names = open(file_name, 'r')
    num_lines = sum(1 for ln in train_names)
    x = np.zeros((num_lines, 224, 224, 3))
    y = np.zeros((num_lines))

    n = 0
    train_names = open(file_name, 'r')
    for i, line in enumerate(train_names):
        class_name = line.split('/')[0]
        img_name = os.path.join(images_dir, line[:-1])
        if not class_name in class_dict.keys():            
            class_dict[class_name] = n
            n += 1
        I = imread(img_name, mode='RGB').astype('float32')

        I = imresize(I, [224,224])
        x[i] = I
        y[i] = int(class_dict[class_name])
    print (x.shape)
    print (class_dict)
    return x, y, class_dict
    

def load_cortexica_texture_dummy():
    
    root_dir = '/home/bojana/bojana/datasets/cortexica_texture_ex1'#'/media/bgajic/Datos/data_bojana/datasets/cortexica_texture'
    labels_dir = os.path.join(root_dir, 'labels')
    images_dir = os.path.join(root_dir, 'images')
    
    # read train images:
    train_file = os.path.join(labels_dir, 'train3.txt')
    X_train, Y_train, class_dict = get_images(train_file, images_dir)
    
    val_file = os.path.join(labels_dir, 'val3.txt')
    X_val, Y_val, class_dict = get_images(val_file, images_dir, class_dict)

    test_file = os.path.join(labels_dir, 'test3.txt')
    X_test, Y_test, class_dict = get_images(test_file, images_dir, class_dict)
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def build_model(architecture='ResNet50', loss_function=L2_triplet_loss, weights_directory='./weights', model_name='model', input_dim=None):
    # network definition
    if architecture == 'fc':
        base_network = create_base_network(input_dim)
    elif architecture == 'alexnet':
        #base_network = convnet('alexnet',weights_path="/home/bgajic/cvc/siamese_keras/weights/alexnet_weights.h5", heatmap=False)
        #print ('done')
        #base_network = create_AlexNet()
        print ('done')
    elif architecture == 'ResNet50':
        base_network = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
        layer = 'avg_pool'
        print ('resnet loaded')
    elif architecture == 'VGG16':
        base_network = VGG16(include_top=True, weights='imagenet')
        layer = 'block5_pool'
        print ('vgg16 loaded')

    
    plot_model(base_network,'resnet.png', show_shapes=True, show_layer_names=1)
    x = base_network.get_layer(layer).output
    x = Flatten(name='flatten')(x)
    
    x = Dense(512, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='softmax', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    mod_network = Model(inputs=base_network.input, outputs=x)

    input_a = Input(shape=(224,224,3),name='Query')
    input_b = Input(shape=(224,224,3),name='Positive_example')
    input_c = Input(shape=(224,224,3),name='Negative_example')
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the three branches
    processed_a = mod_network(input_a)
    processed_b = mod_network(input_b)
    processed_c = mod_network(input_c)

    distance = Lambda(loss_function, output_shape=triplet_loss_shape)([processed_a, processed_b, processed_c])
    model = Model(inputs=[input_a, input_b, input_c], outputs=distance)

#    Load last best weights 
    # train
    rms = RMSprop()
    sgd = SGD(lr=0.000001, decay=5e-6, momentum=0.9)#, nesterov=True)
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-6)

    all_files = glob.glob(weights_directory+'/be*.hdf5')
    if len(all_files)>0:
        latest_file = max(all_files, key=os.path.getctime)
        print ('Loading weights: {}'.format(latest_file))
        model.load_weights(latest_file)


    model.compile(loss=identity_loss, optimizer=adam)
    return model
    
def main():
    #load_cortexica_texture()
    root_dir = '/home/bojana/bojana/datasets/cortexica_texture_ex1'#'/media/bgajic/Datos/data_bojana/datasets/cortexica_texture_ex1'

    dataset = 'cortexica_texture' #{cifar10, mnist, cortexica_texture}
    architecture = 'ResNet50' #{'alexnet', 'fc', 'ResNet50', 'VGG16'}
#    architecture = 'VGG16' #{'alexnet', 'fc', 'ResNet50'}
    loss_fun = L2_triplet_loss #{BPR_triplet_loss, L2_triplet_loss, L1_triplet_loss}
    
    model_name = architecture +'_'+dataset +'_BPR'
    model_name = architecture + '_compact3_'+dataset +'_L2'
    model_name = architecture + '_'+dataset +'_L2'
    
    train_datagen = tl.TripletsImageDataGenerator(preprocessing_function=image_preprocessing)
    validation_datagen = tl.TripletsImageDataGenerator(preprocessing_function=image_preprocessing)
    
    if dataset == 'cifar10':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        #X_train = resizeimage.resize_cover(X_train, [227,227,3])
        #X_test = resizeimage.resize_cover(X_test, [227,227,3])
        X_train = X_train.reshape(50000, 3, 32, 32)
        X_test = X_test.reshape(10000, 3, 32, 32)
        input_dim = (3,32,32)
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        input_dim = 784
    elif dataset == 'cortexica_texture_dummy':        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cortexica_texture_dummy()
    elif dataset == 'cortexica_texture':
        X_train = None
        
    if X_train is not None:        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=20)
        validation_generator = validation_datagen.flow(X_test, y_test, batch_size=20)
    else:        
        train_generator = train_datagen.flow_from_directory(
                target_size=(224, 224),
                shuffle = True,
                batch_size=20,
                directory=os.path.join(root_dir,'train'),
        		similarity_file='class_to_class_similarity.npy')#file with similarities
        validation_generator = validation_datagen.flow_from_directory(
                directory=os.path.join(root_dir,'validation'),
                target_size=(224, 224),
                batch_size=20)
      
#    prepare folders  

    weights_directory = './weights'
    if not os.path.exists(weights_directory): os.makedirs(weights_directory)
    weights_directory = weights_directory +'/' + model_name 
    if not os.path.exists(weights_directory): os.makedirs(weights_directory)

    logs_directory = '/home/logs/' + model_name 
    if not os.path.exists(logs_directory ): os.makedirs(logs_directory )

#    create model
    model = build_model(architecture, loss_fun, weights_directory, model_name)
    plot_model(model, to_file='base_model.png', show_shapes=True, show_layer_names=1)
    model.summary()

    train = False
    if train:
        checkpointer = ModelCheckpoint(filepath=weights_directory +"/weights_{epoch:02d}.hdf5", verbose=1, save_best_only=False,monitor='val_loss', mode='auto', period=10)
        checkpointer2 = ModelCheckpoint(filepath=weights_directory +"/best_weights_{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_loss', mode='auto')
    
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples,
            epochs=500,
            validation_data=validation_generator,
            validation_steps=300,
            callbacks=[checkpointer, checkpointer2, TensorBoard(log_dir=logs_directory, histogram_freq=1)])


    else:
        predict_model = Model(inputs=model.layers[3].layers[0].input, outputs=model.layers[3].layers[-1].output)
        plot_model(predict_model, to_file='predcit_model.png', show_shapes=True, show_layer_names=True)

        test_datagen = ImageDataGenerator(preprocessing_function=image_preprocessing)
        test_iterator = test_datagen.flow_from_directory(
                directory=os.path.join(root_dir,'test'),
                target_size=(224, 224),
                shuffle = False,
                batch_size=1)
        test_features = predict_model.predict_generator(test_generator, test_generator.samples, verbose=1)
#        test_features = predict_model.predict_generator(test_generator, 100)
#        np.save('/home/bojana/bojana/test_descriptors.npy', test_features)
#        np.save('/home/bojana/bojana/image_names.npy', test_generator.filenames)
        
        test_classes  = test_generator.classes[:test_features.shape[0]]
        
        N=5
        mean_error, error_vec, similar,values = MAPatN(test_features, test_classes, N=N, simililarity = True)
        print ('\nMAP at %d: %f\n' % (7, mean_error))
        for c in test_generator.class_indices:
            c_err = np.mean(error_vec[test_classes == test_generator.class_indices[c]])
            print ('%s: %f' % (c, c_err))



        plt.close("all")
        plt.figure(1)
        for i in range(0,test_features.shape[0],10):
            plt.clf()
                        
            im = io.imread(test_generator.directory+'/'+test_generator.filenames[i])
            #im = rescale(im, 0.4, preserve_range=True)
            plt.subplot(1,similar.shape[1]+1,1)
            plt.axis('off')
            plt.imshow(im)
            
            for k in range(similar.shape[1]):
                im = io.imread(test_generator.directory+'/'+test_generator.filenames[similar[i,k]])
                #im = rescale(im, 0.4, preserve_range=True)
                ax=plt.subplot(1,similar.shape[1]+1,2+k)
                plt.axis('off')
                plt.imshow(im)
                ax.set_title(str(values[i,k]))
            plt.draw()
            plt.pause(0.001)


    
if __name__=='__main__':
#    main()
    with tf.device('/gpu:1'): main()
