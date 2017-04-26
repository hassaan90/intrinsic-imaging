from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, NumpyArrayIterator, img_to_array, load_img, array_to_img
import numpy as np
import re

from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import scipy.misc
import os
import threading
import warnings


from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

class SelectedImageDataGenerator(ImageDataGenerator):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(SelectedImageDataGenerator, self).__init__(featurewise_center=featurewise_center, 
                                                         samplewise_center=samplewise_center,
                                                         featurewise_std_normalization=featurewise_std_normalization,
                                                         samplewise_std_normalization=samplewise_std_normalization,
                                                         zca_whitening=zca_whitening, rotation_range=rotation_range, 
                                                         width_shift_range=width_shift_range,height_shift_range=height_shift_range,
                                                         shear_range=shear_range, zoom_range=zoom_range, 
                                                         channel_shift_range=channel_shift_range,fill_mode=fill_mode, 
                                                         cval=cval, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                                         rescale=rescale, preprocessing_function=preprocessing_function, data_format=data_format)


    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpg',
                            follow_links=False,
                            which_file=None):
        return SelectedDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            which_file=which_file)



class TripletsImageDataGenerator(SelectedImageDataGenerator):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(TripletsImageDataGenerator, self).__init__(featurewise_center=featurewise_center, 
                                                         samplewise_center=samplewise_center,
                                                         featurewise_std_normalization=featurewise_std_normalization,
                                                         samplewise_std_normalization=samplewise_std_normalization,
                                                         zca_whitening=zca_whitening, rotation_range=rotation_range, 
                                                         width_shift_range=width_shift_range,height_shift_range=height_shift_range,
                                                         shear_range=shear_range, zoom_range=zoom_range, 
                                                         channel_shift_range=channel_shift_range,fill_mode=fill_mode, 
                                                         cval=cval, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                                         rescale=rescale, preprocessing_function=preprocessing_function, data_format=data_format)


    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='jpg',
                            follow_links=False,
                            which_file=None,
                            similarity_file=None):
        return TripletsDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            which_file=which_file,
            similarity_file=similarity_file)


    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return TripletsNumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class TripletsNumpyArrayIterator(NumpyArrayIterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        super(TripletsNumpyArrayIterator, self).__init__(x=x, y=y, image_data_generator=image_data_generator,
                                                         batch_size=batch_size, shuffle=shuffle, seed=seed,
                                                         data_format=data_format, save_to_dir=save_to_dir, 
                                                         save_prefix=save_prefix, save_format=save_format)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        inputs = []
        for index in range(3):
            batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
            # build batch of image data
            for i, j in enumerate(index_array):
                if index==0: #loads the image
                    ind = j
                elif index==2: #loads a negative example
                    eligible = [x for x  in range(len(self.filenames)) if self.y[x] != self.y[j]]
                    ind = eligible[np.random.randint(len(eligible))]
                elif index==1: #loads a positive example
                    eligible = [x for x  in range(len(self.filenames)) if self.y[x] == self.y[j] and x != j]
                    ind = eligible[np.random.randint(len(eligible))]

                x = self.x[ind]
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
            inputs.append(batch_x)

            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return inputs
        batch_y = self.y[index_array]
        return inputs, batch_y


class SelectedDirectoryIterator(DirectoryIterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        which_file: file that contains which images should be used in the batches
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False,
                 which_file=None):
        super(SelectedDirectoryIterator, self).__init__(directory=directory, image_data_generator=image_data_generator, 
                                                        target_size=target_size, color_mode=color_mode, 
                                                        classes=classes, class_mode=class_mode, 
                                                        batch_size=batch_size, shuffle=shuffle, seed=seed, 
                                                        data_format=data_format, save_to_dir=save_to_dir, 
                                                        save_prefix=save_prefix, save_format=save_format, follow_links=follow_links)
        if which_file != None:
            if not os.path.isabs(which_file):
                which_file = directory + '/' + which_file

            with open(which_file) as file:
                files = file.read().splitlines()

            indices = [i for i,f in enumerate(files) if f in self.filenames]
            self.filenames = self.filenames[indices]
            self.classes = self.classes[indices]
            self.samples = len(self.filenames)


class TripletsDirectoryIterator(SelectedDirectoryIterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        which_file: file that contains which images should be used in the batches
        similarity_file: file that contains the similarity between classes
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False,
                 which_file=None,
                 similarity_file=None):
        super(TripletsDirectoryIterator, self).__init__(directory=directory, image_data_generator=image_data_generator, 
                                                        target_size=target_size, color_mode=color_mode, 
                                                        classes=classes, class_mode=class_mode, 
                                                        batch_size=batch_size, shuffle=shuffle, seed=seed, 
                                                        data_format=data_format, save_to_dir=save_to_dir, 
                                                        save_prefix=save_prefix, save_format=save_format, 
                                                        follow_links=follow_links, which_file=which_file)

        #similarity_file = 'class_to_class_similarity.npy'
        self.class_similarities = None
        if similarity_file is not None:
            if not os.path.isabs(similarity_file):
                similarity_file = directory + '/' + similarity_file
            
            if os.path.isfile(similarity_file):
                self.class_similarities = np.load(similarity_file).item()


    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        inputs = []
        for index in range(3):
            batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
            grayscale = self.color_mode == 'grayscale'
            # build batch of image data
            for i, j in enumerate(index_array):
                if index==0: #loads the image
                    ind = j
                elif index==2: #loads a negative example
                    eligible = [x for x  in range(len(self.filenames)) if self.classes[x] != self.classes[j]]
                    if self.class_similarities is not None:
                        class_im = [k for k in self.class_indices.keys() if self.class_indices[k]==self.classes[j]][0]
                        sort_sim = sorted(self.class_similarities[class_im].items(), key= lambda y:y[1])                        
                        sort_sim = sort_sim[:-int(len(sort_sim)*0.2)]
                        opposite_classes = [self.class_indices[c[0]] for c in sort_sim]
                        eligible = [e for e in eligible if self.classes[e] in opposite_classes]
                    ind = eligible[np.random.randint(len(eligible))]
                elif index==1: #loads a positive example
                    eligible = [x for x  in range(len(self.filenames)) if self.classes[x] == self.classes[j] and \
                                                                          self.filenames[x] != self.filenames[j]]
                    ind = eligible[np.random.randint(len(eligible))]

                fname = self.filenames[ind]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
            inputs.append(batch_x)
            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return inputs

#        img1 = array_to_img(inputs[0][0], self.data_format, scale=True)
#        img2 = array_to_img(inputs[1][0], self.data_format, scale=True)
#        img3 = array_to_img(inputs[2][0], self.data_format, scale=True)
        img  = array_to_img(np.hstack([inputs[0][0], inputs[1][0], inputs[2][0]]), self.data_format, scale=True)
        img.save( 'im'+str(current_index)+'.jpg')

        
        return inputs, batch_y
 