from __future__ import print_function
import os
import numpy as np
import cv2
import pickle

sintel_scenes = dict(
    train=['alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1', 'temple_2'],
    test=['alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2', 'temple_3'])


def get_pad_multiple(shape, num):
    # Get pad values for y and x axeses such that an image
    # has a shape of multiple of 32 for each side
    def _f(s):
        n = s // num
        r = s % num
        if r == 0:
            return 0
        return num - r

    return _f(shape[0]), _f(shape[1])


def bgr2rgb(img):
    # Convert an BGR image to RGB
    return img[..., ::-1]


def minmax_01(img):
    # Put values in a range of [0, 1]
    ma, mi = img.max(), img.min()
    return (img - mi) / (ma - mi)


def unpad_img(img, pad):
    # Crop an image according to pad, used in post_process
    y = img.shape[0] - pad[0]
    x = img.shape[1] - pad[1]
    return img[:y, :x, :]


def post_process(img, pad, i=0):
    # Post processes of Direct intrinsics net.
    # The output of direct intrinsics net is in log domain with bias +0.5, and in BGR order.
    return unpad_img(minmax_01(bgr2rgb((np.exp(img[i]) - 0.5).transpose(1, 2, 0))), pad)


def binarylab(img, labels, num_classes):
    # print ('real shape >> i:'+str(img.shape[0]) + ', j: ' + str(img.shape[1]))
    x = np.zeros([img.shape[0], img.shape[1], num_classes])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x[i, j, labels[i][j]] = 1
    return x


def normalize(rgb, mode=0):
    if mode == 0:
        return rgb / 255.0
    else:
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

    return norm


def precomp_weights(img, preweights):
    for c in range(img.shape[0]):
        for r in range(img.shape[1]):
            preweights[(img[c][r])] = preweights[(img[c][r])] + img[c][r]
    return preweights


def unison_shuffled_copies(a, b):
    '''
    X_val,y_val = unison_shuffled_copies(X_val,y_val)
    img_x = X_val[10].transpose(1,2,0)
    cv2.imwrite('X.jpg',img_x)
    img_y = y_val[10].transpose(1,2,0)
    cv2.imwrite('y.jpg',img_y)
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))  # 352 -> array([1, 5, 8, 6])
    return a[p], b[p]


def generate_local_data(img, o_width=1024, o_height=436, w_factor=4, osize=128, re_pad=True):
    # cv2.imwrite('global_img.jpg', img)
    window = np.array([o_height / w_factor, o_width / w_factor, 3])
    img_in_slices = []
    # Image partition
    for widy in range(4):
        for widx in range(4):
            curr_window = img[widy * window[0]:(widy + 1) * window[0], widx * window[1]:(widx + 1) * window[1],
                          0:window[2]]
            if re_pad:
                pad = get_pad_multiple(curr_window.shape, osize)
                curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT)
                curr_window = cv2.resize(curr_window, (osize, osize))
                img_in_slices.append(curr_window)
                # oname = 'local_y'+str(widy)+'-x'+str(widx)+'.jpg'
                # cv2.imwrite(oname,curr_window)
    return np.array(img_in_slices).transpose(0, 3, 1, 2)


def get_pad_multiple(shape, num):
    # Get pad values for y and x axeses such that an image
    # has a shape of multiple of 32 for each side
    def _f(s):
        n = s // num
        r = s % num
        if r == 0:
            return 0
        return num - r

    return _f(shape[0]), _f(shape[1])


def bgr2rgb(img):
    # Convert an BGR image to RGB
    return img[..., ::-1]


def minmax_01(img):
    # Put values in a range of [0, 1]
    ma, mi = img.max(), img.min()
    return (img - mi) / (ma - mi)


def unpad_img(img, pad):
    # Crop an image according to pad, used in post_process
    y = img.shape[0] - pad[0]
    x = img.shape[1] - pad[1]
    return img[:y, :x, :]


def post_process(img, pad, i=0):
    # Post processes of Direct intrinsics net.
    # The output of direct intrinsics net is in log domain with bias +0.5, and in BGR order.
    return unpad_img(minmax_01(bgr2rgb((np.exp(img[i]) - 0.5).transpose(1, 2, 0))), pad)


def binarylab(img, labels, num_classes):
    # print ('real shape >> i:'+str(img.shape[0]) + ', j: ' + str(img.shape[1]))
    x = np.zeros([img.shape[0], img.shape[1], num_classes])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x[i, j, labels[i][j]] = 1
    return x


def normalize(rgb, mode=0):
    if mode == 0:
        return rgb / 255.0
    else:
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:, :, 0]
        g = rgb[:, :, 1]
        r = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(b)
        norm[:, :, 1] = cv2.equalizeHist(g)
        norm[:, :, 2] = cv2.equalizeHist(r)

    return norm


def precomp_weights(img, preweights):
    for c in range(img.shape[0]):
        for r in range(img.shape[1]):
            preweights[(img[c][r])] = preweights[(img[c][r])] + img[c][r]
    return preweights


def unison_shuffled_copies(a, b):
    '''
    X_val,y_val = unison_shuffled_copies(X_val,y_val)
    img_x = X_val[10].transpose(1,2,0)
    cv2.imwrite('X.jpg',img_x)
    img_y = y_val[10].transpose(1,2,0)
    cv2.imwrite('y.jpg',img_y)
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))  # 352 -> array([1, 5, 8, 6])
    return a[p], b[p]


def generate_local_data(img, o_width=1024, o_height=436, w_factor=4, osize=128, re_pad=True):
    # cv2.imwrite('global_img.jpg', img)
    window = np.array([o_height / w_factor, o_width / w_factor, 3])
    img_in_slices = []
    # Image partition
    for widy in range(4):
        for widx in range(4):
            curr_window = img[widy * window[0]:(widy + 1) * window[0], widx * window[1]:(widx + 1) * window[1],
                          0:window[2]]
            if re_pad:
                pad = get_pad_multiple(curr_window.shape, osize)
                curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT)
                curr_window = cv2.resize(curr_window, (osize, osize))
                img_in_slices.append(curr_window)
                # oname = 'local_y'+str(widy)+'-x'+str(widx)+'.jpg'
                # cv2.imwrite(oname,curr_window)
    return np.array(img_in_slices).transpose(0, 3, 1, 2)


def new_prepare_data(img_rows=32, img_cols=32, color_type=3, only_val=False, data_augmentation=False, print_step=True):
    print('-- prepare_data: Version: 17.03.09.')

    train = ['alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1',
             'temple_2']  # 440
    val = ['alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2',
           'temple_3']  # 450

    path = os.path.join(os.getcwd(), '../input/data', 'segnet')

    if print_step == True:
        print('-- Current file: load.py.')

    img_extension = '.png'

    folders = ['albedo', 'clean']

    make_border = False
    normalize = True

    pad = -1

    X_train = []
    y_train = []

    X_val = None
    y_val = None
    num_images_val = None

    # Designed folders to val
    val_method = 2  # 0 = no val., 1 = automatic building of a val. set, 2 = load predefined val

    # Automatic val
    load_previous_val = False
    if val_method > 0:
        val_dict = {}
        X_val = []
        y_val = []
        val_split = 0.2
        val_scene = None

    if data_augmentation:
        Xa = []
        ya = []
        if val_method == 2:
            Xa_val = []
            ya_val = []
    w_factor = 4
    o_width = img_cols * w_factor  # 112*2*w_factor
    o_height = img_rows * w_factor  # 80*2*w_factor

    if print_step == True:
        print('-- Validation method: ' + str(val_method) + '...')
        print('-- Processing images...')

    for experiment_folder in folders:  # ['albedo','clean']

        current_folder = os.path.join(path, experiment_folder)

        if print_step == True:
            print('-- current_folder: ' + experiment_folder)

        if val_method == 2:
            for scene_folder in val:
                current_scene = os.path.join(current_folder, scene_folder)
                if print_step == True:
                    print('-- Validation, scene folder: ' + scene_folder)

                fileList = os.listdir(current_scene)
                globalList = filter(lambda element: img_extension in element, fileList)

                for filename in globalList:
                    current_image = os.path.join(current_scene, filename)
                    img = cv2.imread(current_image)
                    img = cv2.resize(img, (o_width, o_height))

                    if normalize:
                        img = img / 255.0

                    if make_border:
                        pad = get_pad_multiple(img.shape, img_rows)
                        rimg = cv2.copyMakeBorder(img, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT)
                        rimg = cv2.resize(rimg, (img_cols, img_rows))
                    else:
                        rimg = cv2.resize(img, (img_cols, img_rows))

                    if experiment_folder == folders[1]:
                        X_val.append(np.rollaxis((rimg), 2))
                        if data_augmentation:
                            osize = img_rows
                            # cv2.imwrite('global_img.jpg', img)
                            window = np.array([o_height / w_factor, o_width / w_factor, 3])
                            img_in_slices = []
                            # Image partition
                            for widy in range(4):
                                for widx in range(4):
                                    curr_window = img[widy * window[0]:(widy + 1) * window[0],
                                                  widx * window[1]:(widx + 1) * window[1], 0:window[2]]
                                    if make_border:
                                        pad = get_pad_multiple(curr_window.shape, osize)
                                        curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1],
                                                                         cv2.BORDER_CONSTANT)
                                        curr_window = cv2.resize(curr_window, (osize, osize))
                                    Xa_val.append(curr_window)
                    if experiment_folder == folders[0]:
                        y_val.append(np.rollaxis((rimg), 2))
                        if data_augmentation:
                            osize = img_rows
                            # cv2.imwrite('global_img.jpg', img)
                            window = np.array([o_height / w_factor, o_width / w_factor, 3])
                            img_in_slices = []
                            # Image partition
                            for widy in range(4):
                                for widx in range(4):
                                    curr_window = img[widy * window[0]:(widy + 1) * window[0],
                                                  widx * window[1]:(widx + 1) * window[1], 0:window[2]]
                                    if make_border:
                                        pad = get_pad_multiple(curr_window.shape, osize)
                                        curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1],
                                                                         cv2.BORDER_CONSTANT)
                                        curr_window = cv2.resize(curr_window, (osize, osize))
                                    ya_val.append(curr_window)

        for scene_folder in train:

            current_scene = os.path.join(current_folder, scene_folder)

            if print_step == True:
                print('-- Training, scene folder: ' + scene_folder)

            # Generate TRAIN X and y data
            fileList = os.listdir(current_scene)
            globalList = filter(lambda element: img_extension in element, fileList)

            # Validation
            if val_method == 1:
                if load_previous_val:
                    file = open('val_dict.txt', 'r')
                    val_dict = pickle.load(file)
                else:
                    tot_img_in_scene = len(globalList)
                    val_img_in_scene = int(tot_img_in_scene * val_split)
                    train_img_in_scene = int(tot_img_in_scene - val_img_in_scene)
                    if scene_folder not in val_dict:
                        val_list = random.sample(globalList, val_img_in_scene)
                        val_dict[scene_folder] = val_list

            for filename in globalList:

                current_image = os.path.join(current_scene, filename)

                # if print_step == True:
                #    print ('-- Training, image: '+ filename)
                img = cv2.imread(current_image)
                img = cv2.resize(img, (o_width, o_height))

                if normalize:
                    img = img / 255.0

                # local_data = generate_local_data(img, 1024, 436, 4, img_rows, True)
                if make_border:
                    pad = get_pad_multiple(img.shape, img_rows)
                    rimg = cv2.copyMakeBorder(img, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT)
                    rimg = cv2.resize(rimg, (img_cols, img_rows))
                else:
                    rimg = cv2.resize(img, (img_cols, img_rows))

                # X
                if experiment_folder == folders[1]:
                    # img = normalized(img, 0)
                    if val_method == 1 and filename in val_dict[scene_folder]:
                        print('-- X: Validation sample ' + filename + '...')
                        X_val.append(np.rollaxis((rimg), 2))
                    else:
                        X_train.append(np.rollaxis((rimg), 2))
                    if data_augmentation:
                        osize = img_rows
                        # cv2.imwrite('global_img.jpg', img)
                        window = np.array([o_height / w_factor, o_width / w_factor, 3])
                        img_in_slices = []
                        # Image partition
                        for widy in range(4):
                            for widx in range(4):
                                curr_window = img[widy * window[0]:(widy + 1) * window[0],
                                              widx * window[1]:(widx + 1) * window[1], 0:window[2]]
                                if make_border:
                                    pad = get_pad_multiple(curr_window.shape, osize)
                                    curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1],
                                                                     cv2.BORDER_CONSTANT)
                                    curr_window = cv2.resize(curr_window, (osize, osize))
                                Xa.append(curr_window)
                # y
                if experiment_folder == folders[0]:
                    if val_method == 1 and filename in val_dict[scene_folder]:
                        print('-- y: Validation sample ' + filename + '...')
                        y_val.append(np.rollaxis((rimg), 2))
                    else:
                        y_train.append(np.rollaxis((rimg), 2))
                    if data_augmentation:
                        osize = img_rows
                        # cv2.imwrite('global_img.jpg', img)
                        window = np.array([o_height / w_factor, o_width / w_factor, 3])
                        img_in_slices = []
                        # Image partition
                        for widy in range(4):
                            for widx in range(4):
                                curr_window = img[widy * window[0]:(widy + 1) * window[0],
                                              widx * window[1]:(widx + 1) * window[1], 0:window[2]]
                                if make_border:
                                    pad = get_pad_multiple(curr_window.shape, osize)
                                    curr_window = cv2.copyMakeBorder(curr_window, 0, pad[0], 0, pad[1],
                                                                     cv2.BORDER_CONSTANT)
                                    curr_window = cv2.resize(curr_window, (osize, osize))
                                ya.append(curr_window)

    # Final processing, to numpy arrays
    X = np.array(X_train)
    y = np.array(y_train)
    # Randomize the samples. As the images cames from a video sequence
    X, y = unison_shuffled_copies(X, y)

    if val_method > 0:
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        # Randomize the samples. As the images cames from a video sequence
        X_val, y_val = unison_shuffled_copies(X_val, y_val)

        # Write the validation dictionary
        if not load_previous_val:
            file = open('val_dict.txt', 'w')
            pickle.dump(val_dict, file)
        file.close()

    if data_augmentation:
        Xa = np.array(Xa).transpose(0, 3, 1, 2)
        ya = np.array(ya).transpose(0, 3, 1, 2)
        # Randomize the samples. As the images cames from a video sequence
        Xa, ya = unison_shuffled_copies(Xa, ya)

        if val_method == 2:
            Xa_val = np.array(Xa_val).transpose(0, 3, 1, 2)
            ya_val = np.array(ya_val).transpose(0, 3, 1, 2)

            Xa_val, ya_val = unison_shuffled_copies(Xa_val, ya_val)

            # X = np.vstack([X, Xa])
            # y = np.vstack([y, ya])
    else:
        Xa = None
        ya = None
        Xa_val = None
        ya_val = None
    if val_method < 2:
        Xa_val = None
        ya_val = None

    return X, y, X_val, y_val, Xa, ya, Xa_val, ya_val


def save_img(item, name, v, label):
    cv2.imwrite('prediction/' + name + '_x.jpg', v[item].transpose(1, 2, 0) * 255)
    cv2.imwrite('prediction/' + name + '_y.jpg', label[item].transpose(1, 2, 0) * 255)


if __name__ == '__main__':
    img_rows = 80 * 2  # 80*5 #400/4 #224 #436/4
    img_cols = 112 * 2  # 112*5 #1008/4 #224 #1024/4
    color_type = 3
    data_augmentation = False
    # X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = prepare_data(img_rows, img_cols, color_type, False, True, True)
    X, y, X_val, y_val, Xa, ya, Xa_val, ya_val = new_prepare_data(img_rows, img_cols, color_type, False,
                                                                  data_augmentation, True)