import json
import os
import random
from ast import literal_eval as make_tuple
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from neural_nets.scripts.models import MODELS
from q_sigmoid.formula import qsigmoid

SM_FRAMEWORK=tf.keras
INPUT_SHAPE = (None,None,1)
TEST_PATH = r"D:\Development\Zigante\cardiomyopathy-monograph\SCRIPTS\test_paths"
TRAIN_PATH = r"D:\Development\Zigante\cardiomyopathy-monograph\SCRIPTS\train_paths"
PATH_PREDS = r'D:\Development\Zigante\cardiomyopathy-monograph\PREDICOES'
EPOCHS = 100
BATCH_SIZE = 4
TRAIN_TYPE = 'qsigmoid'
MAX_BLACK_IMAGES = 300

class CustomMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

class Loader:
    def load_incor_image(self,path):
        image = plt.imread(path)[:,:,0]
        slice_image = np.array(image)
        slice_image = resize(slice_image, (128,128), order=0, preserve_range=True, anti_aliasing=False)
        slice_image = np.nan_to_num(slice_image,0)

        if 'gt' in path:
            slice_image[(slice_image != 1) & (slice_image != 0)] = 2
            return tf.one_hot(slice_image.astype(np.int64),3)
        else:
            slice_image = slice_image.reshape(slice_image.shape[0],slice_image.shape[1],1)
            maior = np.max(slice_image) if np.max(slice_image) > 0 else 1
            slice_image = slice_image/maior
            if 'qsigmoid' in TRAIN_TYPE:
                slice_image = qsigmoid(slice_image)
            return slice_image

    def load_acdc(self,path):
        slices = []

        image = nib.load(path)
        image_array = image.get_fdata()
        for _slice in range(image_array.shape[2]):

            slice_image = np.array(image_array[:,:,_slice])
            slice_image = resize(slice_image, (128,128), order=0, preserve_range=True, anti_aliasing=False)


            if 'gt' in path:
                slice_image[slice_image==1] = 0
                slice_image[slice_image==2] = 1
                slice_image[slice_image==3] = 2
                slices.append(tf.one_hot(slice_image.astype(np.int64),3))
            else:
                slice_image = slice_image.reshape(slice_image.shape[0],slice_image.shape[1],1)
                maior = np.max(slice_image) if np.max(slice_image) > 0 else 1
                slice_image = slice_image/maior

                if 'qsigmoid' in TRAIN_TYPE:
                    slice_image = qsigmoid(slice_image)
                slices.append(slice_image)

        return slices

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, batch_size = 4):
        self.paths = paths
        self.batch_size = batch_size
        self.loader = Loader()
        self.get_indexes()

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        paths_to_read = [self.paths[_index] for _index in indexes]
        X,y = self.__getdata(paths_to_read)

        return X,y

    def get_indexes(self):
        self.indexes = np.arange(len(self.paths))

    def __getdata(self,paths_to_read):
        X, y = [],[]

        for path_tuple in paths_to_read:
            if 'png' in path_tuple[0]: # incor
                X.append(self.loader.load_incor_image(path_tuple[0]))
                y.append(self.loader.load_incor_image(path_tuple[1]))
            else: # acdc nii
                for image_x in self.loader.load_acdc(path_tuple[0]):
                    X.append(image_x)
                for image_y in self.loader.load_acdc(path_tuple[1]):
                    y.append(image_y)

        return np.array(X),np.array(y)

def get_black_list_images_incor():
    if not os.path.exists('black_images_incor.json'):
        print('[WARN] black_images_incor.json not exists! All images will be used')
        return set()

    with open('black_images_incor.json') as file:
        return set(json.load(file)['patient_black_images'])

def get_data_train_test_paths(
        acdc_path=r'D:/Dados/Eduardo Coltri/TCC2/ACDC',
        incor_path=r'D:/Dados/Eduardo Coltri/TCC2/INCOR',
        load_from_archive = False,
        archive_train_path = None,
        archive_test_path = None,
        train_size_percentile = 0.7):


    if load_from_archive:
        assert archive_train_path and archive_test_path, "Archive path must be passed!"
        train_paths = []
        test_paths = []
        with open(archive_train_path,'r') as file:
            for line in file:
                train_paths.append(make_tuple(line))

        with open(archive_test_path,'r') as file:
            for line in file:
                test_paths.append(make_tuple(line))

        return train_paths, test_paths

    assert train_size_percentile < 1. and train_size_percentile > 0, "invalid range! Should be more than 0. and less than or equal to 1."
    paths_grouped = {}
    for path in Path(acdc_path).rglob('*.nii.gz'):
        if 'frame' not in path.name:
            continue

        path_name = path.name
        current_patient_frame = path_name.split('.')[0].split('_gt')[0]

        if current_patient_frame not in paths_grouped:
            paths_grouped[current_patient_frame] = []

        paths_grouped[current_patient_frame].append(str(path))

    black_images_incor = get_black_list_images_incor()
    for path in Path(incor_path).rglob('*.png'):
        path_name = path.name
        current_patient_frame = path_name.split('.')[0].split('_gt')[0]

        if current_patient_frame not in paths_grouped:
            paths_grouped[current_patient_frame] = []

        paths_grouped[current_patient_frame].append(str(path))


    paths = []
    INCOR_MAX = 0
    TOTAL_BLACK_IMAGES = 0
    for group_key in paths_grouped:
        record = paths_grouped[group_key]
        if 'png' in record[0] and record[0] not in black_images_incor:
            INCOR_MAX += 1
            paths.append(tuple(record))
        if 'nii' in record[0]:
            paths.append(tuple(record))
        elif TOTAL_BLACK_IMAGES < MAX_BLACK_IMAGES:
            paths.append(tuple(record))
            TOTAL_BLACK_IMAGES +=1

    paths_shuffle = random.sample( paths, len(paths) )
    train_index_split = int(len(paths)*train_size_percentile)
    return paths_shuffle[:train_index_split],paths_shuffle[train_index_split:]

def get_generators(train, test, batch_size=8):
    train_generator = DataGenerator(train,batch_size=batch_size)
    test_generator = DataGenerator(test,batch_size=batch_size)
    return train_generator, test_generator

def dice_coef(y_true, y_pred,smooth=1e-7):
    print(f'y_true min = {tf.reduce_min(y_true)}, y_pred min = {tf.reduce_min(y_pred)}')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def multiclass_hausdorff(y_true,y_pred, labels = 3):
    res = 0
    for label in range(labels):
        res += directed_hausdorff(y_true[:,:,label],y_pred[:,:,label])[0]
    return res

def compute_iou(y_true,y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def enable_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('problems when setting memory growth!')
        pass

def get_model_checkpoint(model_name):
    return tf.keras.callbacks.ModelCheckpoint(
        os.path.join('outputs',f'{model_name}_weights.h5'),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch"
    )

MODEL_TO_USE = 'unet_bce_jaccard_loss_relu'

enable_memory_growth()
train,test =  get_data_train_test_paths()
train_generator, test_generator = get_generators(train,test, batch_size=BATCH_SIZE)

print('DATASET OK')

steps_per_epoch_train = (len(train)) // BATCH_SIZE
steps_per_epoch_test = (len(test)) // BATCH_SIZE

print(f'steps_per_epoch_train = {steps_per_epoch_train} | steps_per_epoch_test = {steps_per_epoch_test}')


model = MODELS[MODEL_TO_USE]()
print(f'using model {MODEL_TO_USE}')

model_checkpoint_callback = get_model_checkpoint(MODEL_TO_USE)

tf.keras.backend.clear_session()
history = model.fit(train_generator,callbacks=[model_checkpoint_callback],epochs = EPOCHS,validation_data= test_generator,steps_per_epoch = steps_per_epoch_train,validation_steps = steps_per_epoch_test)

model.save(f'{MODEL_TO_USE}_{TRAIN_TYPE}')

dice_index = []
hausdorff = []
iou = []
df = pd.DataFrame(history.history)
df.to_csv(f'{MODEL_TO_USE}_{TRAIN_TYPE}_history_results.csv')
iou_index = CustomMeanIOU(num_classes=3)

for index in range(len(test_generator)):

    if index > EPOCHS:
        break

    batch = test_generator[index]
    preds = model.predict(batch[0])
    root = os.path.join(PATH_PREDS,MODEL_TO_USE)

    if not os.path.exists(root):
        os.mkdir(root)

    for image_index in range(preds.shape[0]):
        plt.imsave(os.path.join(root,f'y_predicted_batch_{index}_{image_index}.png'),np.argmax(preds[image_index],axis=-1))
        plt.imsave(os.path.join(root,f'y_true_batch_{index}_{image_index}.png'),np.argmax(batch[1][image_index],axis=-1))
        _ = iou_index.update_state(batch[1][image_index],preds[image_index])
        dice_index.append(dice_coef(batch[1][image_index],preds[image_index]).numpy())
        hausdorff.append(multiclass_hausdorff(batch[1][image_index],preds[image_index]))
        iou.append(iou_index.result().numpy())

df = pd.DataFrame()
df['dice'] = dice_index
df['hausdorff'] = hausdorff
df['iou'] = iou
df.to_csv(f'{MODEL_TO_USE}_{TRAIN_TYPE}_metrics_results.csv')

df = pd.DataFrame()
df['x_test'] = [_tuple[0] for _tuple in test]
df['y_test'] = [_tuple[1] for _tuple in test]
df.to_csv(f'{MODEL_TO_USE}_{TRAIN_TYPE}_images_for_test.csv')
