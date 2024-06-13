import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style= "darkgrid", color_codes = True)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import  load_img
import warnings
warnings.filterwarnings('ignore')


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True



from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile


IMAGE_SIZE = [150, 150]

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                    rotation_range = 20,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.1,
                                    zoom_range = 0.1,
                                    horizontal_flip = True

                                   )

test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory('/content/gdrive/MyDrive/Pneumonia_Detect/chest_xray/train',
                                                    batch_size = 32,
                                                    class_mode = 'categorical',
                                                    target_size = (150, 150))

validation_generator =  test_datagen.flow_from_directory( '/content/gdrive/MyDrive/Pneumonia_Detect/chest_xray/val',
                                                          batch_size  = 32,
                                                          class_mode  = 'categorical',
                                                          target_size = (150, 150),
                                                          shuffle = False
                                                    )

test_generator = test_datagen.flow_from_directory('/content/gdrive/MyDrive/Pneumonia_Detect/chest_xray/test',
    batch_size=32,
    class_mode='categorical',
    target_size=(150, 150),
    shuffle=False
)