#######################################################
# Model definitions
# 06/09/2019 
#
#######################################################
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import cv2
from math import floor


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, regularizers

# defining global variables
DEBUG_MODE = False
seed        = 29 # fix random seed for reproducibility
np.random.seed(seed)
optmz       = 'adam'    # optimizers.RMSprop(lr=0.0001)
modelname   = 'CA2'
num_classes = 3
imgrows = 224
imgclms = 224
channel = 3
input_shape = (imgrows, imgclms, channel)

def my_preprocess(x):
    # zero-center by mean pixel per channel
    x /= 255
    mean = [0.496, 0.475, 0.343] # obtained from ISY5002_CA2_01_Preprocess getPerChannelMeansAndStd()
    std = [0.280, 0.258, 0.282]  # obtained from ISY5002_CA2_01_Preprocess getPerChannelMeansAndStd()
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]

    return x

# ----------------------------------------------------------------------------
# Define the deep learning models
# ----------------------------------------------------------------------------
def createModel(imgrows, imgclms, channel, index = 0):

    # Edmund - Use indexes 0 to 10
    if (index <=0): return createE0(imgrows, imgclms, channel)
    elif (index == 1): return createE1(imgrows, imgclms, channel)
    # Wilson - Use indexes 11 to 20
    elif (index == 11): return createModel_CNN1(imgrows, imgclms, channel)
    elif (index == 12): return createModel_CNN2(imgrows, imgclms, channel)
    elif (index == 15): return createModel_ResNet1(imgrows, imgclms, channel)
    elif (index == 16): return createModel_ResNet2(imgrows, imgclms, channel)
    # Meiying - Use indexes 21 to 30
    elif (index == 21): return None
    #
    # Popular pretrained models - Use indexes 90 to 100
    elif (index == 90): return createModel_transferLearning(imgrows, imgclms, channel, mod='vgg16')
    elif (index == 91): return createModel_transferLearning(imgrows, imgclms, channel, mod='resnet')
    elif (index == 92): return createModel_transferLearning(imgrows, imgclms, channel, mod='inception')
    else: return None

def createE0(imgrows=imgrows, imgclms=imgclms, channel=channel):
    # Basic model - no batch normalisation, no dropout
    xin = Input(shape=(imgrows, imgclms, channel))
    x = Conv2D(32,(3,3), padding='same')(xin)
    x = Activation('relu')(x)

    x = Conv2D(32,(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(32,(3,3), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(64,(3,3), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(64,(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128,(3,3), padding='same')(x) 
    x = Activation('relu')(x)

    x = Conv2D(128,(3,3), padding='same')(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=xin, outputs=x)
    model.compile(loss='categorical_crossentropy', 
      optimizer=optmz, metrics=['accuracy'])

    return model


def createE1(imgrows=imgrows, imgclms=imgclms, channel=channel):
    xin = Input(shape=(imgrows, imgclms, channel))
    x = Conv2D(32,(3,3), padding='same')(xin)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32,(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(32,(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64,(3,3), padding='same')(x)
    x = BatchNormalization()(x)     # Do batch normalisation BEFORE activation
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64,(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128,(3,3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation
    x = BatchNormalization()(x)     # Do batch normalisation BEFORE activation
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128,(3,3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.50)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=xin, outputs=x)
    model.compile(loss='categorical_crossentropy', 
      optimizer=optmz, metrics=['accuracy'])

    return model



from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.initializers import he_normal 
from tensorflow.python.keras.optimizers import Adam
def createModel_transferLearning(imgrows=imgrows, imgclms=imgclms, channel=channel, mod='vgg16'):

    m_optmz = optmz
    if (mod == 'vgg16'):
      model = VGG16(include_top=False, input_tensor=Input(shape=(imgrows,imgclms,channel)))
    elif (mod == 'resnet'):
      model = ResNet50(include_top=False, input_tensor=Input(shape=(imgrows,imgclms,channel)))
      m_optmz = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # adam_fine (20x smaller than standard)

    elif (mod == 'inception'):
      model = InceptionV3(include_top=False, input_tensor=Input(shape=(imgrows,imgclms,channel)))
      
    
    # mark loaded layers as not trainable
    for layer in model.layers:
      layer.trainable = False

    x = Flatten()(model.outputs[0])
    x = Dense(1024, activation='relu', kernel_initializer=he_normal(33))(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model.inputs, outputs=x)
    model.compile(loss='categorical_crossentropy', 
      optimizer=m_optmz, metrics=['accuracy'])
    return model

# ----------------------------------------------------------------------------
# Define the deep learning CNN models
# ----------------------------------------------------------------------------

# CNN1 Model

def createModel_CNN1(imgrows=imgrows, imgclms=imgclms, channel=channel):
    
  xin = Input(shape=(imgrows, imgclms, channel))
  
  x = Conv2D(64, (3, 3), padding='same')(xin)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(128, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Flatten()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = BatchNormalization()(x)
  x = Dense(192, activation='relu')(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.4)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)

  x = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)

  model = Model(inputs=xin, outputs=x)
  model.compile(loss='categorical_crossentropy', 
      optimizer=optmz, metrics=['accuracy'])
  
  return model


# CNN2 Model
   
def createModel_CNN2(imgrows=imgrows, imgclms=imgclms, channel=channel):
    
  xin = Input(shape=(imgrows, imgclms, channel))
  
  x = Conv2D(128, (3, 3), padding='same')(xin)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(256, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.001))(x) # With L2 regularisation)(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Flatten()(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = BatchNormalization()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.3)(x)
  x = BatchNormalization()(x)
  x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.4)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)

  x = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(x)

  model = Model(inputs=xin, outputs=x)
  model.compile(loss='categorical_crossentropy', 
      optimizer=optmz, metrics=['accuracy'])
  
  return model

# ----------------------------------------------------------------------------
# Define the deep learning ResNets models
# ----------------------------------------------------------------------------

# ResNet1 Model

def resLyr(inputs,
           numFilters=16,
           kernelSz=3,
           strides=1,
           activation='relu',
           batchNorm=True,
           convFirst=True,
           lyrName=None):

    convLyr     = Conv2D(numFilters,
                     kernel_size=kernelSz,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=lyrName+'_conv' if lyrName else None)
    x           = inputs
    if convFirst:
        x       = convLyr(x)
        if batchNorm:
            x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x   = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
    else:
        if batchNorm:
            x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x   = Activation(activation,
                           name=lyrName+'_'+activation if lyrName else None)(x)
        x       = convLyr(x)

    return x

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

def lrSchedule(epoch):
    lr  = 1e-3
    
    if epoch > 90:
        lr  *= 0.5e-3
        
    elif epoch > 80:
        lr  *= 1e-3
        
    elif epoch > 60:
        lr  *= 1e-2
        
    elif epoch > 30:
        lr  *= 1e-1
        
    print('Learning rate: ', lr)
    
    return lr

def resBlkV1(inputs,
             numFilters=16,
             numBlocks=3,
             downsampleOnFirst=True,
             names=None):
    
    x = inputs    
    for run in range(0,numBlocks):
        strides = 1
        blkStr  = str(run+1) 
        if downsampleOnFirst and run == 0:
            strides     = 2        
        y       = resLyr(inputs=x,
                     numFilters=numFilters,
                         strides=strides,
                         lyrName=names+'_Blk'+blkStr+'_Res1' if names else None)
        y       = resLyr(inputs=y,
                         numFilters=numFilters,
                         activation=None,
                         lyrName=names+'_Blk'+blkStr+'_Res2' if names else None)   
        if downsampleOnFirst and run == 0:
            x   = resLyr(inputs=x,
                         numFilters=numFilters,
                         kernelSz=1,
                         strides=strides,
                         activation=None,
                         batchNorm=False,
                         lyrName=names+'_Blk'+blkStr+'_lin' if names else None)
          
        x       = add([x,y], name=names+'_Blk'+blkStr+'_add' if names else None)
        x       = Activation('relu',
                             name=names+'_Blk'+blkStr+'_relu' if names else None)(x)    
        
    return x
    

def createResNetV1(inputShape=input_shape,
                   numClasses=num_classes):
    inputs      = Input(shape=inputShape)
    v           = resLyr(inputs,
                         lyrName='Inpt')
    v           = resBlkV1(inputs=v,
                           numFilters=16,
                           numBlocks=5,
                           downsampleOnFirst=False,
                           names='Stg1')
    v           = resBlkV1(inputs=v,
                             numFilters=32,
                             numBlocks=5,
                             downsampleOnFirst=True,
                             names='Stg2')
    v           = resBlkV1(inputs=v,
                           numFilters=64,
                           numBlocks=8,
                           downsampleOnFirst=True,
                           names='Stg3')
    v           = AveragePooling2D(pool_size=8,
                                   name='AvgPool')(v)
    v           = Flatten()(v)
    v           = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.25)(v)
    v           = BatchNormalization()(v)
    v           = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.3)(v)
    v           = BatchNormalization()(v)
    v           = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.35)(v)
    v           = BatchNormalization()(v)
    v           = Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.4)(v)
    v           = BatchNormalization()(v)
    v           = Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.5)(v)
    v           = BatchNormalization()(v)
    v           = Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.5)(v)
    v           = BatchNormalization()(v)

    outputs     = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(v)
    model       = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optmz, 
                  metrics=['accuracy'])
    return model
 
def createResNetV2(inputShape=input_shape,
                   numClasses=num_classes):
    inputs      = Input(shape=inputShape)
    v           = resLyr(inputs,
                         lyrName='Inpt')
    v           = resBlkV1(inputs=v,
                           numFilters=16,
                           numBlocks=3,
                           downsampleOnFirst=False,
                           names='Stg1')
    v           = resBlkV1(inputs=v,
                             numFilters=32,
                             numBlocks=5,
                             downsampleOnFirst=True,
                             names='Stg2')
    v           = resBlkV1(inputs=v,
                           numFilters=64,
                           numBlocks=5,
                           downsampleOnFirst=True,
                           names='Stg3')
    v           = AveragePooling2D(pool_size=8,
                                   name='AvgPool')(v)
    v           = Flatten()(v)
    v           = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.25)(v)
    v           = BatchNormalization()(v)
    v           = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.3)(v)
    v           = BatchNormalization()(v)
    v           = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.35)(v)
    v           = BatchNormalization()(v)
    v           = Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(v)
    v           = Dropout(0.4)(v)
    v           = BatchNormalization()(v)

    outputs     = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(v)
    model       = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optmz, 
                  metrics=['accuracy'])
    return model

def createModel_ResNet1(imgrows=imgrows, imgclms=imgclms, channel=channel):
    
    input_shape = (imgrows, imgclms, channel)
    
    model       = createResNetV1(input_shape,num_classes)
    
    return model
 
def createModel_ResNet2(imgrows=imgrows, imgclms=imgclms, channel=channel):
    
    input_shape = (imgrows, imgclms, channel)
    
    model       = createResNetV2(input_shape,num_classes)
    
    return model