#Version 5.2
#-------------
'''
Note:
Attention!!!!!
Lon or lat
Lat or lon
Up or down
Edition 08.21

Use all month data of one 
v1:sigle month,point pair point
v2:all month ,point pair poit 
v3:all month, region pair region, use block pair block ,sperate dataLoader
version 4 add  attention 
version 5.1 add Lat and Time
version 5.2 add WeightNeighbor data ,sperate model data input
version 5.3 use TopN as WeightNerighbor,and use CT as direct input
'''

import pandas as pd
import numpy as np
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import  linregress 
import scipy
from math import ceil

#For Deep learning 
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras
from keras.optimizers import Adam
import keras.backend as K
import keras.layers as KL
from DataInput import ModelDataInput
from keras import metrics
from tensorflow.python.util.tf_export import keras_export



tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, 
#                  batch_size=32,    
                 write_graph=True, write_grads=True, 
                 write_images=True,embeddings_freq=0, embeddings_layer_names=None,embeddings_metadata=None)


#Global Var Region
BLOCK_SIZE=None
CENTER_INDEX=None
INPUT_DIM=None
NeighborNumber=None

resolution=0.25
LAT_GRID=np.linspace(0,60-resolution,int(60/resolution))
LON_GRID=np.linspace(70,140-resolution,int(70/resolution))
del resolution



channel_axis = 1 if K.image_data_format() == "channels_first" else 3


@keras_export('keras.layers.GlobalMinPool2D', 'keras.layers.GlobalMinPooling2D')
class GlobalMinPool2D(tf.keras.layers.Layer):
    def call(self, inputs):
        if K.image_data_format() == 'channels_last':
            return tf.keras.backend.min(inputs, axis=[1, 2])
        else:
            return tf.keras.backend.min(inputs, axis=[2, 3])


def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    minpool_channel=GlobalMinPool2D()(input_xs)    #TODO
    minpool_channel=KL.Reshape((1,1,channel))(minpool_channel)
    
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    #min path
    mlp_1_min=Dense_One(minpool_channel)
    mlp_2_min=Dense_Two(mlp_1_min)
    mlp_2_min=KL.Reshape(target_shape=(1,1,int(channel)))(mlp_2_min)
   
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg,mlp_2_min])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    minpool_spatial = KL.Lambda(lambda x: K.min(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial,minpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])



def v6m(input_xs,input_xa,input_xd,input_xv):  #the shape is (patch_size,patch_size,channel)
#    input_xs2=KL.Lambda(lambda x:x[:,:,:,:-1])(input_xs)
    DF1=DepthwiseConv2D(kernel_size=(2, 2),depth_multiplier=6, activation='selu', padding='valid',name='DF1')(input_xs)
    #DF2=DepthwiseConv2D(kernel_size=(2,2),depth_multiplier=4,activation='selu',padding='same',name='DF2')(DF1)
    CBAM1=cbam_module(DF1)
    #CBAM2=cbam_module(CBAM1)
    Conv1=Conv2D(4,(3,3),activation='selu',padding='valid',name='ConvF1')(CBAM1)
    Conv2=Conv2D(8,(3,3),activation='selu',padding='valid',name='ConvF2')(Conv1)
    BN1=BatchNormalization(momentum=0.2)(Conv2)
    FL=Flatten()(BN1)
    Drop1=Dropout(0.5)(FL)


    #Dense1=Dense(512,activation='selu',name='D1')(Drop1)
    Merge1=KL.Concatenate(axis=-1)([Drop1,input_xa,input_xd,input_xv])
    Dense2=Dense(64,activation='selu')(Merge1)
    Dense3=Dense(32,activation='selu')(Dense2)
    Dense4=Dense(8,activation='selu')(Dense3)
    Dense5=Dense(1,activation='linear')(Dense4)

    return Dense5

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


#TODO Note Changed model, at 3.11 2022

def build_model(TrainSet,TestSet):
    xs=Input(shape=(BLOCK_SIZE,BLOCK_SIZE,INPUT_DIM),name="BlockData")
    xa=Input(shape=(4),name="SecondaryData")   #TODO  Alert!
    xd=Input(shape=(NeighborNumber),name="WeightDistance")
    xv=Input(shape=(NeighborNumber),name="WeightValue")
    #y=v5m(xs,xa,xw)
    y=v6m(xs,xa,xd,xv)
    #model=Model(inputs=[xs,xa,xw],outputs=y)
    model=Model(inputs=[xs,xa,xd,xv],outputs=y)
    print("Test in Build model")
    optimizer = Adam()
    lr_metric = get_lr_metric(optimizer)
    model.compile(loss='logcosh', optimizer=optimizer, metrics=[metrics.mse,metrics.mae,lr_metric])  #huber_loss
    print(model.summary())
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1, cooldown=0,mode='min',min_lr=1e-6)
    h=model.fit(x=TrainSet, y=TrainSet['TargetXCO2'],validation_data=(TestSet,TestSet['TargetXCO2']),epochs=80, batch_size=500,callbacks=[reduce_lr,tbCallBack])
    show_loss(h)
    model.save("V5_model.h5")
    return model
    
def scatter_density(X,Y,pic_name):
    import mpl_scatter_density
    from matplotlib.colors import LinearSegmentedColormap
    white_jet = LinearSegmentedColormap.from_list('white_jet', [
    (0, '#ffffff'),
    (1e-20, '#00007f'),
    (0.2, '#004dff'),
    (0.4, '#29ffce'),
    (0.6, '#ceff29'),
    (0.8, '#ff6800'),
    (1, '#7f0000'),
    ], N=256)
    plt.close("all")
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(X, Y, cmap=white_jet)
    fig.colorbar(density, label='Number of points per pixel')
    plt.savefig(pic_name,dpi=600,bbox_inches='tight',pad_inches=0.0)
    plt.close('all')


def LoadData(FilePath,FileName):
    AllTrainData=np.load(FilePath+FileName,allow_pickle=True).item()
    KEYS=list(AllTrainData.keys())
    Split_Dataset=train_test_split(*list(AllTrainData.values()),test_size=0.2)
    TrainSet={}
    TestSet={}  #0 Test
    for i in range(len(Split_Dataset)):
        if i%2==0:
            TrainSet[KEYS[int(i/2)]]=Split_Dataset[i]
        else:
            TestSet[KEYS[int(i/2)]]=Split_Dataset[i]
    return TrainSet,TestSet

#def shutff_xdxv(d):
#    d["WeightDistance"][:,:]=


if __name__ == '__main__':
    
    
    BLOCK_SIZE=7
    CENTER_INDEX=int(BLOCK_SIZE/2)
    INPUT_DIM=18
    NeighborNumber=50  #TODO 
    from collections import namedtuple
    StatisticUnit = namedtuple('StatisticUnit', ['Mean', 'Std'])
    MinMaxUnit=namedtuple("MinMaxUnit",['Min','Max'])
    
    FilePath="/data/mqzhang/aux_data/ModelInputStoreV2_1/"
    TrainSet,TestSet=LoadData(FilePath,"TrainDataFileDict_03_19.npy")
    TrainSet["WeightDistance"][:,:]=TrainSet["WeightDistance"][:,:]
    np.save("TestSet.npy",TestSet)
    StandardScaler=np.load(FilePath+"StandardScalerDict_03_19.npy",allow_pickle=True).item()
    
    model=build_model(TrainSet,TestSet)
    #model = keras.models.load_model('./V5_model.h5',custom_objects={"GlobalMinPool2D":GlobalMinPool2D,'lr':get_lr_metric})
    print(StandardScaler)
    y_real=TestSet["TargetXCO2"]
    LinearTransform=lambda x:x*StandardScaler["TargetXCO2"].Std+StandardScaler["TargetXCO2"].Mean
    
    y_predict=model.predict(TestSet).reshape(-1)
    a=linregress(y_real,y_predict)
    PreLT=lambda x:(x-a.intercept)/a.slope
    y_real=LinearTransform(y_real)
    y_predict=LinearTransform(y_predict)
    plot_scatter(y_real,y_predict)

   
    np.save("Model_Evalution_Real_Predict.npy",{"Real":y_real,"Predict":y_predict})
    xco2_predict=[]
    for year in range(2003,2020):
        predict_input=np.load(FilePath+"SingleYear_"+str(year)+".npy",allow_pickle=True).item()
        Year_XCO2=model.predict(predict_input).reshape(12,240,280)
        xco2_predict.append(Year_XCO2)
    
    xco2_predict=np.vstack(xco2_predict)
    xco2_predict=LinearTransform(xco2_predict)

    #Post Process. margin Np.nan.
    xco2_predict[:,:BLOCK_SIZE,:]=np.nan
    xco2_predict[:,-BLOCK_SIZE:,:]=np.nan
    xco2_predict[:,:,:BLOCK_SIZE]=np.nan
    xco2_predict[:,:,-BLOCK_SIZE:]=np.nan
  
    np.save("model_v7_xco2.npy",xco2_predict)


    #X_train, X_test, y_train, y_test,aux2_train,aux2_test=make_model_train_data()
    #fraction=0.1
    #random_sample_train=np.random.uniform(0,X_train.shape[0],size=int(X_train.shape[0]*fraction))
    #random_sample_test=np.random.uniform(0,X_test.shape[0],size=int(X_test.shape[0]*fraction))
    #model=build_model(X_train, y_train , X_test, y_test,aux2_train,aux2_test)

