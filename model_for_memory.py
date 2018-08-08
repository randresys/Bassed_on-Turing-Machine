
import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.preprocessing import  MaxAbsScaler
from keras import optimizers
from sklearn.externals import joblib
import pandas as pd 
from pandas import Series
import matplotlib.pyplot as plt
import numpy as np
from Cell_TM_Layer import Cell_TM as cMemory

data = pd.read_csv("energydata_complete.csv",encoding = "ISO-8859-1")
print(data.shape)

data["date"] = pd.to_datetime(data["date"])
data["dayofyear"] = data['date'].dt.dayofyear
#data["dayofweek"] = data['date'].dt.dayofweek
data["hour"] = data['date'].dt.hour

data=data.sort_values(by=['rv1'],ascending=False)
data=data.drop(['rv1', 'rv2',"date", "Visibility"], axis=1)
print(data.shape)
print(data.dtypes)
def add_item(list, item):
    list.append(item)
    return list
data=np.asarray(data)
print(data.shape)
Appliances=data[:,0]
Appliances=Appliances.reshape(Appliances.shape[0],1)
lights =data[:,1]
data=np.delete(data,0,1)
print(data.shape,Appliances.shape)

data_conca=np.concatenate((data,Appliances ), axis=1)

data_train=data_conca[0:18000,]
data_test=data_conca[18000:19700,]
data_pred=data_conca[19700:19735,]
print(data.shape)

def create_train(data,look_back,steps_future):
    label, features = [], []
    for i in range(data.shape[0]-look_back-steps_future):
        row=data[i:i+look_back]
        features.append(row)
        
        a=data[i+look_back+steps_future,6]
        
        label.append(a)
    return np.array(features), np.array(label)
 
x,y=create_train(data_train,13,2)
r=np.random.choice(x.shape[0], x.shape[0], replace=False)
x=x[r,:,:]
y=y[r]
#x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])


x_test,y_test=create_train(data_test,13,2)
r_test=np.random.choice(x_test.shape[0], x_test.shape[0], replace=False)
x_test=x_test[r_test,:,:]
y_test=y_test[r_test]
#x_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])


x1,y1=create_train(data_pred,13,2)
r1=np.random.choice(x1.shape[0], x1.shape[0], replace=False)
x1=x1[r1,:,:]
y1=y1[r1]
#x1=x1.reshape(x1.shape[0],x1.shape[1]*x1.shape[2])


print(x.shape,y.shape,x_test.shape,y_test.shape)       


print(y_test.shape)
y=y.reshape(-1, 1)
y_test=y_test.reshape(-1,1)
scaler = MaxAbsScaler()
scaler.fit(y)
scaler_filename = "scalerEnergy.save"
joblib.dump(scaler, scaler_filename) 
y=scaler.transform(y)
y_test=scaler.transform(y_test)

batch_size=100

input_general = Input(shape=(x.shape[1],x.shape[2],), name='input_general')
model1= BatchNormalization()(input_general)
model1=Flatten()(model1)
main_output= cMemory(name='main_output')(model1)


model = Model(inputs=[input_general], output=[main_output])

#a loss funtion to differents scenarios
def mega_loss (y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #loss1=K.mean(tf.abs(tf.abs(y_true_f-y_pred_f )**3)
    loss1= tf.reduce_mean(tf.squared_difference(tf.log(y_true_f+1),tf.log(y_pred_f+1)))
    #loss2= tf.reduce_mean(tf.squared_difference(tf.log(t_true+1),tf.log(t_pred+1)))
    loss3= tf.reduce_mean(tf.abs(tf.abs((y_true_f-y_pred_f ))**3))
    #loss=tf.add(tf.multiply(loss1,0.99),tf.multiply(loss2,0.01)) 
    loss4=K.categorical_crossentropy(y_true_f,y_pred_f)
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_f, logits=y_pred_f)
    loss = tf.reduce_mean(xent, name='loss')
    return loss3
 
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

optimizer=optimizers.Adam()
lr_metric = get_lr_metric(optimizer)

model.compile(optimizer= optimizer,
              loss='logcosh' ,metrics=['accuracy','mae',lr_metric])

filepath="weights_model1_ntm_mea.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
filepath0="weights_model1_ntm_valmea.h5"
checkpoint0 = ModelCheckpoint(filepath0, monitor='mean_absolute_error', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint,checkpoint0]
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.1 ,
                              patience=5, min_lr=0.0000000000000000000001)

history=model.fit(

                  x,y,
                  epochs=100, batch_size=batch_size,callbacks=callbacks_list+[reduce_lr],
                  validation_data=(x_test,y_test)
                  )
print(history.history.keys())

plt.plot(history.history['loss'])
plt.figsize=(16,20)
plt.title('loss') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='best')
plt.savefig('loss_modelo_energy.png')
plt.show()

plt.plot(history.history['val_loss'])
plt.figsize=(16,20)
plt.title('val_loss') 
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['test'], loc='best')
plt.savefig('val_loss_modelo_energy.png')
plt.show()

fig=plt.plot(history.history['mean_absolute_error'])
plt.figsize=(16,20)
plt.title('mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train'], loc='best')
plt.savefig('mae_modelo_energy.png')
plt.show()

fig=plt.plot(history.history['val_mean_absolute_error'])
plt.figsize=(16,20)
plt.title('val_mean_absolute_error')
plt.ylabel('val_mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['test'], loc='best')
plt.savefig('val_modelo_energy.png')
plt.show()

fig=plt.plot(history.history['acc'])
plt.figsize=(16,20)
plt.title('model accuracy')
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='best')
plt.savefig('acc_modelo_energy.png')
plt.show()
