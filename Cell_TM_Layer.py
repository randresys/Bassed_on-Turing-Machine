import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
import pandas as pd 
import numpy as np

class Cell_TM(Layer):

    def __init__(self, 
                 dense_dim=10, 
                 lado_memory=40,
                 writter=0.5,
                 **kwargs):
        self.writter = writter
        self.dense_dim = dense_dim
        self.lado_memory= lado_memory

        
        super(Cell_TM, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.memory_0 = self.add_weight(name='memory_0', 
                                      shape=(self.lado_memory,self.lado_memory),
                                      initializer='RandomUniform',
                                      trainable=False)
        
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                      shape=(input_shape[1], 60),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_1 = self.add_weight(name='bias_1', 
                                      shape=(1,60),
                                      initializer='zeros',
                                      trainable=False)
        
        
        
        self.kernel_2_0=self.add_weight(name='kernel_2_0', 
                                      shape=(60,50),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_2_0 = self.add_weight(name='bias_2_0', 
                                      shape=(1,50),
                                      initializer='zeros',
                                      trainable=False)
        self.kernel_3_0 = self.add_weight(name='kernel_3_0', 
                                      shape=(60,50),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_3_0 = self.add_weight(name='bias_3_0', 
                                      shape=(1,50),
                                      initializer='zeros',
                                      trainable=False)
        self.kernel_4_0 = self.add_weight(name='kernel_3_0', 
                                      shape=(60,50),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_4_0 = self.add_weight(name='bias_3_0', 
                                      shape=(1,50),
                                      initializer='zeros',
                                      trainable=False)
        
        
        
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                      shape=(50,self.dense_dim),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_2 = self.add_weight(name='bias_2', 
                                      shape=(1,self.dense_dim),
                                      initializer='zeros',
                                      trainable=False)
        self.kernel_3 = self.add_weight(name='kernel_3', 
                                      shape=(50,self.dense_dim),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_3 = self.add_weight(name='bias_3', 
                                      shape=(1,self.dense_dim),
                                      initializer='zeros',
                                      trainable=False)
        self.kernel_4 = self.add_weight(name='kernel_4', 
                                      shape=(50,self.lado_memory),
                                      initializer='RandomUniform',
                                      trainable=False)
        self.bias_4 = self.add_weight(name='bias_4', 
                                      shape=(1,self.lado_memory),
                                      initializer='zeros',
                                      trainable=False)
        #self.kernel_4 = self.add_weight(name='1kernel_4', 
        #                              shape=(12,self.dense_dim),
        #                              initializer='RandomUniform',
        #                              trainable=True)
        #self.bias_4 = self.add_weight(name='bias_4', 
        #                              shape=(1,self.dense_dim),
        #                              initializer='zeros',
        #                              trainable=True)
        
        self.kernel_r = self.add_weight(name='kernel_r', 
                                      shape=(self.lado_memory,self.dense_dim,self.lado_memory),
                                      initializer='RandomUniform',
                                      trainable=True)

        self.memory= self.add_weight(name='memory', 
                                      shape=(self.lado_memory,self.lado_memory),
                                      initializer='RandomUniform',
                                      trainable=False)
        
        self.w_0=0.5
        
        
        self.kernel_w = self.add_weight(name='kernel_w', 
                                      shape=(self.lado_memory,self.dense_dim,self.lado_memory),
                                      initializer='RandomUniform',
                                      trainable=True)
        #self.bias_w = self.add_weight(name='bias_w', 
        #                              shape=(self.lado_memory,self.lado_memory),
        #                              initializer='RandomUniform',
        #                              trainable=True)
        self.w_sig = self.add_weight(name='w_sig', 
                                      shape=((self.lado_memory*self.lado_memory),1),
                                      initializer='RandomUniform',
                                      trainable=True)
        

        super(Cell_TM, self).build(input_shape) 
        


    def call(self, x):
        
        if tf.reduce_sum(self.memory)==0:
            self.memory=self.memory_0


        writter_f=1.0*(self.w_0)
    
        
        l1 = tf.nn.relu(tf.add(tf.matmul(x, self.kernel_1) , self.bias_1))#transform_gate 
        lfw_0 = tf.nn.relu(tf.add(tf.matmul(l1, self.kernel_3_0), self.bias_3_0))
        lfw = tf.nn.relu(tf.add(tf.matmul(lfw_0, self.kernel_3), self.bias_3))
        #l3 = K.sigmoid(tf.add(tf.matmul(l2, self.kernel_3), self.bias_3))
        #lf = K.sigmoid(tf.add(tf.matmul(l3, self.kernel_4), self.bias_4))
        lbw_0 = tf.nn.relu(tf.add(tf.matmul(l1, self.kernel_4_0), self.bias_4_0))
        lbw = K.tanh(tf.add(tf.matmul(lbw_0, self.kernel_4), self.bias_4))
        #C = tf.subtract(1.0,lbw)
        
        
        memory_w=[]
        for j in range(self.lado_memory):
            lwj = tf.cast(tf.nn.softmax(tf.matmul(lfw, self.kernel_w[j,]) ), tf.float32)
            wj=tf.multiply( tf.multiply(self.memory[j,],lwj),tf.add(1.0,tf.multiply(writter_f,lbw[j])))
            #wj=tf.add(tf.multiply(tf.multiply(self.memory[j,],lwj),lbw[j,]),tf.multiply(lwj, C[j,]))
            
            if j==0:
                memory_w=wj
            else:
                memory_w=K.concatenate([memory_w,wj],axis=0)
        
        self.memory=memory_w
        self.w_0=writter_f
         
        
        lf_0 = tf.nn.relu(tf.add(tf.matmul(l1, self.kernel_2_0), self.bias_2_0))
        lf = tf.nn.relu(tf.add(tf.matmul(lf_0 , self.kernel_2), self.bias_2))
        readed= []
        for i in range(self.lado_memory):
            lri=tf.cast(tf.nn.softmax(tf.matmul(lf, self.kernel_r[i,]) ), tf.float32)
            ri=tf.multiply(self.memory[i,],lri)
            if i==0:
                readed=ri
            else:
                readed=K.concatenate([readed,ri],axis=1)

        l = K.sigmoid(tf.matmul(readed, self.w_sig))
        #C = tf.subtract(1.0,T) #carry_gate
        #y = tf.add( tf.multiply(H, T),  tf.multiply(x, C))
        return l

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
