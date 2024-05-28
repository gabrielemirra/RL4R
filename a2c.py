# basic tensorflow imports
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Conv2D, LSTM, Reshape, Activation, MaxPooling2D, Dropout, Input, Lambda, LeakyReLU, ReLU, Softmax, Conv2DTranspose, BatchNormalization,Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import TensorBoard

# other imports

import cv2, sys, random, time, datetime, collections
import math as m
import numpy as np
from collections import deque
from matplotlib import pyplot

# define custom layers

class residual_block(keras.layers.Layer):
    def __init__(self, nb_channels):
        super(residual_block, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        self.model.add(ReLU())
        self.model.add(Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    def call(self, y):
        residual = self.model(y)

        residual = Add()([residual, y])

        return ReLU()(residual)

class Location(keras.layers.Layer):
    def __init__(self, shape):
        super(Location, self).__init__()
        self.w,self.h = shape
        self.linear = Dense(16)

    def call(self, action):
        
        with tf.device('/CPU:0'):
            remainder_1 = tf.math.floormod(action, self.w) #elementwise remainder of division (equivalent to numpy.remainder)
            remainder_2 = tf.math.floormod(action, self.w**2)

        x = -1 + 2 *(remainder_2 - remainder_1) / (self.w - 1) / self.w
        y = -1 + 2 *(remainder_1) / (self.w - 1)

        action = tf.stack([y,x],axis = 1)
        action = tf.squeeze(action, axis=-1)

        action = Flatten()(action)
        return self.linear(action)

class Scalar(keras.layers.Layer):
    def __init__(self, shape):
        super(Scalar, self).__init__()

        self.linear = Dense(16)
        self._shape = shape

    def call(self, action):
        action = tf.one_hot(tf.squeeze( K.cast(action, dtype='int32'), axis=1), self._shape)
        return self.linear(action)

class MaskMLP(keras.layers.Layer):
    def __init__(self, action_shape, grid_shape): # 
        super(MaskMLP, self).__init__()
        self.w, self.h = grid_shape
        self.modules = []
        for i, shape in enumerate(action_shape):
            if i < 1: ############################################################ change according to the number of spatial actions
                module = Location(grid_shape)
            else:
                module = Scalar(shape)
            self.modules.append(module)

    def call(self, action, action_mask = None):
        actions = tf.unstack(action,axis=1)        
        y = [self.modules[i](tf.expand_dims(action_i,axis=1)) for i, action_i in enumerate(actions)]
        y = tf.stack(y, axis=1)
        y = Flatten()(y)
        return y

class _grid(keras.layers.Layer): #batch is B*T
    def __init__(self, batch, h, w):
        super(_grid, self).__init__()
        y_grid = tf.linspace(-1, 1, h)
        y_grid = tf.reshape(y_grid,(1, h, 1, 1))
        self.y_grid = tf.tile(y_grid,(batch, 1, h, 1))

        x_grid = tf.linspace(-1, 1, w)
        x_grid = tf.reshape(x_grid,(1, 1, w, 1))
        self.x_grid = tf.tile(x_grid,(batch, w, 1, 1))

    def call(self, input_tensor):
        grid = tf.concat([self.y_grid, self.x_grid], axis=-1)
        grid = tf.cast(grid, tf.float32)
        return  tf.concat([input_tensor,grid],axis=-1)#concatenate along C

# define model architecture

class Decoder(Model):
    def __init__(self, order, action_shape, grid_shape):
        super(Decoder, self).__init__()

        self.SPATIAL_ACTIONS = ["end_point"]
        self.ORDER = ["end_point", "id", "place_flag"]

        self._order = [k for k in self.ORDER if k in order] #this allows to keep the same order regardless of the implementation
        self._action_order = order # sequence of string identifiers for actions

        action_shape = dict(zip(self._action_order, action_shape))

        modules = {}

        for k in self._action_order: 
            if k in self.SPATIAL_ACTIONS: #if the action is spatial
                module = tf.keras.Sequential([
                    Reshape((4,4,16),input_shape = (256,)),
                    Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'),
                    *[residual_block(32) for _ in range(4)],
                    Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'),
                    Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'),
                    Conv2D(1, (3,3),strides = (1,1), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
                    Flatten()
                    ])
            else:
                module = Dense(action_shape[k], kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
            modules[k] = module
        self.decode = modules

        # get logits

        modules = {}
        for k in self._order[:-1]: # the last one is unnecessary because it doesn't get concatenated with the previous embedding to sample new actions #
            if k in self.SPATIAL_ACTIONS:
                module = Location(grid_shape)
            else:
                module = Scalar(action_shape[k])
            modules[k] = module 

        self.mlp = modules

        self.concat_fc = Dense(256, activation='relu')

        self.relu = ReLU()

    def call(self, h, actions=None, is_inference=False): #h shape is (B*T,256). each h[i] returns a list of logits and actions (i.e. for every state -> tuple of actions)        
        dict_logits = collections.OrderedDict({k: None for k in self._action_order})

        if is_inference==False: # actions are provided, no sampling

            dict_actions = collections.OrderedDict(zip(self._action_order, tf.unstack(tf.expand_dims(actions,-1), axis=1))) 
            for k in self._order: #the decoding process is perfomed in parallel for each action component

                logit = self.decode[k](h)
                dict_logits[k] = logit

                if k ==self._order[-1]:
                    break

                concat = tf.concat([h, self.mlp[k](dict_actions[k])], axis=1) #dic_actions is the input for the modules
                residual = self.concat_fc(concat)
                h = self.relu(h + residual)

            
            logits = list(dict_logits.values())
            logits = tf.concat(logits,axis=-1)
            
            return actions, logits

        else:
            dict_actions = collections.OrderedDict({k: None for k in self._action_order})
        
            for k in self._order:
                logit = self.decode[k](h)

                dist = tf.compat.v1.distributions.Categorical(probs=tf.nn.softmax(logit))
                action = dist.sample()
                action = tf.expand_dims(action,-1)

                dict_actions[k] = action
                dict_logits[k] = logit

                if k == self._order[-1]:
                    break
                concat = tf.concat([h,self.mlp[k](action)],axis=1) #mlp is either scalar or location
                residual = self.concat_fc(concat)
                h = self.relu(h + residual)

            actions = tf.concat(list(dict_actions.values()), axis=1)
            logits = list(dict_logits.values())
        
            return actions, logits

def build_policy(batch_size, episode_len, state_size, action_shape, stock_size, is_inference=False):

    if is_inference:
        X_input = Input(batch_shape = (1,state_size[0],state_size[1])) #canvas
        S_input = Input(batch_shape = (1,stock_size)) #stock vector
        A_input = Input(batch_shape = (1,len(action_shape)))
        M_input = Input(batch_shape = (1,len(action_shape)))        
        N_input = Input(batch_shape = (1,10))
    else:
        X_input = Input((state_size[0],state_size[1]))
        S_input = Input((stock_size))
        A_input = Input((len(action_shape))) #previous actions
        M_input = Input((len(action_shape))) #previous actions        
        N_input = Input((10))
        stored_actions = Input((len(action_shape)))
        ht = Input((256))
        ct = Input((256))


    A = MaskMLP(action_shape=action_shape, grid_shape=(32,32))(A_input,M_input)
    A = Dense(64, activation='relu')(A)
    A = Dense(32, activation='relu')(A)
    A = Dense(32, activation='relu')(A)
    
    N = Dense(64, activation='relu')(N_input)
    N = Dense(32, activation='relu')(N)
    N = Dense(32, activation='relu')(N)

    S = Dense(64, activation='relu')(S_input)
    S = Dense(32, activation='relu')(N)
    S = Dense(32, activation='relu')(N)

    #condition = Add()([A,N,S])
    condition = Reshape((1,1,32))(A)

    X = Reshape((state_size[0],state_size[1],1),input_shape = (state_size[0],state_size[1]))(X_input)
    
    if is_inference:
        X = _grid(1, state_size[0], state_size[1])(X)
    else:
        X = _grid(batch_size*episode_len, state_size[0], state_size[1])(X) #concatenate canvas with grid
    
    X = Conv2D(32, (5,5),strides = (1,1), padding='same')(X)

    X_A_N = Add()([X,condition])
    conditioned_X = ReLU()(X_A_N)

    X = Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')(conditioned_X)
    X = Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')(X)
    X = Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')(X)

    for _ in range(4):
        X = residual_block(32)(X)

    X = Flatten()(X)
    embedding = Dense(256, activation='relu')(X)

    # ----------------------

    if is_inference:
        X_ = K.reshape(embedding, (-1,1,256))
        lstm_out, hidden_state, cell_state = LSTM(256, stateful=True, return_state=True,return_sequences=True,name='lstm')(X_)
        seed = K.reshape(lstm_out, (-1,256))
    else:
        X_ = K.reshape(embedding, (-1,episode_len,256))
        lstm_out, hidden_state, cell_state = LSTM(256, return_state=True, return_sequences=True)(X_,initial_state=[ht,ct])
        seed = K.reshape(lstm_out, (-1,256))

    # ------------------------

    value = Dense(1)(seed)
    
    #decoder

    decoder = Decoder(["end_point", "id", "place_flag"],action_shape,(32,32))

    if is_inference:
        actions_,logits_ = decoder(seed, actions=None, is_inference=is_inference) # if inference do not provide actions, there is sampling. otherwise provide actions from batch
    else:
        actions_,logits_ = decoder(seed, actions=stored_actions, is_inference=is_inference)


    if is_inference:
        return Model(inputs = [X_input, S_input, A_input, M_input, N_input], outputs = [actions_, logits_, value, hidden_state, cell_state])
    else:
        return Model(inputs = [X_input, S_input, A_input, M_input, N_input, stored_actions, ht, ct], outputs = [actions_, logits_, value]) #, Model(inputs = [X_input, A_input, M_input, N_input, stored_actions], outputs = value)

######################################################################################################################################

