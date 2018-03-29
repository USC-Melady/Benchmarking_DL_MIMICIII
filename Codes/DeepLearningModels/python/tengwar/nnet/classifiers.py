# A package including neural network classes, which provide the same methods as what sklearn does

from copy import deepcopy
import sys

from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Merge, BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM, GRU 
from keras.models import model_from_json, Sequential
from keras.optimizers import RMSprop
from keras import optimizers
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import theano


sys.path.append("..\..")
from tengwar.data.data import make_theano_shared
from tengwar.nnet.callbacks import EarlyStoppingRestoringWeights
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger

from random import choice
from string import ascii_uppercase

import os

def is_power2(num):

    'states if a number is a power of two'

    return num != 0 and ((num & (num - 1)) == 0)

def get_length(modality_list):
    length = 0
    for item in modality_list:
        if type(item) == list:
            length += get_length(item)
        else:
            length += 1
    return length


# Simple feed-forward network, used to get network outputs and evaluations
# Input: 0-mean, 1-std, 2-dimension array (n_samples, n_dimensions)
# Output: 0/1 binary label (n_samples)
# Layers: 2 hidden layers and 1 prediction layer
# Activation: sigmoid for all layers
# Objective: binary cross entropy for prediction
class FeedForwardNetwork(object):
    def __init__(self, hidden_dim=None, final_activation='sigmoid', loss='binary_crossentropy',
                 batch_size=10, nb_epoch=50, learning_rate=0.001, validation_split=0.25, patience=20, early_stopping='True_BestWeight', ffn_depth=2,
                 batch_normalization='False'):
        self.hidden_dim = hidden_dim
        self.final_activation = final_activation
        self.loss = loss
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.patience = patience
        self.EarlyStopping = early_stopping
        self.type_MMDL = 1
        self.random_str = ''.join(choice(ascii_uppercase) for i in range(12))
        self.ffn_depth = ffn_depth
        self.batch_normalization = batch_normalization

    # get the i-th (hidden) layer output as the features of input X
    def transform_features(self, X, layers=None):
        pass
 
    def transform(self, X):
        return None

    # sklearn.$anymodel$.fit
    def fit(self, X, y):
        n_features = X.shape[1]
        model = Sequential()

        hidden_dim = self.hidden_dim
        final_activation = self.final_activation
        loss = self.loss
        if hidden_dim is None:
            hidden_dim = 8 * n_features

        model.add(Dense(hidden_dim, input_shape=(n_features, ), activation='sigmoid'))
        model.add(Dropout(0.1))
        for t in range(self.ffn_depth - 1):
            print('layer%d'%t)
            model.add(Dense(hidden_dim, activation='sigmoid'))
            model.add(Dropout(0.2))
        model.add(Dense(1, activation=None))
        if self.batch_normalization == 'True':
            model.add(BatchNormalization())
        model.add(Activation(activation=final_activation))
        self.model = model
        print 'number of parameters: ' + str(self.model.count_params())
        print 'before compiling'
        self.model.compile(loss=loss, optimizer=RMSprop(lr=self.learning_rate), metrics=['accuracy'])
        print 'before fitting'

        if self.EarlyStopping == 'True_BestWeight':
            self.model.fit(
                x=X, y=y,
                batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                # show_accuracy=True,
                verbose=2, shuffle=False,
                validation_split=self.validation_split,
                # callbacks = [EarlyStopping(patience=10)],
                callbacks=[EarlyStoppingRestoringWeights(patience=self.patience, weight_path='ES_best_' + str(
                    self.type_MMDL) + '_' + self.random_str + ' .h5')],
            )

        elif self.EarlyStopping == 'True':
            self.model.fit(
                x=X, y=y,
                batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                # show_accuracy=True,
                verbose=2,
                validation_split=self.validation_split,
                callbacks=[EarlyStopping(patience=self.patience)],
            )

        elif self.EarlyStopping == 'False':
            self.model.fit(
                x=X, y=y,
                batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                # show_accuracy=True,
                verbose=2,
                validation_split=self.validation_split,
            )

        print 'after fitting'
        return self
            
    # get output layer values of input X
    def decision_function(self, X):
        return self.model.predict(X)
        
    # get prediction of input X (compare output value and threshold)
    def predict(self, X):
        if self.loss == 'mean_squared_error':
            return self.model.predict(X)
        else:
            return self.model.predict_classes(X)

        
# Simple LSTM network with a overall prediction layer
# Input: 0-mean, 1-std, 3-dimension array (n_samples, n_timesteps, n_dimensions)
# Output: 0/1 binary label    (n_samples)
# Layers: 2 hidden layers and 1 prediction layer
# Activation: sigmoid for all layers
# Objective: binary cross entropy for prediction        
class SimpleLSTMNetwork(object):
    def __init__(self):
        pass

    # get the i-th (hidden) layer output as the features of input X
    def transform_features(self, X, layers=None):
        pass
 
    def transform(self, X):
        return self.get_activations(X)

    # sklearn.$anymodel$.fit
    def fit(self, X, y):
        t_length = X.shape[1]
        n_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(output_dim=n_features, activation='sigmoid', 
                       return_sequences=True, 
                       input_dim = n_features, input_length = t_length))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model
        print 'before compiling'
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        print 'before fitting'
        self.model.fit(
                X=X, y=y, 
                batch_size=10, nb_epoch=50, 
                show_accuracy=True, verbose=1,
                )
        print 'after fitting'
        self.get_activations = theano.function(
                [self.model.layers[0].get_input(train=False)], 
                self.model.layers[-3].get_output(train=False),
                allow_input_downcast=True)
        return self
            
    # get output layer values of input X
    def decision_function(self, X):
        return self.model.predict(X)
        
    # get prediction of input X (compare output value and threshold)
    def predict(self, X):
        return self.model.predict(X) >= 0.5
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'model':
                setattr(result, k, deepcopy(v, memo))
            else:
                json_string = v.to_json()
                weight = {}
                weight['nb_layers'] =  len(v.layers)        
                for k, l in enumerate(v.layers):
                    weight['layer_{}'.format(k)] = {}
                    weights = l.get_weights()
                    weight['layer_{}'.format(k)]['nb_params'] = len(weights)
                    for n, param in enumerate(weights):
                       param_name = 'param_{}'.format(n)
                       weight['layer_{}'.format(k)][param_name] = param
                
                new_model = model_from_json(json_string)
                for k in range(weight['nb_layers']):
                    # This method does not make use of Sequential.set_weights()
                    # for backwards compatibility.
                    g = weight['layer_{}'.format(k)]
                    weights = [g['param_{}'.format(p)] for p in range(g['nb_params'])]
                    new_model.layers[k].set_weights(weights)
                setattr(result, 'model', new_model)
        return result


# functions for StackedAutoencoderNetwork
def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def threshold(Z):
    return (Z>=0.5).astype(int)

def softmax(Z):
    eZ = np.exp(Z)
    eZ = eZ / np.sum(eZ,axis=1)[:,None]
    return eZ

def argmax(Z):
    return Z.argmax(axis=1)
 
 
       
class HierarchicalMultimodal(object):
    '''
    Input:    
    static: True or False, indicate whether there are static features in the dataset
    size_Xs: dimension of static features 
    temporal: True of False, indicate whether there are temporal features in the dataset
    number_modality: int 
    size_of_modality: a list of number, indicate the size of each modality, len(size_of_modality) = number_modality
    td_of_modality: a list of number, indicate the time series length of each modality, len(td_of_modality) = number_modality
    type_MMDL: 1 or 2 or 3. 1: MMDL_St; 2: MMDL_Sm;  3: HMMDL
    HMMDL_struture: a list of list. Indicate the HMMDL structure

    fit_parameters:  [output_dim, static_depth, merge_depth]
    optimizer: 'RMSprop' or 'adam'
    EarlyStopping: 'False','True', 'True_BestWeight'
    y_tasks: 1 for single task, n for multitask

    ''' 
    def __init__(self, static = False, remove_sapsii = False, size_Xs= 0, temporal = True, number_modality = 1, size_of_modality = [], td_of_modality = [],
                       type_MMDL = 1, HMMDL_struture = [] ,fit_parameters = [], optimizer = 'RMSprop', activation = 'sigmoid', final_activation = 'sigmoid',loss = 'binary_crossentropy',
                       EarlyStopping = 'True_BestWeight', EarlyStopping_patience = 10, dropout = 0.1, batch_normalization = 'False', batch_size = 20, validation_split = 0.25,
                       nb_epoch = 100, learning_rate=0.001, return_sequences = True, y_tasks = 1, logdir = './log'):

        if not static and not temporal:
            print 'Empty dataset.'
            return self


        if type_MMDL == 3:
            assert get_length(HMMDL_struture) == number_modality

        self.static = static
        self.remove_sapsii = remove_sapsii
        self.optimizer = optimizer
        self.EarlyStopping = EarlyStopping
        self.patience = EarlyStopping_patience
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss = loss
        self.validation_split = validation_split
        self.type_MMDL = type_MMDL
        self.random_str = ''.join(choice(ascii_uppercase) for i in range(12))
        self.size_Xs = size_Xs
        self.size_of_modality = size_of_modality
        self.number_modality = number_modality
        self.td_of_modality = td_of_modality
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.return_sequences = return_sequences
        self.y_tasks = y_tasks
        self.activation = activation
        self.final_activation = final_activation
        self.HMMDL_struture = HMMDL_struture

        assert len(size_of_modality) == number_modality
        assert len(td_of_modality) == number_modality


        self.output_dim = fit_parameters[0]
        self.static_depth = fit_parameters[1]
        self.merge_depth = fit_parameters[2]

        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        with open(os.path.join(self.logdir, 'log.csv'), 'w') as f:
            pass

    def __remove_sapsii(self, X):
        '''
        remove features used in sapsii, starting from 99(GCSverbal) and ending at 113(bilirubin_level)
        :param X: X_t, time series
        :return: X_t after removing
        '''
        return X[:,:,list(range(99))+list(range(114, 136))]


    def create_HMMDL_model(self,modality_list):

        size_list =[]
        merge_list = []
        for item in modality_list:
            if type(item) == list:
                len_combine,GRUmodel = self.create_HMMDL_model(item)
                size_list.append(len_combine)
                merge_list.append(GRUmodel)
                
            else:
                t_length = self.td_of_modality[item]
                n_features = self.size_of_modality[item]

                GRUmodel = Sequential()
                GRUmodel.add(GRU(output_dim=n_features*self.output_dim, activation=self.activation, 
                                return_sequences=self.return_sequences, 
                                input_dim = n_features, input_length = t_length))
                if self.return_sequences:
                    GRUmodel.add(Flatten())
                GRUmodel.add(Dropout(self.dropout))

                size_list.append(n_features)
                merge_list.append(GRUmodel)

        len_combine = np.sum(size_list)
        GRUmodel = Sequential()
        GRUmodel.add(Merge(merge_list, mode='concat', concat_axis=-1))
        GRUmodel.add((Dense(len_combine*self.output_dim, activation=self.activation)))   

        return len_combine,GRUmodel


    def crete_model(self):
        if self.static:
            # FFN model
            FFNmodel = Sequential()
            FFNmodel.add(Dense(self.size_Xs*self.output_dim, input_shape=(self.size_Xs, ), activation= self.activation))
            FFNmodel.add(Dropout(self.dropout))
            for i in range(self.static_depth):
                FFNmodel.add(Dense(self.size_Xs*self.output_dim, activation= self.activation))
                FFNmodel.add(Dropout(self.dropout))

        if self.type_MMDL == 1:
            t_length = self.td_of_modality[0]
            n_features = np.sum(self.size_of_modality)

            if self.remove_sapsii:
                n_features -= (114-99)

            GRUmodel = Sequential()
            GRUmodel.add(GRU(output_dim=n_features*self.output_dim, activation=self.activation, 
                        return_sequences=self.return_sequences, 
                        input_dim = n_features, input_length = t_length))
            if self.return_sequences:
                GRUmodel.add(Flatten())
            GRUmodel.add(Dropout(self.dropout))

            if self.static:
                len_combine = self.size_Xs + n_features
                self.model = Sequential()
                self.model.add(Merge([FFNmodel, GRUmodel], mode='concat', concat_axis=-1))
                self.model.add((Dense(len_combine*self.output_dim, activation=self.activation)))
                self.model.add(Dropout(self.dropout))

                for i in range(self.merge_depth):
                    self.model.add(Dense(len_combine*self.output_dim/np.power(2,i+1), activation=self.activation))
                    self.model.add(Dropout(self.dropout))

                # A linear layer
                self.model.add(Dense(self.y_tasks, activation=None))
                if self.batch_normalization == 'True':
                    self.model.add(BatchNormalization())
                # self.model.add(Dense(self.y_tasks, activation='sigmoid'))
                self.model.add(Activation(activation=self.final_activation))

            else:
                len_combine = n_features
                self.model = Sequential()
                self.model.add(GRUmodel)
                self.model.add(Dense(self.y_tasks, activation=None))
                if self.batch_normalization == 'True':
                    self.model.add(BatchNormalization())
                self.model.add(Activation(activation=self.final_activation))
                print 'No static features!'
                
        elif self.type_MMDL == 2:

            assert self.number_modality >= 1
            GRUmodel_list = []

            if self.static:
                GRUmodel_list.append(FFNmodel)
                
            for i in range(self.number_modality):
                t_length = self.td_of_modality[i]
                n_features = self.size_of_modality[i]


                GRUmodel = Sequential()
                GRUmodel.add(GRU(output_dim=n_features*self.output_dim, activation=self.activation, 
                            return_sequences=self.return_sequences, 
                            input_dim = n_features, input_length = t_length))
                if self.return_sequences:
                    GRUmodel.add(Flatten())
                GRUmodel.add(Dropout(self.dropout))
                GRUmodel_list.append(GRUmodel)

            len_combine = self.size_Xs + np.sum(self.size_of_modality)
            self.model = Sequential()
            self.model.add(Merge(GRUmodel_list, mode='concat', concat_axis=-1))
            self.model.add((Dense(len_combine*self.output_dim, activation=self.activation)))
            self.model.add(Dropout(self.dropout))

            for i in range(self.merge_depth):
                self.model.add(Dense(len_combine*self.output_dim/np.power(2,i+1), activation=self.activation))
                self.model.add(Dropout(self.dropout))

            # self.model.add(Dense(self.y_tasks, activation='sigmoid'))
            # A linear layer
            self.model.add(Dense(self.y_tasks, activation=None))
            if self.batch_normalization == 'True':
                self.model.add(BatchNormalization())
            self.model.add(Activation(activation='sigmoid'))

        elif self.type_MMDL == 3:
            len_combine,GRUmodel = self.create_HMMDL_model(self.HMMDL_struture)

            if self.static:
                len_combine = self.size_Xs + len_combine
                self.model = Sequential()
                self.model.add(Merge([FFNmodel, GRUmodel], mode='concat', concat_axis=-1))
                self.model.add((Dense(len_combine*self.output_dim, activation=self.activation)))
                self.model.add(Dropout(self.dropout))
            else:
                self.model = GRUmodel
                self.model.add(Dropout(self.dropout))

            for i in range(self.merge_depth):
                self.model.add(Dense(len_combine*self.output_dim/np.power(2,i+1), activation=self.activation))
                self.model.add(Dropout(self.dropout))

            # A linear layer
            self.model.add(Dense(self.y_tasks, activation=None))
            if self.batch_normalization == 'True':
                self.model.add(BatchNormalization())
            self.model.add(Activation(activation='sigmoid'))

        

    # get the i-th (hidden) layer output as the features of input X
    def transform_features(self, X, layers=None):
        pass
 
    def transform(self, X):
        # return self.get_activations(*X)
        return None

    '''
        Input:    
        X: format will be [X_static, X_modality1, X_modality2, ... ]  if static = True
           format will be [ X_modality1, X_modality2, ...  ]  if static = False
    '''

    def fit(self, X, y):
        print(y[:10])
        self.crete_model()    

        print 'number of parameters: ' + str(self.model.count_params())

        print 'before compiling'
        self.model.compile(loss=self.loss, optimizer=getattr(optimizers, self.optimizer)(lr=self.learning_rate), metrics=['accuracy'])
        
        print 'before fitting'

        csvLogger = CSVLogger(os.path.join(self.logdir, 'log.csv'), append=True)

        if self.remove_sapsii:
            X = [X[0], self.__remove_sapsii(X[1])]
            print(X[1].shape)

        if self.static == False:
            X = X[1]
            print(X.shape)

        if  self.EarlyStopping == 'True_BestWeight':
            self.model.fit(
                    x=X, y=y,
                    batch_size=  self.batch_size, nb_epoch=self.nb_epoch, 
                    verbose=2,shuffle = False,
                    validation_split = self.validation_split,
                    callbacks = [EarlyStoppingRestoringWeights(patience= self.patience, weight_path =  'ES_best_' + str(self.type_MMDL) + '_' +    self.random_str  + ' .h5'), csvLogger],
                    )

        elif self.EarlyStopping == 'True':
            self.model.fit(
                    x=X, y=y,
                    batch_size= self.batch_size, nb_epoch=self.nb_epoch, 
                    # show_accuracy=True,
                    verbose=2,
                    validation_split = self.validation_split,
                    callbacks = [EarlyStopping(patience=self.patience), csvLogger],
                    )

        elif self.EarlyStopping == 'False':
            self.model.fit(
                    x=X, y=y,
                    batch_size= self.batch_size, nb_epoch=self.nb_epoch, 
                    # show_accuracy=True,
                    verbose=2,
                    validation_split = self.validation_split,
                    #callbacks = [EarlyStopping(patience=self.patience)],
                    callbacks=[csvLogger]
                    )

        print 'after fitting'

        return self
        
    # get output layer values of input X
    def decision_function(self, X):
        if self.remove_sapsii:
            X = [X[0], self.__remove_sapsii(X[1])]
        if self.static == False:
            X = X[1]
        return self.model.predict(X)
        
    # get prediction of input X (compare output value and threshold)
    def predict(self, X):
        if self.remove_sapsii:
            X = [X[0], self.__remove_sapsii(X[1])]
        if self.static == False:
            X = X[1]
        if self.loss == 'mean_squared_error':
            return self.model.predict(X)
        else:
            return self.model.predict(X) >= 0.5

