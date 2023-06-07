
import tensorflow as tf
import keras as k
from keras import backend as K

#General
import numpy as np
import os
import pandas as pd
import math

#Data
import wfdb
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter 
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True
K.set_session(tf.compat.v1.Session(config = config))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# #### Variable names and description

# __bal_trainX_1beat :__ single beat segmentation train set.
# 
# __bal_trainX_3beat :__ triple beat segmentation train set.
# 
# __bal_trainY_lbl :__ class of each train set instance, string ('F', 'N', 'S' or 'V').
# 
# __bal_trainY_int :__ class of each train set instance, integer (0, 1, 2 or 3).
# 
# __bal_trainY_oh :__ class of each train set instance, one-hot-encoding ((1,0,0,0), (0,1,0,0), (0,0,1,0) or (0,0,0,1)).
# 
# __testX_1beat :__ single beat segmentation test set.
# 
# __testX_3beat :__ triple beat segmentation test set.

# #### Import data from MIT-BIH repository


#Database import
#wfdb.dl_database('mitdb', 'mitdb')

def butter_bandpass(lowcut, highcut, fs, order = 5):
	return butter(order, [lowcut, highcut], fs = fs, btype = "band")

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
	b, a = butter_bandpass(lowcut, highcut, fs, order = order)
	y = lfilter(b, a, data)
	return y 


complete_data = pd.DataFrame()

half_qrs = 210 #Single beat segmentation -> initial beat length: 420pt
pts = 320 #Triple beat segmentation -> beat length: 320pt


#Beat selection from Physionet beats annotation guidlines
good_beats = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r','F', 'e', 'j', 'n', 'E']

for filename in os.listdir('mitdb'):
    if filename.endswith(".dat"):
        ann = wfdb.rdann('mitdb/' + filename.strip('.dat'), 'atr')
        record = wfdb.rdsamp('mitdb/' + filename.strip('.dat'))
        record_num = filename.strip('.dat')
        
        #Signals
        data = record[0]
        #data = butter_bandpass_filter(data, 0.05, 60, 360)
        signals1, signals2, classes, signals3 = [], [], [], []
        
        #Beat extraction
        for it, beat in enumerate(ann.symbol):
            if it>0:
                if beat in good_beats:
                    
                    #Single beat extraction
                    sta1 = ann.sample[it] - half_qrs
                    end1 = ann.sample[it] + half_qrs
                    qrs1 = data[sta1 : end1, :]
                    
                    #Triple beat extraction
                    sta2 = math.floor((ann.sample[it] + ann.sample[it-1])/2)
                    end2 = sta2 + pts
                    qrs2 = data[sta2 : end2, :]
                    
                    #Avoid problems at the edges
                    if len(qrs1) != 2 * half_qrs: continue
                    if end2 > len(data[:,0]): continue

                    curr_beat1 = qrs1.reshape(half_qrs*2,2)
                    curr_beat2 = qrs2.reshape(pts,2)
                    signals1.append(curr_beat1)
                    signals2.append(curr_beat2)
                    classes.append(beat)

        #Triple beat arrangement
        for i in range(1,len(signals2)-1):
            temp = np.hstack((signals2[i-1],signals2[i],signals2[i+1]))
            signals3.append(temp)
            temp=0
            
        #Single and triple beat consistency
        signals1.pop(0)
        signals1.pop(-1)
        classes.pop(0)
        classes.pop(-1)
        
        #Obtained data 
        frame = pd.DataFrame({'beat' : signals1,
                              '3beat' : signals3,
                              'label' : classes,
                              'record' : record_num})

        complete_data = complete_data.append(frame)


# #### Train/test set definition

#TRAIN SET

#Train record selection
mask = np.isin(complete_data['record'],['101', '106', '108', '109', '112', '115',
                                                          '116', '118', '119', '122', '201', '203', 
                                                          '205', '207', '208', '209', '215', '220', '223', '230'])
train_df = complete_data[mask]

#eliminate S and e classes, which are not included in the test set
train_df = train_df[np.isin(train_df['label'],['S','e'], invert = True)]

#Single beat
train = train_df['beat'].values
trainX_1beat_420 = np.vstack(train).reshape(train.shape[0],half_qrs*2,2)

#Triple beat
train = train_df['3beat'].values
trainX_3beat = np.vstack(train).reshape(train.shape[0],320,6)

#MIT annotations
trainY_lbl_MIT = train_df['label'].values


# STANDARDIZATION
#mean_vals = np.mean(trainX_1beat_420, axis=1)[:, np.newaxis]  # shape (n_samples, 1)
#trainX_1beat_420 = trainX_1beat_420 - mean_vals
#mean_vals = np.mean(trainX_3beat, axis=1)[:, np.newaxis]  # shape (n_samples, 1)
#trainX_3beat = trainX_3beat - mean_vals


#TEST SET

#Test record selection
mask =  np.isin(complete_data['record'],['100', '105', '111', '113', '121',
                                                          '200', '202', '210', '212', '213', '214', 
                                                          '219', '221', '222', '228', '231', '232', '233', '234'])
test_df = complete_data[mask]

#Single beat
test = test_df['beat'].values
testX_1beat_420 = np.vstack(test).reshape(test.shape[0],half_qrs*2,2)

#Triple beat
test = test_df['3beat'].values
testX_3beat = np.vstack(test).reshape(test.shape[0],320,6)

#MIT annotations
testY_lbl_MIT = test_df['label'].values

# STANDARDIZATION
#mean_vals = np.mean(testX_1beat_420, axis=1)[:, np.newaxis]  # shape (n_samples, 1)
#testX_1beat_420 = testX_1beat_420 - mean_vals
#mean_vals = np.mean(testX_3beat, axis=1)[:, np.newaxis]  # shape (n_samples, 1)
#testX_3beat = testX_3beat - mean_vals

# #### Change MIT annotations to AAMI

#Find AAMI classes
def which_class(x):
    if np.isin(x,N): return 'N'
    elif np.isin(x,S): return 'S'
    elif np.isin(x,V): return 'V'
    elif np.isin(x,F): return 'F'
    else: return 'none'
    
#definition of AAMI classes
N = ['.','N','L','R','e','j','n'];
S = ['A','a','J','S'];
V = ['V','E','r'];
F = ['F'];

#AAMI labelling
trainY_lbl = np.array([which_class(x) for x in trainY_lbl_MIT])
testY_lbl = np.array([which_class(x) for x in testY_lbl_MIT])

#Integer label reference
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(trainY_lbl)
trainY_int = label_encoder.transform(trainY_lbl)
testY_int = label_encoder.transform(testY_lbl)

#One-hot-encoding label reference
trainY_oh = to_categorical(trainY_int)
testY_oh = to_categorical(testY_int)


# #### Balancing

def balancing_dataset(X, Y_lbl, Y_int, Y_oh):
    
    N_indeces = np.argwhere('N' == Y_lbl).reshape(1,-1)[0,:]
    balanced_X = X[N_indeces]
    balanced_Y_lbl = Y_lbl[N_indeces]
    balanced_Y_int = Y_int[N_indeces]
    balanced_Y_oh = Y_oh[N_indeces]
    
    N = len(N_indeces)
    
    for lbl in np.unique(Y_lbl):
        i = np.argwhere(lbl == Y_lbl).reshape(1,-1)[0,:]
        
        if lbl=='N':
            x = 1
        else:
            x = math.floor(2*N/len(i))

        a = np.vstack([X[i]]*x)
        balanced_X = np.vstack((balanced_X, a))
        a = np.hstack([Y_lbl[i]]*x)
        balanced_Y_lbl = np.hstack((balanced_Y_lbl, a))
        a = np.hstack([Y_int[i]]*x)
        balanced_Y_int = np.hstack((balanced_Y_int, a))
        a = np.vstack([Y_oh[i]]*x)
        balanced_Y_oh = np.vstack((balanced_Y_oh, a))
        
    return balanced_X, balanced_Y_lbl, balanced_Y_int, balanced_Y_oh


#SINGLE-BEAT SEGMENTATION -> BALANCING AND PEAK SHIFTING

#Train set balancing
bal_trainX_1beat_420, bal_trainY_lbl, bal_trainY_int, bal_trainY_oh = balancing_dataset(trainX_1beat_420, trainY_lbl, trainY_int, trainY_oh)

#Train set peak-shifting
bal_trainX_1beat = np.zeros((bal_trainX_1beat_420.shape[0], 320, 2))

for i in range(bal_trainX_1beat.shape[0]):
    sta = np.random.choice(range(0,101))
    end = sta + 320
    bal_trainX_1beat[i,:,:] = bal_trainX_1beat_420[i, sta:end, :]

#Test set peak-shifting
testX_1beat = np.zeros((testX_1beat_420.shape[0], 320, 2))

for i in range(testX_1beat.shape[0]):
    sta = np.random.choice(range(0,100))
    end = sta + 320
    testX_1beat[i,:,:] = testX_1beat_420[i, sta:end, :]

#TRIPLE-BEAT SEGMENTATION -> BALANCING

#Train set balancing
bal_trainX_3beat, bal_trainY_lbl, bal_trainY_int, bal_trainY_oh = balancing_dataset(trainX_3beat, trainY_lbl, trainY_int, trainY_oh)

