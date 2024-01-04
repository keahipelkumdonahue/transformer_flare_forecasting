import pandas as pd
import numpy as np
from sklearn.utils import resample
import joblib
from sklearn.model_selection import train_test_split
from helpers import norm_data

list_of_type = ['M_24']  # change this parameter for which model case (class/data duration) is being train-test-split
window =48  # use this to change the length of data duration being loaded for train_test_split
sampling = 'Down' # adjust to switch from down- to up-sampling data

for type in list_of_type:

    print('=============================================================================++++++++++++++++++++++======\n')
    print('Now I am creating and saving data to train a Transformer Network for data cut-off at {}-type flares for {}h time delay\n'.format(type[0:1], type[2:]))
    print('===========================================================================+++++++++++++++++++++++=======\n')

    input_path = 'your/own/path'+str(window)+'/'+type+'/'

    # load data generated in data_preprocessing
    xdata = joblib.load(input_path + 'x_'+str(window)+'.pkl')
    ydata = joblib.load(input_path + 'y_'+str(window)+'.pkl')

    # shuffle data
    shuffle_index = np.arange(xdata.shape[0])  # shuffle the data indices
    np.random.shuffle(shuffle_index)
    ydata_shuffled = ydata[shuffle_index]
    xdata_shuffled = xdata[shuffle_index, :]

    ''' separating training, testing, and validation data '''
    x_train_val, x_test, y_train_val, y_test = train_test_split(xdata_shuffled, ydata_shuffled, test_size=0.20)

    del xdata_shuffled, ydata_shuffled, xdata, ydata, shuffle_index

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.20)

    # normalize data using function in helpers.py
    max_train = pd.DataFrame(np.concatenate(x_train)).max().values
    x_train = norm_data(x_train, max_train)
    x_val = norm_data(x_val, max_train)
    x_test = norm_data(x_test, max_train)

    print('\n....................correcting class imbalance....................')
    print('Number of class 0 before:', len(x_train[y_train == 0]))
    print('Number of class 1 before:', len(x_train[y_train == 1]))

    if len(x_train[y_train == 0]) > len(x_train[y_train == 1]):

        if sampling == 'Down':

            print('Negative class needs downsampling!\n')

            # remove negative examples until data is balanced
            x_train_downsampled, y_train_downsampled = resample(x_train[y_train == 0], y_train[y_train == 0],
                                                            replace=True, n_samples=x_train[y_train == 1].shape[0],
                                                            random_state=1)

            x_train = np.vstack((x_train[y_train == 1], x_train_downsampled))
            y_train = np.hstack((y_train[y_train == 1], y_train_downsampled))

            del x_train_downsampled, y_train_downsampled

            print('Number of class 0 after: ', len(x_train[y_train == 0]))
            print('Number of class 1 after: ', len(x_train[y_train == 1]))

        elif sampling == 'Up':

            print('Positive class needs upsampling!\n')

            # add positive examples until data is balanced
            x_train_upsampled, y_train_upsampled = resample(x_train[y_train == 1], y_train[y_train == 1],
                                                            replace=True,
                                                            n_samples=x_train[y_train == 0].shape[0],
                                                            random_state=1)

            x_train = np.vstack((x_train[y_train == 0], x_train_upsampled))
            y_train = np.hstack((y_train[y_train == 0], y_train_upsampled))

            del x_train_upsampled, y_train_upsampled

            print('Number of class 0 after: ', len(x_train[y_train == 0]))
            print('Number of class 1 after: ', len(x_train[y_train == 1]))

    # save separated training, testing, and validation data
    joblib.dump(x_train, input_path + 'x_train.pkl')
    joblib.dump(y_train, input_path + 'y_train.pkl')
    joblib.dump(x_val, input_path + 'x_val.pkl')
    joblib.dump(y_val, input_path + 'y_val.pkl')
    joblib.dump(x_test, input_path + 'x_test.pkl')
    joblib.dump(y_test, input_path + 'y_test.pkl')
    joblib.dump(max_train, input_path + 'max_train.pkl')
