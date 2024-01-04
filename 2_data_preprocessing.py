import numpy as np
import joblib
from helpers import prepare_data_within
import os

input_path = 'your/own/path'

print('Loading data from disk, generated in load_data')
A_flr = joblib.load(input_path+'A_flr.pkl')
B_flr = joblib.load(input_path+'B_flr.pkl')
C_flr = joblib.load(input_path+'C_flr.pkl')
M_flr = joblib.load(input_path+'M_flr.pkl')
X_flr = joblib.load(input_path+'X_flr.pkl')

# 24-hr sequences ending within 24 hr of flare are positive, so the time window for positive sequence starts is 48 hr
time_delay = 24
case_cutoff = ['C', 'M']
window = 48
make_positive_negative = True

if make_positive_negative == True:

    print('Preparing data for time delay of {}h\n'.format(time_delay))
    print('Separating A class into positive and negative classes\n')
    # separates data into positive/negative using function in helpers
    A_pos, A_neg = prepare_data_within(A_flr, window_size=window, time_delay=time_delay,
                                       first_diff=True)

    print('Separating B class into positive and negative classes\n')
    B_pos, B_neg = prepare_data_within(B_flr, window_size=window, time_delay=time_delay,
                                       first_diff=True)


    print('Separating C class into positive and negative classes\n')
    C_pos, C_neg = prepare_data_within(C_flr, window_size=window, time_delay=time_delay,
                                       first_diff=True)


    print('Separating M class into positive and negative classes\n')
    M_pos, M_neg = prepare_data_within(M_flr, window_size=window, time_delay=time_delay,
                                       first_diff=True)


    print('Separating X class into positive and negative classes\n')
    X_pos, X_neg = prepare_data_within(X_flr, window_size=window, time_delay=time_delay,
                                       first_diff=True)


window_length = (window*5 + 1) + 1

# iterate over different flare class prediction cases (>=C vs >=M)
for case in case_cutoff:

    output_path = 'your/own/path'+ str(window)+'/'+ case + '_' + str(time_delay) + '/'

    print('Saving data for {}-class flare cutoff with {}h delay into'.format(case, time_delay))
    print(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # if predicting >=C-class flares, only sequences in which there is a >=C-class flare within 24 hr are pos
    if 'C' in case:
        neg = A_pos + A_neg + B_pos + B_neg + C_neg + M_neg + X_neg
        pos = C_pos + M_pos + X_pos

    # if predicting >=M-class flares, only sequences in which there is a >=M-class flare within 24 hr are pos
    elif 'M' in case:

        neg = A_pos + A_neg + B_pos + B_neg + C_pos + C_neg + M_neg + X_neg
        pos = M_pos + X_pos

    negative = []
    positive = []
    for dat in pos:
        num = dat.shape[0]-window_length+1  # this is the number of possible start locations for an input sequence
        if num == 0 or num < 0:
            continue
        else:
            # generate random (window_length)-length sequences starting at random times in the large time series
            # this enables within-(24)-hr prediction, rather than using a fixed time interval
            samples = [dat.iloc[x:x+window_length].values for x in np.unique(np.random.randint(num, size=50))]
            positive.append(samples)

    # same as above function, but for negative examples
    for dat in neg:
        num = dat.shape[0] - window_length + 1
        if num == 0 or num < 0:
            continue
        else:
            samples = [dat.iloc[x:x+window_length].values for x in np.unique(np.random.randint(num, size=50))]
            negative.append(samples)

    positive = [item for sublist in positive for item in sublist]
    negative = [item for sublist in negative for item in sublist]

    print('I have {} positive and {} negative cases in total'.format(len(positive), len(negative)))

    # create 0-1 class labels for data sequences generated
    neg_label = np.zeros(len(negative))
    pos_label = np.ones(len(positive))
    x = np.vstack((negative, positive))
    y = np.hstack((neg_label, pos_label))

    joblib.dump(x, output_path + 'x_'+str(window)+'.pkl')
    joblib.dump(y, output_path + 'y_'+str(window)+'.pkl')


