import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# normalize data relative to max value for that parameter found in training set
def norm_data(data, max_data):
    for dat in data:
        for i in range(dat.shape[1]):
            dat[:, i] = dat[:, i] / max_data[i]
    return data

# separate model data into that which will be used for positive and negative sequences
def prepare_data_within(dat_list, window_size, time_delay, first_diff):

    window_length = (window_size * 5) + 1  # convert hours window size to minutes (5 data points per hour)
    negative= []
    positive = []
    for j ,dat in tqdm(enumerate(dat_list), total=len(dat_list), position=0, desc="data", leave=False, colour='green', ncols=80):
        # discard data for which there is not enough data to generate the positive (within-24-hr) sequences
        if dat.shape[0] < ((window_size*5)+1):
            #print('There is no enough data, therefore I will skip this data\n')
            continue
        else:
            # clean data using linear interpolation for missing steps, backward/forward fill for gaps @ end or start
            dat = dat.drop_duplicates('time')
            dat = dat.set_index('time')
            dat = dat.resample('12T').interpolate('linear')
            dat = dat.fillna(method='bfill')
            dat = dat.fillna(method='ffill')
            dat = dat.reset_index()

            # all data points before (prediction window) + (data duration) marked as negative:
            # any (data duration)-length windows starting before this time will not
            # end within (prediction window) hr of flare -> saved as negative
            neg = dat.loc[(dat.time <= (dat.time.iloc[-1] - timedelta(hours=time_delay) - timedelta(hours=window_size)))]
            pos = dat.loc[(dat.time > (dat.time.iloc[-1] - timedelta(hours=time_delay) - timedelta(hours=window_size)))]

            # remove SHARPs parameters not included in model input, such as NOAA num
            neg = neg.iloc[:, 1:-5].reset_index(drop=True)
            pos = pos.iloc[:, 1:-5].reset_index(drop=True)

            # convert to first-degree differences
            if first_diff == True:
                neg = neg.diff().dropna()
                pos = pos.diff().dropna()

            # save sequences to dataframes, separated by pos/neg label

            if neg.shape[0] < window_length:
                neg=pd.DataFrame()
            if pos.shape[0] < window_length:
                pos=pd.DataFrame()

            negative.append(neg)
            positive.append(pos)

            negative = [df for df in negative if not df.empty]
            positive = [df for df in positive if not df.empty]

    return positive, negative

# function to calculate all 6 metrics given predicted and true labels
def calc_scores(y_pred, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    bacc = (tp/(tp+fn) + tn/(tn+fp)) / 2.0
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    hss = (2 * (tp * tn - fp * fn)) / (((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)))

    return acc, bacc, precision, recall, tss, hss


# function to calculate all metrics at each threshold
def calculate_all_metrics(pred_y, y_test, thresholds):

    acc_arr = []
    bacc_arr = []
    precision_arr = []
    recall_arr = []
    tss_arr = []
    hss_arr = []

    # iterate over all thresholds, calculating metrics for each one
    for threshold in thresholds:
        # convert continuous probabilities to 0,1 label by comparing to threshold
        pred_p = pred_y.flatten()
        pred_y_conv = np.where(pred_p > threshold, 1., 0.)

        metrics = calc_scores(pred_y_conv, y_test)

        acc_arr.append(metrics[0])
        bacc_arr.append(metrics[1])
        precision_arr.append(metrics[2])
        recall_arr.append(metrics[3])
        tss_arr.append(metrics[4])
        hss_arr.append(metrics[5])

    print('max acc of', np.max(acc_arr), 'with threshold', thresholds[np.argmax(acc_arr)], '(index', np.argmax(acc_arr), ')')
    print('max bacc of', np.max(bacc_arr), 'with threshold', thresholds[np.argmax(bacc_arr)], '(index', np.argmax(bacc_arr), ')')
    print('max precision of', np.max(precision_arr), 'with threshold', thresholds[np.argmax(precision_arr)], '(index', np.argmax(precision_arr), ')')
    print('max recall of', np.max(recall_arr), 'with threshold', thresholds[np.argmax(recall_arr)], '(index', np.argmax(recall_arr), ')')
    print('max tss of', np.max(tss_arr), 'with threshold', thresholds[np.argmax(tss_arr)], '(index', np.argmax(tss_arr), ')')
    print('max hss of', np.max(hss_arr), 'with threshold', thresholds[np.argmax(hss_arr)], '(index', np.argmax(hss_arr), ')')

    return acc_arr, bacc_arr, precision_arr, recall_arr, tss_arr, hss_arr