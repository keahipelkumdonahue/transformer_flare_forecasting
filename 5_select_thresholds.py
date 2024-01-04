import joblib
import numpy as np
from helpers import calculate_all_metrics

print('\n...........Loading data...........')

# adjust these files depending on which model is being tested
y_pred = joblib.load('your/path/for/predicted_probabilities_file')
y_test = joblib.load('your/path/for/true_labels_file')

print('\n...........Calculating all metrics for each threshold...........')

thresholds = np.linspace(0.05, 0.95, 91)  # these are the thresholds to be tested
acc, bacc, pre, rec, tss, hss = calculate_all_metrics(y_pred, y_test, thresholds)  # calculate using helpers function

print('\n...........Calculating scores at threshold yielding max TSS...........')

max_tss = max(tss)
max_tss_index = tss.index(max_tss)
max_threshold = thresholds[max_tss_index]

print('Max TSS occurs with threshold', max_threshold)

acc_max_tss = acc[max_tss_index]
bacc_max_tss = bacc[max_tss_index]
pre_max_tss = pre[max_tss_index]
rec_max_tss = rec[max_tss_index]
tss_max_tss = tss[max_tss_index]
hss_max_tss = hss[max_tss_index]

print('\nMetrics with this threshold:')
print('acc:', acc_max_tss)
print('bacc:', bacc_max_tss)
print('pre:', pre_max_tss)
print('rec:', rec_max_tss)
print('tss:', tss_max_tss)
print('hss:', hss_max_tss)
