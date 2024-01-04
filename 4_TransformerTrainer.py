import keras_tuner
import tensorflow as tf
import joblib
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from TransformerClassifier import build_model
import os
from datetime import datetime

list_of_type = ['C_24']  # flare prediction case to train model on
window = 24

for type in list_of_type:

    print('=========================================================================================\n')
    print('Now I am training a Transformer Network for data cut-off at {}-type flares for {}h time delay\n'.format(type[0:1], type[2:]))
    print('=========================================================================================\n')

    input_path = 'your/own/path'+str(window)+'/'+type+'/'
    out_path = 'your/own/path'+datetime.today().strftime('%Y_%m_%d')+'_transformer_'+str(window)+'_'+type+'/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load train, validation, test data from make_data_for_train_val_test.py
    x_train = joblib.load(input_path + 'x_train.pkl')
    y_train = joblib.load(input_path + 'y_train.pkl')
    x_val = joblib.load(input_path + 'x_val.pkl')
    y_val = joblib.load(input_path + 'y_val.pkl')
    x_test = joblib.load(input_path + 'x_test.pkl')
    y_test = joblib.load(input_path + 'y_test.pkl')

    print('Number of class 0:', len(x_train[y_train == 0]))
    print('Number of class 1:', len(x_train[y_train == 1]))

    # build transformer model from TransformerClassifier file
    build_model(keras_tuner.HyperParameters())

    objective = 'val_roc_auc'  # metric to optimize model parameters in training

    # optimize model hyperparameters via Bayesian Optimization
    tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                             objective=keras_tuner.Objective(objective, direction='max'),
                                             max_trials=10, executions_per_trial=1, overwrite=False,
                                             directory=out_path, project_name='hyper_'+type)

    tuner.search_space_summary()

    # stop optimization if val_roc_auc does not improve over 5 epochs
    early_stopping = EarlyStopping(monitor=objective, mode='max', patience=5)

    tensor_board_1 = TensorBoard(log_dir=out_path+ 'hyper_'+type+'_tb')

    print('searching for hyperparameters')
    # use selected tuner to train model on training/validation data with early stopping
    tuner.search(x=x_train, y=y_train, epochs=100, validation_data=(x_val, y_val),
                 callbacks=[early_stopping, tensor_board_1])

    # retrieve best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]

    model = tuner.hypermodel.build(best_hp)

    dot_img_file = out_path + 'Transformer_'+type+'.png'

    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    checkpoint_callback = ModelCheckpoint(out_path + 'Transformer_'+type+'.h5',
                                          monitor=objective, verbose=1,
                                          save_best_only=True, mode='max')

    tensor_board = TensorBoard(log_dir=out_path + 'Transformer_'+type+'_tb')

    # re-train model on best hyperparameter combination, again with early stopping
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=100,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint_callback, tensor_board],
                        batch_size=32)

    # save (0-1) continuous model predictions on test set
    pred_y = model.predict(x_test)
    joblib.dump(pred_y, out_path + 'pred_y.pkl')

print('=================================================================================')
print('             Done training Transformer Networks for Flare Prediction             ')
print('=================================================================================')