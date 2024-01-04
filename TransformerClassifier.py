import keras
from keras import layers
from keras.optimizers import Adam
from tensorflow import keras

# create transformer encoder block using given hyperparameters
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # residual connection with initial inputs

    # Feed forward part - Conv1D w/ dropout
    x = layers.LayerNormalization(epsilon=1e-6)(res)  # normalize from addition of residual
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)
    return x + res  # residual connection with pre-feed-forward values


# build transformer model using encoder from prior function, given hyperparams and input vector shape
def build_model(hp, input_shape=(122, 18, )):  # input length changed depending on data window length
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # hyperparameters for transformer blocks and optimizer
    num_transformer_blocks = hp.Choice('num_transformer_blocks', values=[1, 2, 3, 4])
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2])

    # for however many encoder blocks are chosen, create one with listed hyperparameter search spaces
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x,
                                head_size=hp.Choice('head_size', values=[32, 64, 128, 256]),
                                num_heads=hp.Choice('num_heads', values=[2, 4, 8]),
                                ff_dim=hp.Choice('ff_dim', values=[2, 4, 8]),
                                dropout=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
                                )

    # global average pooling layer
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)

    # hyperparameters for MLP with dropout
    mlp_units = hp.Choice('mlp_units', values=[256, 128, 64, 32])
    mlp_units = [mlp_units]
    mlp_dropout = hp.Float('mlp_dropout', min_value=0, max_value=0.5, step=0.1)

    # add MLP layers - densely connected w/ dropout @ end
    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dense(dim / 2, activation='tanh')(x)
        x = layers.Dense(dim / 4, activation='tanh')(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)  # final output sigmoid layer - continuous (0,1) output

    # model training parameters
    model = keras.Model(inputs, outputs)
    optimizer = Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()
    roc_auc = keras.metrics.AUC(name='roc_auc')
    pr_auc = keras.metrics.AUC(name='pr_auc')
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', roc_auc, pr_auc])

    return model