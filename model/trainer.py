import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
from datetime import datetime
import pickle

from keras import backend as K


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

K.set_image_dim_ordering('th')


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#X = tf.keras.utils.normalize(X, axis=1)
X = X/255.0
now = datetime.now()

dense_layers = [4]
layer_sizes = [24]
conv_layers = [4]
optimizers = ["adam"]
batch_size = [24]
epochs = [20]

# training a shit ton of models in order to try to optimize parameters.
# this is a poor man's RandomSearchCV
for epoch in epochs:
    for batch in batch_size:
        for optimizer in optimizers:
            for dense_layer in dense_layers:
                for layer_size in layer_sizes:
                    for conv_layer in conv_layers:

                        NAME = "{}-opt-{}-conv-{}-nodes-{}-dense-{}-batch-{}-epochs-{}".format(optimizer, conv_layer, layer_size, dense_layer,batch, epoch, now.strftime("%Y%m%d-%H%M%S"))
                        tensorboard = TensorBoard(log_dir='logs/adam-SGD-batch-epochs/{}'.format(NAME))
                        print(NAME)

                        model = Sequential()

                        # Layer 1
                        model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:])) # this input_shape is grabbing everything from when we reshaped X in the create_dataset file. It is grabbing everything after the -1
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                        for l in range(conv_layer-1):
                            model.add(Conv2D(conv_layer, (3, 3)))
                            model.add(Activation("relu"))
                            model.add(MaxPooling2D(pool_size=(2,2)))

                        model.add(Flatten())
                        for l in range(dense_layer):
                            model.add(Dense(layer_size))
                            model.add(Activation("relu"))

                        # Output Layer
                        model.add(Dense(1))
                        model.add(Activation('sigmoid'))

                        model.compile(loss="binary_crossentropy", # because it is dogs/cats, we use binary
                                    optimizer = optimizer,
                                    metrics=['accuracy'])

                        model.fit(X, y, batch_size=batch, epochs=epoch, validation_split=0.3, callbacks=[tensorboard])
                        model.save('model.h5')
                    # use batch_size so we aren't passing the entire dataset in at a time,
                    # validation_split is us using 10% of the dataset as an out of dataset test
                    # it is a more advanced way of training than train_test_split()
