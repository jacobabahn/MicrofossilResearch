import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras import layers
from tensorflow.keras import backend as K 
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from matplotlib import pyplot
from livelossplot.tf_keras import PlotLossesCallback
import numpy as np
from sklearn.metrics import classification_report


trainingData = "./Clusters/ScaledClusters/train"
validationData = "./Clusters/ScaledClusters/validate"

epochs = 100
batch_size = 32

def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding ="same", input_shape = (180, 180, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2, 2), padding = "same", activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = "softmax"))
   
    return model

cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
strategy = tf.distribute.MirroredStrategy(["device:GPU:%d" % i for i in range(2)],cross_device_ops=cross_device_ops)  #,"/gpu:1","/gpu:0"]
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  # out 2

with strategy.scope():
    model = create_model()
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = optimizers.Adam(epsilon = 1e-08, learning_rate = 0.001),
        metrics = ["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.3, zoom_range = 0.3, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    trainingData, 
    target_size =(180, 180), 
    batch_size = batch_size, 
    class_mode ='categorical',
    shuffle=True) 

validation_generator = test_datagen.flow_from_directory(
    validationData, 
    target_size =(180, 180), 
    batch_size = batch_size, 
    shuffle=False)

TRAIN_STEPS = train_generator.samples // batch_size
VAL_STEPS = validation_generator.samples // batch_size

with strategy.scope():
    trainingmodel = model.fit(train_generator, 
                                steps_per_epoch = TRAIN_STEPS,
                                epochs = epochs,
                                validation_data = validation_generator,
                                validation_steps = VAL_STEPS,
                                callbacks=[PlotLossesCallback()],
                                verbose=1
                            )

model.save_weights('test_weights.h5')
model.save('test_model.h5')

with strategy.scope():
    predictions = model.predict(validation_generator)

predicted_classes = np.argmax(predictions, axis = 1)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel("string")
    plt.legend([string, "val_"+string])
    return plt

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

accuracy = summarize_diagnostics(trainingmodel)

report = classification_report(true_classes, predicted_classes, target_names = class_labels)

print(model.summary())
print(report)