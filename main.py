# Predicting Orientation from Images

# Helper fuction to read the input (whether train or test)
def reading_file_name(base_dir):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
    return onlyfiles

# Reading their names
test_files_name = reading_file_name('Large Files/test.rotfaces/test/')
train_files_name = reading_file_name('Large Files/train.rotfaces/train/')

# Getting them to memory (Here only the TrainSet)
from imageio import imread
from os.path import join

track = len(train_files_name)
count = 0

train = dict()
for f in train_files_name:
    train[f] = (imread(join('Large Files/train.rotfaces/train/', f)))
    count += 1
    print('{0}%'.format(round((count/track)*100), 2), end='\r')

# Reading the Labels
from pandas import read_csv
labels = read_csv('Large Files/train.rotfaces/train.truth.csv')

# Ordering the Set with the labels
X = []
for item in labels.fn:
    X.append(train[item])

Ya = []
for item in labels.label:
    Ya.append(item)

# Transform the labels on Categorical in 2 ways
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Ya)
print(le.classes_)

Y = le.transform(Ya)
print(Y)

from tensorflow.keras.utils import to_categorical
Y = to_categorical(Y, num_classes=4)

# Creating the Split Sets
from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(X, Y)

# Taking them to float16
from numpy import array
Xtr = array(Xtr).astype('float16')
Xte = array(Xte).astype('float16')
Ytr = array(Ytr).astype('float16')
Yte = array(Yte).astype('float16')
print('{0}, {1}, {2}, {3}'.format(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape))

# Starting the Model
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# Problem Parameters
batch_size = 32
num_classes = 4
epochs = 20
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_rotated_images.h5'

# Starting the Model, using the Cifar10 model.
# * If you desire to load the model instead. comment this piece o code from here
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=Xtr.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
# An evolution of RProp which helps with the gradient descent problem
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
# Using categorical_crossentropy for multi-class classification
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

Xtr /= 255
Xte /= 255

# Fitting the model
print('Not using data augmentation.')
model.fit(Xtr, Ytr,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(Xte, Yte),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
# * To here.

# # Load Model, a model has been saved before.
# # Uncomment this if you prefere load the model right away
# from keras.models import load_model
# model_path = os.path.join(save_dir, model_name)
# model = load_model(model_path)

# Score trained model.
scores = model.evaluate(Xte, Yte, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Predictions on the Test Set

track = len(test_files_name)
count = 0

test = dict()
for f in test_files_name:
    test[f] = (imread(join('/home/souza/Documents/ds-test/Large Files/test.rotfaces/test', f)))
    count += 1
    print('{0}%'.format(round((count/track)*100), 2), end='\r')

Ynew = dict()
for key in test:
    Ynew[key] = model.predict_classes(test[key].reshape(1, 64, 64, 3))

CSV = list()
CSV.append(['fn', 'label'])
from numpy import argmax
for item in Ynew:
    CSV.append([item, le.inverse_transform(Ynew[item])[0]])

# Saving it on the CSV file.
from pandas import DataFrame
DataFrame(CSV).to_csv('Large Files/test.rotfaces/test.preds.csv', index=False, header=False)
