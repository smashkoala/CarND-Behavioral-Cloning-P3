import csv
import cv2
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

def generator(samples, batch_size=8):
    current_path = '../data/IMG/'
    num_samples = len(samples)
    del_angle = 0.85
    add_rate = 0.7
    while 1:#AP: Not add flipped images to validation samples
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                camera = np.random.choice(['center', 'left', 'right'])
                i = 0
                if camera == 'center':
                    correction = 0.0
                    i = 0
                elif camera == 'left':
                    correction = 0.3
                    i = 1
                else:
                    correction = -0.3
                    i = 2
                filename = batch_sample[i].split('/')[-1]
                add_flag = False
                if abs(float(batch_sample[3])) >= del_angle:
                    add_flag = True
                else:#remove 70% of angle less than 0.85
                    if np.random.random() < add_rate:
                        add_flag = True
                if add_flag is True:
                    img = cv2.imread(current_path + filename)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
                    height = int(img.shape[0]/2)
                    width = int(img.shape[1]/2)
                    img = cv2.resize(img,(width, height))
                    images.append(img)

                    measurement = float(batch_sample[3]) + correction
                    measurements.append(measurement)

                    img_flipped = np.fliplr(img)
                    measurement_flipped = -measurement
                    images.append(img_flipped)
                    measurements.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
del(lines[0])

# images = []
# measurements = []
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = '../data/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#
#     measurement = float(line[3])
#     measurements.append(measurement)

# X_train = np.array(images)
# y_train = np.array(measurements)
import random
#lines = random.shuffle(lines)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
# split_number = int(len(lines)*8/10)
# train_samples = lines[:split_number]
#print(len(train_samples))
train_generator = generator(train_samples, batch_size=32)
# validation_samples = lines[split_number:]
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Cropping2D, Lambda, Activation

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,3),output_shape=(80,160,3)))
model.add(Cropping2D(cropping=((25, 10), (0, 0)), input_shape=(80,160,3)))

model.add(Convolution2D(24, 5, 5))
model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5))
model.add(Activation('elu'))

model.add(Convolution2D(48, 5, 5))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*4, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)*4, verbose=1, nb_epoch=3)
model.save('./model.h5')

import gc; gc.collect()
