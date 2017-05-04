import csv
import cv2
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def generator(samples, batch_size=8):
    current_path = '../data/IMG/'
    num_samples = len(samples)
    del_angle = 0.1
    add_rate = 0.7
    debug = False
    while 1:#AP: Not add flipped images to validation samples
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            number_center = 0
            number_left = 0
            number_right = 0
            number_low_ang = 0
            number_high_ang = 0
            number_flipped = 0
            for batch_sample in batch_samples:
                camera = np.random.choice(['center', 'left', 'right'])
                i = 0
                if camera == 'center':
                    correction = 0.0
                    i = 0
                    number_center += 1
                elif camera == 'left':
                    correction = 0.25
                    i = 1
                    number_left += 1
                else:
                    correction = -0.25
                    i = 2
                    number_right += 1
                filename = batch_sample[i].split('/')[-1]
                add_flag = False
                if abs(float(batch_sample[3])) >= del_angle:
                    add_flag = True
                    number_high_ang += 1
                else:#remove 70% of angle less than 0.85
                    ran = np.random.rand()
                    if ran < add_rate:
                        add_flag = True
                        number_low_ang += 1

                if add_flag is True:
                    img = cv2.imread(current_path + filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height = int(img.shape[0]/2)
                    width = int(img.shape[1]/2)
                    img = cv2.resize(img,(width, height))
                    img = augment_brightness_camera_images(img)
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
                    images.append(img)

                    measurement = float(batch_sample[3]) + correction
                    measurements.append(measurement)

                    flip_prob = np.random.random()
                    if flip_prob > 0.5:
                        img_flipped = np.fliplr(img)
                        img_flipped = augment_brightness_camera_images(img_flipped)
                        measurement_flipped = -measurement
                        images.append(img_flipped)
                        measurements.append(measurement_flipped)
                        number_flipped += 1

            X_train = np.array(images)
            y_train = np.array(measurements)

            if debug is True:
                print("¥n")
                print("Total:", len(images))
                print("Center image:", number_center)
                print("Right image:", number_right)
                print("Left image:", number_left)
                print("High angle:", number_high_ang)
                print("Low angle", number_low_ang)
                print("flipped", number_flipped)

            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
del(lines[0])


import random

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Cropping2D, Lambda, Activation

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,3),output_shape=(80,160,3)))
model.add(Cropping2D(cropping=((35, 12), (0, 0)), input_shape=(80,160,3)))

model.add(Convolution2D(24, 5, 5,subsample=(1,1),activation="elu"))
#model.add(Activation('elu'))

model.add(Convolution2D(36, 5, 5,subsample=(1,1),activation="elu"))
#model.add(Activation('elu'))

model.add(Convolution2D(48, 5, 5,subsample=(1,1),activation="elu"))
#model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3,activation="elu"))
#model.add(Activation('elu'))

model.add(Convolution2D(64, 3, 3,activation="elu"))
#model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)*6, verbose=1, nb_epoch=4)
model.save('./model.h5')

import gc; gc.collect()
