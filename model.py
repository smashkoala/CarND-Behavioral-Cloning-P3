import csv
import cv2
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

DEBUG = False
BRIDGE = True

def load_brige_data(share_rate):
    image_paths = []
    measurements = []
    share_flags = []
    samples = []

    with open('../bridge_data/driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        samples.append(line)
    del(samples[0])

    for sample in samples:
        filename = sample[0].split('/')[-1]
        current_path = '../bridge_data/IMG/' + filename
        image_paths.append(current_path)
        measurement = float(sample[3])
        measurements.append(measurement)
        share_flags.append(False)

        if np.random.random() < share_rate:
            image_paths.append(current_path)
            measurements.append(measurement)
            share_flags.append(True)

        return image_paths, measurements, share_flags

def load_data():
    del_rate = 0.85
    del_angle = 0.03
    share_rate = 0.0

    samples = []
    with open('../data/driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        samples.append(line)
    del(samples[0])

    #print(len(samples))

    image_paths = []
    measurements = []
    share_flags = []

    for sample in samples:
        angle = abs(float(sample[3]))
        if angle < del_angle:
            if np.random.random() < del_rate:
                continue
        camera_prob = np.random.random()
        i = 0
        if camera_prob < 0.4:
            correction = 0.0
            i = 0
        elif camera_prob >= 0.4 and camera_prob < 0.70:
            correction = 0.3
            i = 1
        else:
            correction = -0.3
            i = 2
        filename = sample[i].split('/')[-1]
        current_path = '../data/IMG/' + filename
        image_paths.append(current_path)
        measurement = float(sample[3]) + correction
        measurements.append(measurement)
        share_flags.append(False)

        if np.random.random() < share_rate:
            image_paths.append(current_path)
            measurements.append(measurement)
            share_flags.append(True)

    if BRIDGE:
        img_pth, mea, sha = load_brige_data(share_rate)
        image_paths += img_pth
        measurements += mea
        share_flags += sha

    data = np.column_stack((image_paths, measurements, share_flags))
    data = shuffle(data)

    if DEBUG:
        plot_data(measurements)
    return data

def plot_data(data):
    plt.hist(data, bins = 40, rwidth=0.8)
    fig = plt.gcf()
    plt.show()
    fig.savefig("histogram.jpg")

def generator(samples, batch_size=8):
    num_samples = len(samples)
    shift = 10

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                img = cv2.imread(batch_sample[0])
                img = img[70:-25, :, :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img = cv2.GaussianBlur(img, (3,3), 0)
                # height = int(img.shape[0]/2)
                # width = int(img.shape[1]/2)
                height = 64
                width = 64
                img = cv2.resize(img,(width, height))
                measurement = float(batch_sample[1])

                if DEBUG:
                    plt.imshow(img)
                    plt.show()
                    return

                if batch_sample[2]:
                    pts1 = np.float32([[width,0],[0,0],[width/2,height]])
                    pts2 = np.float32([[width,0],[0,0],[width/2+shift,height]])
                    M = cv2.getAffineTransform(pts1,pts2)
                    img = cv2.warpAffine(img, M, (width,height),borderMode=cv2.BORDER_REPLICATE)

                if np.random.random() < 0.5:
                    img = np.fliplr(img)
                    measurement = measurement*(-1.0)
                images.append(img)
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

        if DEBUG:
            return X_train, y_train
        else:
            yield sklearn.utils.shuffle(X_train, y_train)

def model():
    samples = load_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=8)
    validation_generator = generator(validation_samples, batch_size=8)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.layers import Convolution2D, Cropping2D, Lambda, Activation

    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3),output_shape=(64,64,3)))
    # model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,3),output_shape=(80,160,3)))
#    model.add(Cropping2D(cropping=((35, 12), (0, 0)), input_shape=(80,160,3)))

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
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
                nb_val_samples=len(validation_samples), verbose=1, nb_epoch=4)
    model.save('./model.h5')

if DEBUG:
    samples = load_data()
    generator(samples)
else:
    model()
