import csv
import cv2
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

DEBUG = False
BRIDGE = True
bridge_rate = 1.0

CURVE = True
curve_rate = 1.0

#Helper function to load additional data
def load_additional_data(shear_rate, rate, data_name):
    image_paths = []
    measurements = []
    shear_flags = []
    samples = []

    csv_path = '../' + data_name + '/driving_log.csv'
    with open(csv_path) as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        samples.append(line)
    del(samples[0])

    for sample in samples:
        if np.random.random() < bridge_rate:
            filename = sample[0].split('/')[-1]
            current_path = '../' + data_name +'/IMG/'
            current_path = current_path + filename
            image_paths.append(current_path)
            measurement = float(sample[3])
            measurements.append(measurement)
            shear_flags.append(False)

            if np.random.random() < shear_rate:
                image_paths.append(current_path)
                measurements.append(measurement)
                shear_flags.append(True)

    return image_paths, measurements, shear_flags

#Load Udacity sample data and additional data
def load_data():
    del_rate = 0.85
    del_angle = 0.03
    shear_rate = 0

    samples = []
    with open('../data/driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
        samples.append(line)
    del(samples[0])

    image_paths = []
    measurements = []
    shear_flags = []

    for sample in samples:
        angle = abs(float(sample[3]))
        if angle < del_angle:
            if np.random.random() < del_rate:
                continue
        camera_prob = np.random.random()
        i = 0
        #Randomly choose which camera data to use
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
        shear_flags.append(False)

        #Data augumentation with shear is NOT used in the end.
        # shear_rate is set 0 since it is not used.
        if np.random.random() < shear_rate:
            image_paths.append(current_path)
            measurements.append(measurement)
            shear_flags.append(True)

    #Load images and steering angle data on the bridge.
    if BRIDGE:
        img_pth, mea, sha = load_additional_data(shear_rate, bridge_rate, 'bridge_data')
        image_paths += img_pth
        measurements += mea
        shear_flags += sha

    #Load images and steering angle data at two steep curves
    if CURVE:
        img_pth, mea, sha = load_additional_data(shear_rate, curve_rate, 'curve_data')
        image_paths += img_pth
        measurements += mea
        shear_flags += sha

        img_pth, mea, sha = load_additional_data(shear_rate, curve_rate, 'curve_data2')
        image_paths += img_pth
        measurements += mea
        shear_flags += sha

    data = np.column_stack((image_paths, measurements, shear_flags))
    data = shuffle(data)

    if DEBUG:
        plot_data(measurements)
    return data

#Debug function to see the data historgram
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
                #Crop to 70 x 320 and bottom 25 x 320 pixels
                img = img[70:-25, :, :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img = cv2.GaussianBlur(img, (3,3), 0)
                height = 64
                width = 64
                img = cv2.resize(img,(width, height))
                measurement = float(batch_sample[1])

                #Affine transform is not used in the end.
                #Batch_sample[2] is always False
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
            yield sklearn.utils.shuffle(X_train, y_train)

#Model function
def model():
    samples = load_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.layers import Convolution2D, Cropping2D, Lambda, Activation

    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3),output_shape=(64,64,3)))

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
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
                nb_val_samples=len(validation_samples), verbose=1, nb_epoch=4)
    model.save('./model.h5')

#samples = load_data()
#print(samples.shape)

model()
