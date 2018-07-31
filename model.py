import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers import Flatten,Dense,Lambda,Dropout


def preprocess(img):
    '''
    preprocess the image data:crop the top 50 and bottom 20 pixels of the image,
    add GaussianBlur, resize the image size to match the nvidia network model input,
    lastly change the color space from BGR to YUV
    '''
    new_img = img[50:140,:,:]
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)

    return new_img

images=[]
measurements = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile,skipinitialspace = True, delimiter=',')
  for row in reader:
    steering_center = float(row[3])

    #create adjusted steering measurement for the side camera images
    correction = 0.25 # this parameter need to tune
    steering_left = steering_center + correction
    steering_right= steering_center - correction

    #read in images from center, left, right cameras
    path = './data/'
    image_center = preprocess(cv2.imread(path+row[0]))
    image_left   = preprocess(cv2.imread(path+row[1]))
    image_right  = preprocess(cv2.imread(path+row[2]))

    #add images and measurement to data set
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)


X_train = np.array(images)
y_train = np.array(measurements)


def generator(input_samples,label_samples,batch_size=128):
    '''
    create a generator,  which is a great way to work with large amounts of data.
    Instead of storing the preprocessed data in memory all at once, using a generator
    it pulls pieces of the data and process them on the fly only when they are needed,
    which is much more memory-efficient.
    '''
    if (len(input_samples)!= len(label_samples)):
        raise ValueError("inputs and labels dimension don't match!!!")

    num_samples = len(input_samples)
    while 1:
        inputs,labels = shuffle(input_samples,label_samples)
        for offset in range(0, num_samples, batch_size):
            inputs_samples = inputs[offset:offset+batch_size]
            labels_samples = labels[offset:offset+batch_size]

            yield shuffle(inputs_samples,labels_samples)


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(66,200,3))) #normalize the input data

#nvidia network model
#first create THREE 5x5 convolutional network
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode='valid',W_regularizer=l2(0.001),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode='valid',W_regularizer=l2(0.001),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode='valid',W_regularizer=l2(0.001),activation="relu"))
#then create TWO 3x3 convolutional network
model.add(Convolution2D(64,3,3,border_mode='valid',W_regularizer=l2(0.001),activation="relu"))
model.add(Convolution2D(64,3,3,border_mode='valid',W_regularizer=l2(0.001),activation="relu"))
#Then flatten and create three fully connected layers with Dropout
model.add(Flatten())
model.add(Dense(100,W_regularizer=l2(0.001),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50,W_regularizer=l2(0.001),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,W_regularizer=l2(0.001),activation="relu"))
model.add(Dropout(0.5))
#Finally create one output:vehicle control
model.add(Dense(1))
#use mse for loss and Adam optimizer
model.compile(loss='mse', optimizer='adam')

#split the trainning data for both training and validation
X_train,X_valid,y_train,y_valid = train_test_split(X_train, y_train, test_size=0.2)

#train the model and validate the accuracy
model.fit_generator(generator(X_train,y_train,batch_size=256),samples_per_epoch=len(X_train), \
                    validation_data=generator(X_valid,y_valid,batch_size=256), nb_val_samples=len(y_train), nb_epoch=3, verbose=1)

#save the trained model data for future use
model.save('model.h5')
exit()
