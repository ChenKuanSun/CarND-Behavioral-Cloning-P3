import cv2,csv, os
import numpy as np
import sklearn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Lambda, Cropping2D
from keras.callbacks import TensorBoard
def process_image(image):
    #Convert to HSV to adjust brightness
    HSV_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    #Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    #Apply the brightness reduction to the V channel
    HSV_image[:,:,2] = HSV_image[:,:,2]*random_bright

    #Convert to RBG again
    HSV_image = cv2.cvtColor(HSV_image,cv2.COLOR_HSV2RGB)

    #Cut image
    Cut_image = HSV_image[55:135, :, :]

    #Convert to np.array
    processed_image = Cut_image.astype(np.float32)
    return processed_image


def create_model():
    model = Sequential()
    #Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80, 320,3)))

    model.add(Conv2D(24, 5,5,subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5,5,subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5,5,subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3,3, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(64, 3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def dataset():
    #initialization data set
    images = []
    steering_angles = []
    #import data
    project_path = './dataset/'
    datalist = os.listdir(project_path)
    
    for datapath in datalist:
        lines = []
        print("Loading Data: " + datapath)
        with open(project_path + datapath + "/driving_log.csv",newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for line in reader:
                lines.append(line)
        print("Process image")
        for line in lines:
            #Drop some 0 steering data
            if float(line[3])== 0.0:
                if np.random.choice([True, False]):
                    continue
            
            img_center_filepath = project_path + datapath + '/IMG/' + line[0].split('\\')[-1]
            img_left_filepath = project_path + datapath + '/IMG/' + line[1].split('\\')[-1]
            img_right_filepath = project_path + datapath + '/IMG/' + line[2].split('\\')[-1]

            #import image and preprocess
            img_center = process_image(cv2.imread(img_center_filepath))
            img_left = process_image(cv2.imread(img_left_filepath))
            img_right = process_image(cv2.imread(img_right_filepath))

            steering_center = float(line[3])*1
            #Create adjusted steering measurement for the side camera images
            correction = 0.25 

            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

            #flipped image
            img_center_flipped = np.fliplr(img_center)
            img_left_flipped = np.fliplr(img_left)
            img_right_flipped = np.fliplr(img_right)

            steering_center_flipped = steering_center*-1.0
            steering_left_flipped = steering_left*-1.0
            steering_right_flipped = steering_right*-1.0

            images.extend([img_center_flipped, img_left_flipped, img_right_flipped])
            steering_angles.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])
    print("Dataset complete.")

    X_train = np.array(images)
    y_train = np.array(steering_angles)
    return X_train, y_train

if __name__ == "__main__":
    #dataset
    X_train, y_train = dataset()
    #Setting config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)
    model_callback = [TensorBoard(log_dir='./log')]

    print("Create model")
    model = create_model()

    
    model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=10, callbacks=model_callback)
    model.save('model.h5')
    model.summary()