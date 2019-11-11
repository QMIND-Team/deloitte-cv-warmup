import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

class MODEL():  
    def load_data(self):
        X = []
        y = []
        index = -1
        for category in self.CATEGORIES:
            index += 1
            one_hot = np.zeros(len(self.CATEGORIES))  # to encode the class as a one hot vector
            one_hot[index] = 1
            path = os.path.join(self.DATA_DIR, category)
            
            for img in os.listdir(path):  # get all images in the path
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.resize(img_arr, (self.IMG_WIDTH, self.IMG_HEIGHT))
                img_arr = np.asarray(img_arr)
                X.append(img_arr)
                y.append(one_hot)
                
        return X, y

    def train_model(self):
        X, y = self.load_data()

        # Configure the CNN
        self.model = tf.keras.models.Sequential()

        # Create model
        # 3x 2d convolution layers
        # Non-linearity (RELU) - replace all negative pixel values in feature map with zero
        self.model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 1), activation='relu')) 
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Flatten 3d model into 1d
        self.model.add(tf.keras.layers.Flatten())

        # feature vectors
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(3, activation='softmax'))


        # compile model
        self.model.compile(loss='categorical_crossentropy',
                        optimizer="rmsprop",
                        metrics=['accuracy'])

        # prep model for training

        # split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)

        # normalize data
        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_test, axis=1)

        # reshape data into tensor
        X_train = X_train.reshape(X_train.shape[0], self.IMG_WIDTH, self.IMG_HEIGHT, 1)
        X_test = X_test.reshape(X_test.shape[0], self.IMG_WIDTH, self.IMG_HEIGHT, 1)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # run and train model
        self.model.fit(np.array(X_train), np.array(y_train),
                    batch_size=self.BATCH_SIZE,
                    epochs=10, 
                    verbose=1,
                    validation_data=(np.array(X_test), np.array(y_test))
                    )

        # test model
        score = self.model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # save the file for use in future sessions
        tf.keras.models.save_model(self.model, self.FILE, True)

    def detect(self, frame):
        # Convert image for model
        small = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # Reshape for input to NN
        img_arr = gray.reshape(1, self.IMG_WIDTH, self.IMG_HEIGHT, 1)
        # Cast to float to handle error
        img_arr = tf.cast(img_arr, tf.float32)
        # Save colour image to show to user
        colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        prediction = self.model.predict(img_arr)
        # Convert prediction from one hot to category
        index = tf.argmax(prediction[0], axis=0)
        prediction = self.CATEGORIES[index]

        return prediction

    def __init__(self):
        self.FILE = "RPS_GUI.h5"
        self.DATA_DIR = "rockpaperscissors"
        self.CATEGORIES = ["paper", "rock", "scissors"]
        self.IMG_WIDTH = 150
        self.IMG_HEIGHT = 100
        self.BATCH_SIZE = 100

        # load the model if it was saved previously
        if os.path.exists(self.FILE):
            self.model = tf.keras.models.load_model(self.FILE)
        else:
            self.model = None
            self.train_model()