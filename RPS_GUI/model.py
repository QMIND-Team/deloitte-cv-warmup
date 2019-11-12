import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import random
import sys, getopt
import imutils

class MODEL():
    def load_model(self, location):
        # load the model if it was saved previously
        if os.path.exists(location):
            self.model = tf.keras.models.load_model(location)
            return True
        return False

    def prep_image_jd(self, image):
        # Convert image for model
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def prep_image_eric(self, image):
        # Convert image for model
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def prep_image_brendon(self, image):
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = tf.keras.utils.normalize(image)
        return image

    def prep_image_connor(self, image):
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, 100, 200)
        return image

    def prep_image(self, image):
        if self.USER == "JD":
            return self.prep_image_jd(image)
        elif self.USER == "ERIC":
            return self.prep_image_eric(image)
        elif self.USER == "BRENDON":
            return self.prep_image_brendon(image)
        elif self.USER == "CONNOR":
            return self.prep_image_connor(image)

    def generate_new_images_connor(self, image):
        images = [image]
        images.append(cv2.flip(image, 0))
        for i in range(5):
            img = imutils.rotate(image, (random.random()-0.5)*20.0, scale=(0.5+(random.random()*0.5)))
            img = imutils.translate(img, random.random()*20, random.random()*20)
            images.append(img)
            images.append(cv2.flip(img, 0))
        return images

    def generate_new_images(self, image):
        if self.USER == "JD":
            return[image]
        elif self.USER == "ERIC":
            return[image]
        elif self.USER == "BRENDON":
            return[image]
        elif self.USER == "CONNOR":
            return self.generate_new_images_connor(image)
            
    def load_data(self):
        X = []
        y = []
        images = []
        index = -1
        for category in self.CATEGORIES:
            index += 1
            one_hot = np.zeros(len(self.CATEGORIES))  # to encode the class as a one hot vector
            one_hot[index] = 1
            path = os.path.join(self.DATA_DIR, category)
            
            for img_name in os.listdir(path):  # get all images in the path
                image = cv2.imread(os.path.join(path, img_name))
                image = self.prep_image(image)
                for img in self.generate_new_images(image):
                    images.append(img)
                    img_arr = np.asfarray(img)
                    X.append(img_arr)
                    y.append(one_hot)
        self.images = images
        return X, y

    def init_model_erik(self):
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

    def init_model_brendon(self):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(34, kernel_size=(3, 3), strides=(1,1), activation="relu", input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1)))

        self.model.add(tf.keras.layers.Conv2D(34, (3, 3), activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(45, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(.45))
        self.model.add(tf.keras.layers.Dense(len(self.CATEGORIES), activation="softmax"))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=0.02),
                    metrics=['accuracy'])

    def init_model_connor(self):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(34, kernel_size=(3, 3), strides=(2,2), activation="relu", input_shape=(self.IMG_WIDTH, self.IMG_HEIGHT, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(4,4), strides=(1,1)))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(45, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(.45))
        self.model.add(tf.keras.layers.Dense(len(self.CATEGORIES), activation="softmax"))

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=0.02),
                    metrics=['accuracy'])

    def init_model(self):
        if self.USER == "JD":
            self.init_model_erik()
        elif self.USER == "ERIC":
            self.init_model_erik()
        elif self.USER == "BRENDON":
            self.init_model_brendon()
        elif self.USER == "CONNOR":
            self.init_model_connor()

    def train_model(self):
        self.init_model()

        X, y = self.load_data()
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

    def detect(self, image):
        # Save colour image to show to user
        image = self.prep_image(image)
        image = image.reshape(1, self.IMG_WIDTH, self.IMG_HEIGHT, 1)
        # Cast to float to handle error
        image = tf.cast(image, tf.float32)
        prediction = self.model.predict(image)
        # Convert prediction from one hot to category
        index = tf.argmax(prediction[0], axis=0)
        return self.CATEGORIES[index]

    def __init__(self, user, width=150, height=100, batch=100):
        self.USER = user
        self.FILE = "RPS_GUI_{}.h5".format(user)
        self.DATA_DIR = "../rockpaperscissors"
        self.CATEGORIES = ["paper", "rock", "scissors"]
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.BATCH_SIZE = batch
        self.images = []
        if not self.load_model(self.FILE):
            self.train_model()

def main(argv):
    filename=None
    try:
        opts, args = getopt.getopt(argv,"hdf:",["file=", "default"])
    except getopt.GetoptError:
        print("INCORRECT FORMAT: \"model.py -f <output_file> | -d\"")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("model.py -f <output_file> | -d")
            sys.exit()
        elif opt in ("-f", "--file"):
            filename=arg
        elif opt in ("-d", "--default"):
            filename="default"
    if filename == None:
        print("Please specify filename")
        sys.exit()
    model = MODEL("CONNOR")

    if filename != "default":
        model.FILE = filename
    model.train_model()

if __name__ == '__main__':
	main(sys.argv[1:])