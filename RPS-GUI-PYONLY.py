import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from imutils.video import VideoStream
import tkinter as tk
from PIL import Image, ImageTk
import time

# Locally read in data and make our training and testing set
DATA_DIR = "rockpaperscissors"
CATEGORIES = ["paper", "rock", "scissors"]
IMG_WIDTH = 150
IMG_HEIGHT = 100
BATCH_SIZE = 100
    
def load_data(data_dir, categories, img_width, img_height):
    X = []
    y = []
    index = -1
    for category in categories:
        index += 1
        one_hot = np.zeros(len(categories))  # to encode the class as a one hot vector
        one_hot[index] = 1
        path = os.path.join(data_dir, category)
        
        for img in os.listdir(path):  # get all images in the path
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (img_width, img_height))
            img_arr = np.asarray(img_arr)
            X.append(img_arr)
            y.append(one_hot)
            
    return X, y

def train_model():
    X, y = load_data(DATA_DIR, CATEGORIES, IMG_WIDTH, IMG_HEIGHT)

    # Configure the CNN
    classifier = tf.keras.models.Sequential()

    # Create model
    # 3x 2d convolution layers
    # Non-linearity (RELU) - replace all negative pixel values in feature map with zero
    classifier.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 1), activation='relu')) 
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    classifier.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    classifier.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten 3d model into 1d
    classifier.add(tf.keras.layers.Flatten())

    # feature vectors
    classifier.add(tf.keras.layers.Dense(64, activation='relu'))
    classifier.add(tf.keras.layers.Dropout(0.5))
    classifier.add(tf.keras.layers.Dense(3, activation='softmax'))


    # compile model
    classifier.compile(loss='categorical_crossentropy',
                    optimizer="rmsprop",
                    metrics=['accuracy'])

    # prep model for training

    # split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)

    # normalize data
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # reshape data into tensor
    X_train = X_train.reshape(X_train.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
    X_test = X_test.reshape(X_test.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # run and train model
    classifier.fit(np.array(X_train), np.array(y_train),
                batch_size=BATCH_SIZE,
                epochs=10, 
                verbose=1,
                validation_data=(np.array(X_test), np.array(y_test))
                )

    # test model
    score = classifier.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    tf.keras.models.save_model(classifier, "RPS_GUI.h5", True)
    
class Application(tk.Frame):
    def load_classifier(self):
        self.classifier = tf.keras.models.load_model("RPS_GUI.h5")

    def train_model(self):
        self.update_prediction_text("TRAINING")
        self.update_background(np.zeros((self.height, self.width, 3), np.uint8))
        self.update_idletasks()
        self.update()
        train_model()
        self.load_classifier()

    def play_game(self):
        self.play_time = time.time()
        self.update_prediction_text("3")
        print("Let's play a game...")
    
    def exit_game(self):
        self.state = "closed"

    def update_prediction_text(self, text):
        self.PREDICTION.configure(text=text)
        self.prediction_time = time.time()

    def detect_and_display(self):
        # Get frame from video soruce
        frame = self.video_source.read()
        
        # Convert image for model
        small = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # Reshape for input to NN
        img_arr = gray.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
        # Cast to float to handle error
        img_arr = tf.cast(img_arr, tf.float32)
        # Save colour image to show to user
        colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        prediction = self.classifier.predict(img_arr)
        # Convert prediction from one hot to category
        index = tf.argmax(prediction[0], axis=0)
        prediction = CATEGORIES[index]

        self.update_prediction_text(prediction)
        
        #cv2.putText(frame, prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 2)

    def createWidgets(self):
        self.IMAGE = tk.Label(self)
        self.IMAGE.image = None
        self.IMAGE.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.BUTTONS = tk.Frame(self)
        self.BUTTONS.pack(side="top")

        self.QUIT = tk.Button(self.BUTTONS)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["height"]   = "3"
        self.QUIT["width"]   = "15"
        self.QUIT["command"] =  self.exit_game
        self.QUIT.pack(side="left")

        self.PLAY = tk.Button(self.BUTTONS)
        self.PLAY["text"] = "PLAY",
        self.PLAY["height"]   = "3"
        self.PLAY["width"]   = "15"
        self.PLAY["command"] = self.play_game
        self.PLAY.pack(side="left")

        self.TRAIN = tk.Button(self.BUTTONS)
        self.TRAIN["text"] = "TRAIN",
        self.TRAIN["height"]   = "3"
        self.TRAIN["width"]   = "15"
        self.TRAIN["command"] = self.train_model
        self.TRAIN.pack(side="left")

        self.PREDICTION = tk.Label(self, text="PREDICTION")
        self.PREDICTION.pack(side="top")

    def update_background(self, cv_image):
        image = ImageTk.PhotoImage(Image.fromarray(cv_image))
        self.IMAGE.image = image
        self.IMAGE.configure(image=image)

    def update_background_from_video(self):
        image = self.video_source.read()#detect_and_display(model, video_source)
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.update_background(image)

    def update_components(self):
        if self.play_time > 0:
            if self.play_time + 3 < time.time():
                self.play_time = -1
                self.update_prediction_text("Shoot")
                self.detect_and_display()
            elif self.play_time + 2 < time.time():
                self.update_prediction_text("1")
            elif self.play_time + 1 < time.time():
                self.update_prediction_text("2")
        elif self.prediction_time + 3.0 < time.time():
            self.update_prediction_text("")
        if self.state == "liveimage":
            self.update_background_from_video()

    def __init__(self, master=None, video_source=None, width=1080, height=720):
        tk.Frame.__init__(self, master)
        self.state = "liveimage"
        self.video_source = video_source
        self.load_classifier()
        self.pack(side="top", fill="both", expand=1)
        self.createWidgets()
        self.prediction_time = time.time()
        self.width = width
        self.height = height
        self.play_time = -1

def GUI_live_detection(model, width=1080, height=720, xpos=300, ypos=100):
    root = tk.Tk()
    root.title("Rock Paper Scissors")
    geom = "{}x{}+{}+{}".format(width, height, xpos, ypos)
    root.geometry(geom)
    video_source = VideoStream(src=0).start()
    app = Application(master=root, video_source=video_source, width=width, height=height)

    while app.state != "closed":
        app.update_components()
        app.update_idletasks()
        app.update()
    
    video_source.stop()
    root.destroy()

#train_model()  
GUI_live_detection(1)