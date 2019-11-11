#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from imutils.video import VideoStream
from tkinter import *
from PIL import Image, ImageTk
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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

X, y = load_data(DATA_DIR, CATEGORIES, IMG_WIDTH, IMG_HEIGHT)


# In[3]:


# Divide our data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)

# Normalize our data
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because we are using greyscale, we only have a single channel - RGB colour images would have 3
X_train = X_train.reshape(X_train.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
X_test = X_test.reshape(X_test.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)

# convert the data to the right type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[4]:


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


# In[5]:


# run and train model
classifier.fit(np.array(X_train), np.array(y_train),
               batch_size=BATCH_SIZE,
               epochs=13, 
               verbose=1,
               validation_data=(np.array(X_test), np.array(y_test))
               )


# In[ ]:


# test model
score = classifier.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# In[13]:    
# def live_detection(model):
#     video_source = VideoStream(src=0).start()
    
#     while True:
#         frame = detect_and_display(model, video_source)
#         cv2.imshow("Face Liveness Detector", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()
#     video_source.stop()
    
class Application(Frame):
    def play_game(self):
        print("Let's play a game...")
        self.update_background("s")
    
    def exit_game(self):
        self.state = "closed"

    def detect(self, image):
        # Convert image for model
        small = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # Reshape for input to NN
        img_arr = gray.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
        # Cast to float to handle error
        img_arr = tf.cast(img_arr, tf.float32)
        # Save colour image to show to user
        colour = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        prediction = self.classifier.predict(img_arr)
        # Convert prediction from one hot to category
        index = tf.argmax(prediction[0], axis=0)
        prediction = CATEGORIES[index]
        
        cv2.putText(image, prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 2)
        return image

    def createWidgets(self):
        image2 = Image.open("rockpaperscissors/rock/0bioBZYFCXqJIulm.png")
        image = ImageTk.PhotoImage(image2)
        self.IMAGE = Label(self, image=image)
        self.IMAGE.image = image
        self.IMAGE.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.BUTTONS = Frame(self)
        self.BUTTONS.pack(side="top")

        self.QUIT = Button(self.BUTTONS)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["height"]   = "3"
        self.QUIT["width"]   = "15"
        self.QUIT["command"] =  self.exit_game
        self.QUIT.pack(side="left")

        self.PLAY = Button(self.BUTTONS)
        self.PLAY["text"] = "PLAY",
        self.PLAY["height"]   = "3"
        self.PLAY["width"]   = "15"
        self.PLAY["command"] = self.play_game
        self.PLAY.pack(side="left")

    def update_background_opencv(self, cv_image):
        image = ImageTk.PhotoImage(Image.fromarray(cv_image))
        self.IMAGE.image = image
        self.IMAGE.configure(image=image)

    def update_background(self, image_name):
        image2 = Image.open("rockpaperscissors/scissors/0zoQAmDFXehOZsAp.png")
        image = ImageTk.PhotoImage(image2)
        self.IMAGE.image = image
        self.IMAGE.configure(image=image)

    def __init__(self, master=None, model=0, classifier=None):
        Frame.__init__(self, master)
        self.state = "liveimage"
        self.model = model
        self.pack(side="top", fill="both", expand=1)
        self.createWidgets()

def GUI_live_detection(model, width=1080, height=720, posx=300, posy=100):
    root = Tk()
    geom = "{}x{}+{}+{}".format(width, height, posx, posy)
    root.geometry(geom)
    app = Application(master=root, model=1, classifier=classifier)
    
    video_source = VideoStream(src=0).start()

    while app.state != "closed":
        if app.state == "liveimage":
            image = video_source.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            app.update_background_opencv(image)
        app.update_idletasks()
        app.update()
    video_source.stop()
    root.destroy()
    
GUI_live_detection(1)