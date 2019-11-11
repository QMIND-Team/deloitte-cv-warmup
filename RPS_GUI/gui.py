import tensorflow as tf
import numpy as np
import cv2
from imutils.video import VideoStream
import tkinter as tk
from PIL import Image, ImageTk
import time

class GUI(tk.Frame):
    def train_model(self):
        self.update_prediction_text("TRAINING")
        self.update_background(np.zeros((self.height, self.width, 3), np.uint8))
        self.update_idletasks()
        self.update()
        self.model.train_model()

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
        prediction = self.model.detect(frame)
        self.update_prediction_text(prediction)

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
        image = self.video_source.read()
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

    def __init__(self, master=None, model=None, video_source=None, width=1080, height=720):
        tk.Frame.__init__(self, master)
        self.state = "liveimage"
        self.video_source = video_source
        self.pack(side="top", fill="both", expand=1)
        self.createWidgets()
        self.prediction_time = time.time()
        self.width = width
        self.height = height
        self.play_time = -1
        self.model = model