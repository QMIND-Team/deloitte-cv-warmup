import tensorflow as tf
import numpy as np
import cv2
from imutils.video import VideoStream
import tkinter as tk
from PIL import Image, ImageTk
import time
import random
import threading
import os

class GUI(tk.Frame):
    def train_model_thread(self):
        temp = self.state
        self.state = "training"
        self.model.train_model()
        self.state = temp

    def load_model(self):
        location = self.LOAD_LOC.get()
        if not location.endswith(".h5"):
            self.update_prediction_text("ERROR\nIncorrect file type (not *.h5)")
        elif self.model.load_model(location):
            self.update_prediction_text("MODEL LOADED")
        else:
            self.update_prediction_text("ERROR\nFile does not exist")

    def train_model(self):
        self.play_time = -1
        self.thread_start_time = time.time()
        self.update_prediction_text("TRAINING")
        self.update_background(np.zeros((self.height, self.width, 3), np.uint8))
        self.image_index = 0
        self.train_thread = threading.Thread(target=self.train_model_thread)
        self.train_thread.start()

    def play_game(self):
        self.play_time = time.time()
    
    def exit_game(self):
        if self.train_thread != None and self.train_thread.is_alive:
            self.train_thread.join()
        self.state = "closed"
    
    def toggle_canny(self):
        if self.show_prep_image:
            self.show_prep_image = False
            self.CANNY["text"] = "SHOW PREP"
        else:
            self.show_prep_image = True
            self.CANNY["text"] = "HIDE PREP"

    def update_prediction_text(self, text):
        self.PREDICTION.configure(text=text)
        self.prediction_time = time.time()

    def win_or_loss(self, user, bot):
        if user == bot:
            return "TIE"
        elif (
            user == "rock" and bot == "scissors" or
            user == "paper" and bot == "rock" or
            user == "scissors" and bot == "paper"
        ):
           return "WIN"
        return "LOSS"

    def detect_and_display(self):
        # Get frame from video soruce
        frame = self.video_source.read()
        user = self.model.detect(frame)
        bot = self.model.CATEGORIES[random.randint(0,2)]
        win = self.win_or_loss(user, bot)
        msg = "{}\nPrediction: {} vs {}".format(win, user, bot)
        self.update_prediction_text(msg)
        self.play_time = time.time() + 3

    def createWidgets(self):
        self.IMAGE = tk.Label(self)
        self.IMAGE.image = None
        self.IMAGE.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.ACTIONS_FRAME = tk.Frame(self)
        self.ACTIONS_FRAME.pack(side="top", fill="x")

        self.QUIT = tk.Button(self.ACTIONS_FRAME, text="QUIT", height="3", width="10", command=self.exit_game, fg="red")
        self.QUIT.pack(side="left")

        self.PLAY = tk.Button(self.ACTIONS_FRAME, text="PLAY", height="3", width="10", command=self.play_game)
        self.PLAY.pack(side="left")

        self.TRAIN = tk.Button(self.ACTIONS_FRAME, text="TRAIN", height="3", width="10", command=self.train_model)
        self.TRAIN.pack(side="left")

        self.CANNY = tk.Button(self.ACTIONS_FRAME, text="SHOW PREP", height="3", width="13", command=self.toggle_canny)
        self.CANNY.pack(side="left")

        self.LOAD = tk.Button(self.ACTIONS_FRAME, text="LOAD", height="3", width="15", command=self.load_model)
        self.LOAD.pack(side="right")

        self.LOAD_LOC_FRAME = tk.Frame(self.ACTIONS_FRAME)
        self.LOAD_LOC_FRAME.pack(side="right")

        self.LOAD_LOC_LABEL = tk.Label(self.LOAD_LOC_FRAME, text="Load Trained Model From File:")
        self.LOAD_LOC_LABEL.pack(side="top")

        self.LOAD_LOC = tk.Entry(self.LOAD_LOC_FRAME, width="30")
        self.LOAD_LOC.pack(side="bottom")

        self.PREDICATION_FRAME = tk.Frame(self.ACTIONS_FRAME)
        self.PREDICATION_FRAME.place(relx=.5, rely=.5, anchor="c")

        self.PREDICTION = tk.Label(self.PREDICATION_FRAME, text="PREDICTION")
        self.PREDICTION.pack(side="bottom")

    def update_background(self, cv_image):
        image = ImageTk.PhotoImage(Image.fromarray(cv_image))
        self.IMAGE.image = image
        self.IMAGE.configure(image=image)

    def update_background_from_video(self):
        image = self.video_source.read()
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.show_prep_image:
            image = self.model.prep_image(image)
            image = cv2.resize(image, (self.width, self.height))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #cv2.addWeighted(image, 1, prep_image, 1, 0, image)
        self.update_background(image)

    def update_components(self):
        if self.state == "liveimage":
            self.update_background_from_video()
            if self.play_time > 0:
                if self.play_time + 3 < time.time():
                    self.play_time = -1
                    self.update_prediction_text("Shoot")
                    self.detect_and_display()
                elif self.play_time + 2 < time.time():
                    self.update_prediction_text("1")
                elif self.play_time + 1 < time.time():
                    self.update_prediction_text("2")
                elif self.play_time < time.time():
                    self.update_prediction_text("3")
            elif self.prediction_time + 3.0 < time.time():
                self.update_prediction_text("")
        elif self.state == "training":
            t = int((time.time() - self.thread_start_time) * 2) % 20
            text = "TRAINING\n"
            for i in range(20):
                if t == i:
                    text += ">"
                else:
                    text += "="
            self.update_prediction_text(text)
            if self.show_prep_image and len(self.model.images) > 0:
                image = self.model.images[self.image_index % len(self.model.images)]
                image = cv2.resize(image, (self.width, self.height))
                self.update_background(image)
                self.image_index += 1

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
        self.train_thread = None
        self.thread_start_time = -1
        self.show_prep_image = False
        # self.images = []
        # self.load_images()
        self.image_index = 0