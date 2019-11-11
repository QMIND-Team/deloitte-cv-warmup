from tkinter import *
from PIL import Image, ImageTk
import cv2
from imutils.video import VideoStream

class Application(Frame):
    def play_game(self):
        print("Let's play a game...")
        self.update_background("s")
    
    def exit_game(self):
        self.open = False

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

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.open = True
        self.pack(side="top", fill="both", expand=1)
        self.createWidgets()

def live_detection():
    root = Tk()
    root.geometry("1080x720+300+100")
    app = Application(master=root)
    video_source = VideoStream(src=0).start()

    while app.open:
        image = video_source.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        app.update_background_opencv(image)
        app.update_idletasks()
        app.update()
    root.destroy()

live_detection()