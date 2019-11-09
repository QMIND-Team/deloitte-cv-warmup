from tkinter import *
from PIL import Image, ImageTk

class Application(Frame):
    def play_game(self):
        print("Let's play a game...")
        self.update_background("s")

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
        self.QUIT["command"] =  self.quit
        self.QUIT.pack(side="left")

        self.PLAY = Button(self.BUTTONS)
        self.PLAY["text"] = "PLAY",
        self.PLAY["height"]   = "3"
        self.PLAY["width"]   = "15"
        self.PLAY["command"] = self.play_game
        self.PLAY.pack(side="left")

    def update_background(self, image_name):
        image2 = Image.open("rockpaperscissors/scissors/0zoQAmDFXehOZsAp.png")
        image = ImageTk.PhotoImage(image2)
        self.IMAGE.image = image
        self.IMAGE.configure(image=image)

    def __init__(self, master=None):
        Frame.__init__(self, master, background="red", borderwidth=5)
        #self.resizable(width=False, height=False)
        #container = master.Frame(self)
        #container.pack(side="top", fill="both", expand=1)
        self.pack(side="top", fill="both", expand=1)
        self.createWidgets()

root = Tk()
root.geometry("1080x720+300+100")
app = Application(master=root)
app.mainloop()
root.destroy()