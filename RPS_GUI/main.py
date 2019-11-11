from gui import GUI
from model import MODEL
from imutils.video import VideoStream
import tkinter as tk

WIDTH = 1080
HEIGHT = 720
XPOS = 300
YPOS = 100

root = tk.Tk()
root.title("Rock Paper Scissors")
geom = "{}x{}+{}+{}".format(WIDTH, HEIGHT, XPOS, YPOS)
root.geometry(geom)
video_source = VideoStream(src=0).start()
model = MODEL()

app = GUI(master=root, model=model, video_source=video_source, width=width, height=height)

while app.state != "closed":
    app.update_components()
    app.update_idletasks()
    app.update()

video_source.stop()
root.destroy()