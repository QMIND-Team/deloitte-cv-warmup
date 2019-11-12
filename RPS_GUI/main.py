from gui import GUI
from model import MODEL
from imutils.video import VideoStream
import tkinter as tk
import sys, getopt

def main(argv):
    WIDTH = 1080
    HEIGHT = 720
    XPOS = 300
    YPOS = 100
    USER = "CONNOR"

    filename=None
    try:
        opts, args = getopt.getopt(argv,"hu:",["user="])
    except getopt.GetoptError:
        print("INCORRECT FORMAT: \"main.py -u <user>\"")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("main.py -u <user>")
            sys.exit()
        elif opt in ("-u", "--user"):
            USER=arg

    root = tk.Tk()
    root.title("Rock Paper Scissors")
    geom = "{}x{}+{}+{}".format(WIDTH, HEIGHT, XPOS, YPOS)
    root.geometry(geom)
    video_source = VideoStream(src=0).start()
    model = MODEL(USER)

    app = GUI(master=root, model=model, video_source=video_source, width=WIDTH, height=HEIGHT)

    while app.state != "closed":
        app.update_components()
        app.update_idletasks()
        app.update()

    video_source.stop()
    root.destroy()

if __name__ == '__main__':
	main(sys.argv[1:])