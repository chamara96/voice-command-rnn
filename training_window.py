import sys
from PIL import ImageTk, Image
import time

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

from tkinter import messagebox
import neural_network
import dataset_handling

def update_thread():
    # global is_stop
    time.sleep(5)
    is_train_end = 0
    while not is_train_end:
        is_train_end = dataset_handling.end_train
        # text = ""
        w.Label_log.delete("1.0", tk.END)
        if is_stop == 1:
            break
        curr_epoch, total_epoches = neural_network.check_curr_epoch()
        w.TProgressbar1['value'] = int((curr_epoch) * 100 / total_epoches)
        filename = "checkpoints/log.txt"
        try:
            with open(filename) as f:
                text = f.read()
        except IOError:
            text = ""

        w.Label_log.insert("1.0", text)

        try:
            img = Image.open("checkpoints/fig.jpg")
        except IOError:
            img = Image.open("classes/wait.png")

        basewidth = 550
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        w.Label_plot['image'] = img
        w.Label_plot.image = img
        sys.stdout.flush()
        print("Updated")
        time.sleep(2)

    messagebox.showinfo("Training", "Done..!")


def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top


def btn_stop():
    destroy_window()
    sys.stdout.flush()

is_stop = 0

def btn_update_view():
    curr_epoch, total_epoches = neural_network.check_curr_epoch()
    w.TProgressbar1['value'] = int(curr_epoch * 100 / total_epoches)
    filename = "checkpoints/log.txt"
    try:
        with open(filename) as f:
            text = f.read()
    except IOError:
        text = ""

    w.Label_log.insert("1.0", text)

    try:
        img = Image.open("checkpoints/fig.jpg")
    except IOError:
        img = Image.open("classes/wait.png")

    basewidth = 550
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    w.Label_plot['image'] = img
    w.Label_plot.image = img
    sys.stdout.flush()
    print("Updated")
    # time.sleep(2)

def destroy_window():
    global is_stop
    is_stop=1
    print("QQWWEERR")

    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None
    sys.exit()


def vp_start_gui():

    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1245x656+220+79")
        top.minsize(120, 1)
        top.maxsize(2650, 1005)
        top.resizable(0,  0)
        top.title("Training Model")
        top.configure(background="#d9d9d9")

        self.Labelframe1 = tk.LabelFrame(top)
        self.Labelframe1.place(x=20, y=40, height=600, width=600)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(foreground="black")
        self.Labelframe1.configure(text='''Log''')
        self.Labelframe1.configure(background="#d9d9d9")

        self.Label_log=tk.Text(self.Labelframe1)

        # self.Label_log = tk.Label(self.Labelframe1)
        self.Label_log.place(x=20, y=30, height=551, width=564
                , bordermode='ignore')
        # self.Label_log.configure(anchor='nw')
        self.Label_log.configure(background="#d9d9d9")
        # self.Label_log.configure(disabledforeground="#a3a3a3")
        self.Label_log.configure(foreground="#000000")
        # self.Label_log.configure(text='''Label''')

        self.Labelframe2 = tk.LabelFrame(top)
        self.Labelframe2.place(x=630, y=40, height=600, width=600)
        self.Labelframe2.configure(relief='groove')
        self.Labelframe2.configure(foreground="black")
        self.Labelframe2.configure(text='''Training Curves''')
        self.Labelframe2.configure(background="#d9d9d9")

        self.Label_plot = tk.Label(self.Labelframe2)
        self.Label_plot.place(x=20, y=30, height=551, width=554
                , bordermode='ignore')
        self.Label_plot.configure(anchor='nw')
        self.Label_plot.configure(background="#d9d9d9")
        self.Label_plot.configure(disabledforeground="#a3a3a3")
        self.Label_plot.configure(foreground="#000000")
        self.Label_plot.configure(text='''Label''')

        self.Button1 = tk.Button(top)
        self.Button1.place(x=1100, y=10, height=34, width=127)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(command=btn_stop)
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Stop''')

        self.TProgressbar1 = ttk.Progressbar(top)
        self.TProgressbar1.place(x=20, y=10, width=600, height=22)
        self.TProgressbar1.configure(length="600")
        self.TProgressbar1.configure(value="10")

        # self.Button2 = tk.Button(top)
        # self.Button2.place(x=980, y=10, height=34, width=117)
        # self.Button2.configure(activebackground="#ececec")
        # self.Button2.configure(activeforeground="#000000")
        # self.Button2.configure(background="#d9d9d9")
        # self.Button2.configure(command=btn_update_view)
        # self.Button2.configure(disabledforeground="#a3a3a3")
        # self.Button2.configure(foreground="#000000")
        # self.Button2.configure(highlightbackground="#d9d9d9")
        # self.Button2.configure(highlightcolor="black")
        # self.Button2.configure(pady="0")
        # self.Button2.configure(text='''Update View''')

if __name__ == '__main__':
    vp_start_gui()





