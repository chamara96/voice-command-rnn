import sys
from tkinter import filedialog
from tkinter import messagebox
import os
import time
import threading

from record_audio import AudioRecorder

import dataset_handling
import training_window

recorder = AudioRecorder()

import main  # automatically model loaded

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

# import GUI_support
model_filepath = None


# =======================
def set_Tk_var():
    global combobox
    combobox = tk.StringVar()


def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

    directory = 'training_data'
    clz_list = [x[1] for x in os.walk(directory)]
    w.TCombobox_classes['values'] = clz_list[0]
    w.TCombobox_classes.current(0)


xy_gui = threading.Thread(target=dataset_handling.start_model)

x_update_tread = threading.Thread(target=training_window.update_thread)


def btn_createmodel():
    xy_gui.start()
    destroy_window()
    # w.state=tk.DISABLED
    # dataset_handling.start_model()
    x_update_tread.start()
    training_window.vp_start_gui()

    # print('Main_2_support.btn_createmodel')
    # sys.stdout.flush()


def btn_exit():
    destroy_window()
    sys.stdout.flush()
    sys.exit()


def btn_open():
    filename = filedialog.askopenfilename()
    head_tail = os.path.split(filename)
    global model_filepath
    model_filepath = head_tail[1]
    print(filename)
    print(head_tail[1])
    if filename and head_tail[1]:
        model_path = "model/" + head_tail[1]
        main.load_model(model_path)
        print("New model Loaded")
        w.Label_modelpath['text'] = "model/" + head_tail[1]
    else:
        w.Label_modelpath['text'] = "Default model loaded"
    sys.stdout.flush()


def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


# =======================

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()

    root.bind('<Key-t>', press_key)
    root.bind('<KeyRelease-t>', release_key)

    root.bind('<Key-r>', press_key)
    root.bind('<KeyRelease-r>', release_key)

    set_Tk_var()
    top = Toplevel1(root)

    init(root, top)
    root.mainloop()


w = None


def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    # rt = root
    root = rt
    w = tk.Toplevel(root)

    set_Tk_var()
    top = Toplevel1(w)
    init(w, top, *args, **kwargs)
    return (w, top)


def destroy_Toplevel1():
    global w
    w.destroy()
    w = None


def create_bar(cur_val, max_val, n_symbols=21):
    ratio = min(1.0, abs(cur_val) / max_val)
    n1 = int(n_symbols * ratio)
    n2 = n_symbols - n1
    return "=" * n1 + " " * n2


t_prev_pressed = -999.0
key_effective_duration = 0.25
time_count = 0.00


def press_key(event):
    global t_prev_pressed, key_effective_duration, time_count

    if time_count == 0:
        if event.char == 't':
            f_testing_record()
        elif event.char == 'r':
            f_training_record()
        else:
            print("Invalid")

    if time.time() - t_prev_pressed < key_effective_duration:
        pass
    else:
        time_count += key_effective_duration
        t_prev_pressed = time.time()
        bar1 = create_bar(time_count, 2.1)
        if event.char == 't':
            w.Label_test_sound['text'] = bar1
        elif event.char == 'r':
            w.Label_record_sound['text'] = bar1


def f_testing_record():
    print("Start Recoder[Testing]")
    recorder.start_record(folder=main.DST_AUDIO_FOLDER)


def f_training_record():
    selected_clz = w.TCombobox_classes.get()
    recorder.start_record(folder='training_data/' + selected_clz + '/')
    print("Start Recoder[Training]", selected_clz)


def release_key(event):
    global time_count
    time_length = recorder.stop_record()
    print("Stop Recoder")
    time_count = 0

    if event.char == 't':
        w.Label_test_sound['text'] = ''
        final, val = main.voice_command(recorder.filename)
        print(final, val)
        w.Label_predict_prob['text'] = str(round(val * 100, 1)) + "%"
        w.Label_final_predict['text'] = final
        w.Label1_voice_length['text'] = str(round(time_length, 2)) + "s"
        # print("No of Threads - ", threading.activeCount())

    elif event.char == 'r':
        w.Label_record_sound['text'] = ''
        messagebox.showinfo("Saved",
                            "Saved new voice class: '" + w.TCombobox_classes.get() + "' in location:'" + recorder.filename + "'")
        # print("After msg box")

    print("You released KEY:", event.char)


class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.configure('.', font="TkDefaultFont")
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])

        top.geometry("640x474+443+101")
        top.minsize(120, 1)
        top.maxsize(1370, 749)
        top.resizable(0, 0)
        top.title("REHA Boy v1.0 [Voice Commands]")
        top.configure(background="#d9d9d9")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(x=20, y=20, height=50, width=600)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#d9d9d9")

        self.Label1 = tk.Label(self.Frame1)
        self.Label1.place(x=100, y=10, height=30, width=400)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Segoe UI} -size 18 -weight bold")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Voice Command Traning Center''')

        self.Label13 = tk.Label(self.Frame1)
        self.Label13.place(x=490, y=20, height=21, width=94)
        self.Label13.configure(background="#d9d9d9")
        self.Label13.configure(disabledforeground="#a3a3a3")
        self.Label13.configure(foreground="#000000")
        self.Label13.configure(text='''REHA Boy v1.0''')

        self.Labelframe1 = tk.LabelFrame(top)
        self.Labelframe1.place(x=20, y=90, height=365, width=290)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(foreground="black")
        self.Labelframe1.configure(text='''Training Mode''')
        self.Labelframe1.configure(background="#d9d9d9")

        self.TCombobox_classes = ttk.Combobox(self.Labelframe1)
        self.TCombobox_classes.place(x=120, y=30, height=21, width=143
                                     , bordermode='ignore')
        self.TCombobox_classes.configure(textvariable=combobox)
        # self.TCombobox_classes.configure(takefocus="")
        self.TCombobox_classes.configure(state='readonly')

        self.TLabel1 = ttk.Label(self.Labelframe1)
        self.TLabel1.place(x=20, y=30, height=19, width=95, bordermode='ignore')
        self.TLabel1.configure(background="#d9d9d9")
        self.TLabel1.configure(foreground="#000000")
        self.TLabel1.configure(font="TkDefaultFont")
        self.TLabel1.configure(relief="flat")
        self.TLabel1.configure(anchor='w')
        self.TLabel1.configure(justify='left')
        self.TLabel1.configure(text='''Select the class''')

        self.Frame2 = tk.Frame(self.Labelframe1)
        self.Frame2.place(x=20, y=70, height=65, width=250, bordermode='ignore')
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#d9d9d9")

        self.Label2 = tk.Label(self.Frame2)
        self.Label2.place(x=20, y=10, height=21, width=204)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''Press & Hold "R" for Record a Voice''')

        self.Label_record_sound = tk.Label(self.Frame2)
        self.Label_record_sound.place(x=10, y=30, height=21, width=224)
        self.Label_record_sound.configure(background="#d9d9d9")
        self.Label_record_sound.configure(disabledforeground="#a3a3a3")
        self.Label_record_sound.configure(font="-family {Showcard Gothic} -size 12 -weight bold")
        self.Label_record_sound.configure(foreground="#000000")
        # self.Label_record_sound.configure(text='''=====================''')

        self.Button_createmodel = tk.Button(self.Labelframe1)
        self.Button_createmodel.place(x=20, y=150, height=24, width=250
                                      , bordermode='ignore')
        self.Button_createmodel.configure(activebackground="#ececec")
        self.Button_createmodel.configure(activeforeground="#000000")
        self.Button_createmodel.configure(background="#d9d9d9")
        self.Button_createmodel.configure(command=btn_createmodel)
        self.Button_createmodel.configure(disabledforeground="#a3a3a3")
        self.Button_createmodel.configure(foreground="#000000")
        self.Button_createmodel.configure(highlightbackground="#d9d9d9")
        self.Button_createmodel.configure(highlightcolor="black")
        self.Button_createmodel.configure(pady="0")
        self.Button_createmodel.configure(text='''Create Model''')

        self.Labelframe4 = tk.LabelFrame(self.Labelframe1)
        self.Labelframe4.place(x=20, y=190, height=155, width=250
                               , bordermode='ignore')
        self.Labelframe4.configure(relief='groove')
        self.Labelframe4.configure(foreground="black")
        self.Labelframe4.configure(text='''Terminal''')
        self.Labelframe4.configure(background="#d9d9d9")

        self.Message_terminal = tk.Message(self.Labelframe4)
        self.Message_terminal.place(x=10, y=30, height=113, width=230
                                    , bordermode='ignore')
        self.Message_terminal.configure(anchor='nw')
        self.Message_terminal.configure(background="#d9d9d9")
        self.Message_terminal.configure(foreground="#000000")
        self.Message_terminal.configure(highlightbackground="#d9d9d9")
        self.Message_terminal.configure(highlightcolor="black")
        self.Message_terminal.configure(text='''Message''')
        self.Message_terminal.configure(width=230)

        self.Labelframe2 = tk.LabelFrame(top)
        self.Labelframe2.place(x=330, y=90, height=325, width=290)
        self.Labelframe2.configure(relief='groove')
        self.Labelframe2.configure(foreground="black")
        self.Labelframe2.configure(text='''Testing Mode''')
        self.Labelframe2.configure(background="#d9d9d9")

        self.Frame3 = tk.Frame(self.Labelframe2)
        self.Frame3.place(x=20, y=90, height=65, width=250, bordermode='ignore')
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#d9d9d9")

        self.Label4 = tk.Label(self.Frame3)
        self.Label4.place(x=20, y=10, height=21, width=204)
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''Press & Hold "T" for Test a Class''')

        self.Label_test_sound = tk.Label(self.Frame3)
        self.Label_test_sound.place(x=20, y=30, height=21, width=224)
        self.Label_test_sound.configure(anchor='w')
        self.Label_test_sound.configure(background="#d9d9d9")
        self.Label_test_sound.configure(disabledforeground="#a3a3a3")
        self.Label_test_sound.configure(font="-family {Showcard Gothic} -size 12 -weight bold")
        self.Label_test_sound.configure(foreground="#000000")
        # self.Label_test_sound.configure(text='''=====================''')

        self.Label6 = tk.Label(self.Labelframe2)
        self.Label6.place(x=10, y=30, height=21, width=94, bordermode='ignore')
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(text='''Select Model''')

        self.Button_openpath = tk.Button(self.Labelframe2)
        self.Button_openpath.place(x=100, y=30, height=24, width=157
                                   , bordermode='ignore')
        self.Button_openpath.configure(activebackground="#ececec")
        self.Button_openpath.configure(activeforeground="#000000")
        self.Button_openpath.configure(background="#d9d9d9")
        self.Button_openpath.configure(command=btn_open)
        self.Button_openpath.configure(disabledforeground="#a3a3a3")
        self.Button_openpath.configure(foreground="#000000")
        self.Button_openpath.configure(highlightbackground="#d9d9d9")
        self.Button_openpath.configure(highlightcolor="black")
        self.Button_openpath.configure(pady="0")
        self.Button_openpath.configure(text='''Open''')

        self.Label_modelpath = tk.Label(self.Labelframe2)
        self.Label_modelpath.place(x=30, y=60, height=21, width=234
                                   , bordermode='ignore')
        self.Label_modelpath.configure(background="#d9d9d9")
        self.Label_modelpath.configure(disabledforeground="#a3a3a3")
        self.Label_modelpath.configure(foreground="#848480")
        # self.Label_modelpath.configure(text='''model_file_path''')

        self.Label_final_predict = tk.Label(self.Labelframe2)
        self.Label_final_predict.place(x=20, y=160, height=51, width=244
                                       , bordermode='ignore')
        self.Label_final_predict.configure(background="#d9d9d9")
        self.Label_final_predict.configure(disabledforeground="#a3a3a3")
        self.Label_final_predict.configure(font="-family {Segoe UI Black} -size 20 -weight bold")
        self.Label_final_predict.configure(foreground="#000000")
        self.Label_final_predict.configure(highlightbackground="#f0f0f0f0f0f0")
        # self.Label_final_predict.configure(text='''COMMAND''')

        self.Labelframe3 = tk.LabelFrame(self.Labelframe2)
        self.Labelframe3.place(x=70, y=220, height=85, width=150
                               , bordermode='ignore')
        self.Labelframe3.configure(relief='groove')
        self.Labelframe3.configure(foreground="black")
        self.Labelframe3.configure(text='''Details''')
        self.Labelframe3.configure(background="#d9d9d9")

        self.Label9 = tk.Label(self.Labelframe3)
        self.Label9.place(x=10, y=30, height=21, width=74, bordermode='ignore')
        self.Label9.configure(background="#d9d9d9")
        self.Label9.configure(disabledforeground="#a3a3a3")
        self.Label9.configure(foreground="#000000")
        self.Label9.configure(text='''Probability''')

        self.Label10 = tk.Label(self.Labelframe3)
        self.Label10.place(x=10, y=50, height=21, width=54, bordermode='ignore')
        self.Label10.configure(background="#d9d9d9")
        self.Label10.configure(disabledforeground="#a3a3a3")
        self.Label10.configure(foreground="#000000")
        self.Label10.configure(text='''Length''')

        self.Label_predict_prob = tk.Label(self.Labelframe3)
        self.Label_predict_prob.place(x=90, y=30, height=21, width=34
                                      , bordermode='ignore')
        self.Label_predict_prob.configure(background="#d9d9d9")
        self.Label_predict_prob.configure(disabledforeground="#a3a3a3")
        self.Label_predict_prob.configure(foreground="#000000")
        # self.Label_predict_prob.configure(text='''100%''')

        self.Label1_voice_length = tk.Label(self.Labelframe3)
        self.Label1_voice_length.place(x=90, y=50, height=21, width=32
                                       , bordermode='ignore')
        self.Label1_voice_length.configure(background="#d9d9d9")
        self.Label1_voice_length.configure(disabledforeground="#a3a3a3")
        self.Label1_voice_length.configure(foreground="#000000")
        # self.Label1_voice_length.configure(text='''1.80s''')

        self.Button_exit = tk.Button(top)
        self.Button_exit.place(x=330, y=430, height=24, width=290)
        self.Button_exit.configure(activebackground="#ececec")
        self.Button_exit.configure(activeforeground="#000000")
        self.Button_exit.configure(background="#d9d9d9")
        self.Button_exit.configure(command=btn_exit)
        self.Button_exit.configure(disabledforeground="#a3a3a3")
        self.Button_exit.configure(foreground="#000000")
        self.Button_exit.configure(highlightbackground="#d9d9d9")
        self.Button_exit.configure(highlightcolor="black")
        self.Button_exit.configure(pady="0")
        self.Button_exit.configure(text='''Exit''')


if __name__ == '__main__':
    vp_start_gui()
