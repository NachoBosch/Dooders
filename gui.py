import os
import tkinter as tk
from tkinter import *
from tkinter import font
import tensorflow as tf
import PIL
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def main():
    def function_classifier():
        def predict(model):
            image_dir = txt.get("1.0", "end-1c")
            image = np.array(PIL.Image.open(image_dir))
            image = tf.image.resize(image, size=(180, 180))
            image = image/255.
            pred = tf.squeeze(model.predict(tf.expand_dims(image,axis=0))).numpy()
            classes = ['men','women']
            if pred<0.5:
                var.set(classes[0])
            else:
                var.set(classes[1])
            print("Pred:",pred)
            print("Class: ",classes[tf.cast(tf.round(pred),dtype=tf.int32).numpy()])
            print(image.shape)

        model_gender = tf.keras.models.load_model('save_dir_logs/model_gender')
        buton = tk.Button(frame2, text="Predecir", width=20, background="#424242", foreground="#FFFFFF",
                          justify=tk.LEFT, activebackground='#FFFFFF', highlightcolor='#6d6d6d', highlightthickness=1,
                          relief=RAISED, state=NORMAL, font=Lightfont, command=lambda:predict(model_gender))
        buton.grid(row=0,column=0,sticky='nw')
        txt = tk.Text(frame2, width=70, height=1)
        txt.grid(column=0, row=1, sticky='sw', padx=5,pady=10)
        var = StringVar()
        var.set('Label')
        label = tk.Label(frame2,textvariable=var,background='purple',foreground='white',font=Lightfont,width=10, height=1)
        label.grid(column=1,row=2,sticky='se',padx=5,pady=10)

    # .------------------------------.#
    # .        -Main Window-         .#
    ventana = tk.Tk()
    ventana.title("Dooders")
    ventana.configure(background='#424242')
    ventana.state('zoomed')
    ventana.grid_rowconfigure(3, weight=1)
    ventana.grid_columnconfigure(5, weight=1)

    # .-----------------------------.#
    # .      -Frames-    .#
    frame = Frame(ventana)
    frame.grid(row=0, sticky=E + W + N + S)
    frame.config(relief="sunken", bg='#424242')
    frame3 = Frame(ventana, bd=2)
    frame3.grid(row=1, column=0, rowspan=20, sticky="nsew")
    frame3.config(relief="sunken", bg='#424242')
    frame2 = Frame(ventana, bd=2)
    frame2.grid(row=1, column=1, rowspan=20, sticky="nsew")
    frame2.config(relief="flat", bg='#424242')

    # .--------.#
    # .---------------------------.#
    # .          -Head-            .#
    Light = font.Font(family="Corbel Light", size=25, weight="normal")
    Lightfont = font.Font(family="Corbel Light", size=20, weight="normal")

    label1 = Label(frame, text="Smart-Dood", background='#424242', foreground="#FFFFFF", font=Light)
    label1.grid(column=1, row=0)
    # .--------------------------.#
    # .			-Buttons-		.#
    buton = tk.Button(frame3, text="Clasificar", width=20, background="#424242", foreground="#FFFFFF",
                      justify=tk.LEFT, activebackground='#FFFFFF', highlightcolor='#6d6d6d', highlightthickness=1,
                      relief=RAISED, state=NORMAL, font=Lightfont, command=function_classifier)
    buton.grid()
    buton.config(state=tk.NORMAL)
    #buton.bind("<Button-1>", clear)
    # ---Toolbar---#
    menubar = Menu(ventana)
    # Commands
    file = Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Inicio', menu=file)
    file.add_command(label='Salir', command=ventana.destroy)
    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Manual", command=lambda: su.read_manual())
    menubar.add_cascade(label="Ayuda", menu=helpmenu)
    ventana.config(menu=menubar)
    ventana.mainloop()


if __name__ == '__main__':
    main()
