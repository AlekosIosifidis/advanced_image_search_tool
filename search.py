import tkinter as tk
import tkinter.font as tkFont
import pickle
import dlib
from shutil import copyfile
import glob
import os
import cv2
from tkinter import Tk    # from tkinter import Tk for Python 3.x
from tkinter import Canvas
from tkinter.filedialog import askopenfilename
from tkcalendar import Calendar,DateEntry
import numpy as np
import imgsim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import tkinter.ttk
from google_trans_new import google_translator
import win32gui, win32con, time
from tkinter.filedialog import askdirectory

class App:

    def __init__(self, root):
        global face_filename
        face_filename = ''

        global face_selected
        face_selected = False

        global similar_image_chosen
        similar_image_chosen = False

        global ObjectNames
        with open('coco.names.txt', 'rt') as f:
            ObjectNames = f.read().rstrip('\n').split('\n')

        global SceneNames
        with open('categories_places365.txt', 'rt') as f:
            tempSceneNames = f.read().rstrip('\n').split('\n')
            SceneNames = []
            for tempSceneName in tempSceneNames:
                sceneName = tempSceneName[3:].split(' ')[0]
                SceneNames.append(sceneName)

        def MoneyValidation(S):
            if S in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or len(S)>1:
                return True
            root.bell()  # .bell() plays that ding sound telling you there was invalid input
            return False
        def startdate_validation(e):
            if startdate.get_date() > enddate.get_date():
                startdate.set_date(enddate.get_date())
            return True
        def enddate_validation(e):
            if startdate.get_date() > enddate.get_date():
                enddate.set_date(startdate.get_date())
            return True
        #setting title
        root.title("Advanced Image Search Tool")
        #setting window size
        width=1150
        height=780
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        separator = tk.ttk.Separator(root, orient='vertical')
        separator.place(relx=0.88, rely=0.06, relwidth=0.2, relheight=0.56)
        separator2 = tk.ttk.Separator(root, orient='horizontal')
        separator2.place(relx=0.86, rely=0.06, relwidth=0.02, relheight=0.2)
        separator3 = tk.ttk.Separator(root, orient='horizontal')
        separator3.place(relx=0.86, rely=0.62, relwidth=0.02, relheight=0.2)

        GButton_290=tk.Button(root)
        GButton_290["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_290["font"] = ft
        GButton_290["fg"] = "#000000"
        GButton_290["justify"] = "center"
        GButton_290["text"] = "Choose file"
        GButton_290.place(x=270,y=130,width=90,height=35)
        GButton_290["command"] = self.GButton_290_command

        GButton_558=tk.Button(root)
        GButton_558["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_558["font"] = ft
        GButton_558["fg"] = "#000000"
        GButton_558["justify"] = "center"
        GButton_558["text"] = "Choose Folder"
        GButton_558.place(x=35,y=30,width=90,height=25)
        GButton_558["command"] = self.GButton_558_command

        global Directory
        Directory = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        Directory["font"] = ft
        Directory["fg"] = "#29872d"
        Directory["justify"] = "left"
        Directory["text"] = ""
        Directory.place(x=125, y=30, width=149, height=25)

        GLabel_567=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GLabel_567["font"] = ft
        GLabel_567["fg"] = "#333333"
        GLabel_567["justify"] = "right"
        GLabel_567["text"] = "Search similar images"
        GLabel_567.place(x=90,y=130,width=170,height=30)

        GLabel_71=tk.Label(root)
        GLabel_71["activebackground"] = "#ee2f2f"
        ft = tkFont.Font(family='Times',size=16)
        GLabel_71["font"] = ft
        GLabel_71["fg"] = "#333333"
        GLabel_71["justify"] = "center"
        GLabel_71["text"] = "Advanced Image Search Tool"
        GLabel_71.place(x=400,y=20,width=274,height=30)

        GLabel_413=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GLabel_413["font"] = ft
        GLabel_413["fg"] = "#333333"
        GLabel_413["justify"] = "right"
        GLabel_413["text"] = "Search by face"
        GLabel_413.place(x=90,y=190,width=170,height=30)

        GButton_158=tk.Button(root)
        GButton_158["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_158["font"] = ft
        GButton_158["fg"] = "#000000"
        GButton_158["justify"] = "center"
        GButton_158["text"] = "Choose file"
        GButton_158.place(x=270,y=190,width=90,height=35)
        GButton_158["command"] = self.GButton_158_command

        GLabel_308=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GLabel_308["font"] = ft
        GLabel_308["fg"] = "#333333"
        GLabel_308["justify"] = "right"
        GLabel_308["text"] = "Search by shot type"
        GLabel_308.place(x=90,y=250,width=170,height=30)

        global var_close
        var_close = tk.IntVar()
        global GCheckBox_962
        GCheckBox_962=tk.Checkbutton(root, variable = var_close)
        ft = tkFont.Font(family='Times',size=10)
        GCheckBox_962["font"] = ft
        GCheckBox_962["fg"] = "#333333"
        GCheckBox_962["justify"] = "left"
        GCheckBox_962["text"] = "Close"
        GCheckBox_962.place(x=270,y=240,width=70,height=25)
        GCheckBox_962["offvalue"] = "0"
        GCheckBox_962["onvalue"] = "1"
        GCheckBox_962["command"] = self.GCheckBox_962_command

        global var_medium
        var_medium = tk.IntVar()
        global GCheckBox_134
        GCheckBox_134=tk.Checkbutton(root, variable = var_medium)
        ft = tkFont.Font(family='Times',size=10)
        GCheckBox_134["font"] = ft
        GCheckBox_134["fg"] = "#333333"
        GCheckBox_134["justify"] = "left"
        GCheckBox_134["text"] = "Medium"
        GCheckBox_134.place(x=277,y=260,width=70,height=25)
        GCheckBox_134["offvalue"] = "0"
        GCheckBox_134["onvalue"] = "1"
        GCheckBox_134["command"] = self.GCheckBox_134_command

        global var_long
        var_long = tk.IntVar()
        global GCheckBox_934
        GCheckBox_934=tk.Checkbutton(root, variable = var_long)
        ft = tkFont.Font(family='Times',size=10)
        GCheckBox_934["font"] = ft
        GCheckBox_934["fg"] = "#333333"
        GCheckBox_934["justify"] = "left"
        GCheckBox_934["text"] = "Long"
        GCheckBox_934.place(x=270,y=280,width=70,height=25)
        GCheckBox_934["offvalue"] = "0"
        GCheckBox_934["onvalue"] = "1"
        GCheckBox_934["command"] = self.GCheckBox_934_command

        global GMessage_276
        GMessage_276=tk.Message(root)
        ft = tkFont.Font(family='Times',size=14)
        GMessage_276["font"] = ft
        GMessage_276["fg"] = "#378132"
        GMessage_276["justify"] = "center"
        GMessage_276["text"] = ""
        GMessage_276.place(x=370,y=130,width=100,height=30)

        GLabel_313 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        GLabel_313["font"] = ft
        GLabel_313["fg"] = "#333333"
        GLabel_313["justify"] = "right"
        GLabel_313["text"] = "Search by environment type"
        GLabel_313.place(x=50, y=320, width=215, height=30)

        global var_indoor
        var_indoor = tk.IntVar()
        global GCheckBox_26
        GCheckBox_26 = tk.Checkbutton(root, variable = var_indoor)
        ft = tkFont.Font(family='Times', size=10)
        GCheckBox_26["font"] = ft
        GCheckBox_26["fg"] = "#333333"
        GCheckBox_26["justify"] = "left"
        GCheckBox_26["text"] = "Indoor"
        GCheckBox_26.place(x=270, y=320, width=70, height=25)
        GCheckBox_26["offvalue"] = "0"
        GCheckBox_26["onvalue"] = "1"
        GCheckBox_26["command"] = self.GCheckBox_26_command

        global var_outdoor
        var_outdoor = tk.IntVar()
        global GCheckBox_588
        GCheckBox_588 = tk.Checkbutton(root, variable=var_outdoor)
        ft = tkFont.Font(family='Times', size=10)
        GCheckBox_588["font"] = ft
        GCheckBox_588["fg"] = "#333333"
        GCheckBox_588["justify"] = "left"
        GCheckBox_588["text"] = "Outdoor"
        GCheckBox_588.place(x=275, y=340, width=70, height=25)
        GCheckBox_588["offvalue"] = "0"
        GCheckBox_588["onvalue"] = "1"
        GCheckBox_588["command"] = self.GCheckBox_588_command

        global description_var
        description_var = tk.StringVar()
        global GLineEdit_549
        GLineEdit_549 = tk.Entry(root, textvariable = description_var)
        GLineEdit_549["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        GLineEdit_549["font"] = ft
        GLineEdit_549["fg"] = "#333333"
        GLineEdit_549["justify"] = "center"
        GLineEdit_549["text"] = "Entry"
        GLineEdit_549.place(x=270, y=80, width=348, height=30)

        GLabel_610 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        GLabel_610["font"] = ft
        GLabel_610["fg"] = "#333333"
        GLabel_610["justify"] = "right"
        GLabel_610["text"] = "Content based search"
        GLabel_610.place(x=70, y=80, width=196, height=30)

        GLabel_158 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        GLabel_158["font"] = ft
        GLabel_158["fg"] = "#333333"
        GLabel_158["justify"] = "right"
        GLabel_158["text"] = "Search by number of people"
        GLabel_158.place(x=30, y=380, width=239, height=30)

        global num_people_var
        vcmd = (root.register(MoneyValidation), '%S')
        num_people_var = tk.StringVar()
        global GLineEdit_14
        GLineEdit_14 = tk.Entry(root, textvariable=num_people_var, validate='key', vcmd=vcmd)
        GLineEdit_14["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        GLineEdit_14["font"] = ft
        GLineEdit_14["fg"] = "#333333"
        GLineEdit_14["justify"] = "center"
        GLineEdit_14["text"] = "Entry2"
        GLineEdit_14.place(x=270, y=380, width=70, height=25)

        GButton_851 = tk.Button(root)
        GButton_851["activeforeground"] = "#f82b2b"
        GButton_851["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        GButton_851["font"] = ft
        GButton_851["fg"] = "#000000"
        GButton_851["justify"] = "center"
        GButton_851["text"] = "CLEAR"
        GButton_851.place(x=440, y=720, width=70, height=45)
        GButton_851["command"] = self.GButton_851_command

        GButton_227 = tk.Button(root)
        GButton_227["activeforeground"] = "#499d46"
        GButton_227["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        GButton_227["font"] = ft
        GButton_227["fg"] = "#000000"
        GButton_227["justify"] = "center"
        GButton_227["text"] = "SEARCH"
        GButton_227.place(x=570, y=720, width=70, height=45)
        GButton_227["command"] = self.GButton_227_command

        ObjectListClear = tk.Button(root)
        ObjectListClear["activeforeground"] = "#f82b2b"
        ObjectListClear["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        ObjectListClear["font"] = ft
        ObjectListClear["fg"] = "#000000"
        ObjectListClear["justify"] = "center"
        ObjectListClear["text"] = "CLEAR"
        ObjectListClear.place(x=235, y=688, width=60, height=20)
        ObjectListClear["command"] = self.ObjectListClear_command

        SceneListClear = tk.Button(root)
        SceneListClear["activeforeground"] = "#f82b2b"
        SceneListClear["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        SceneListClear["font"] = ft
        SceneListClear["fg"] = "#000000"
        SceneListClear["justify"] = "center"
        SceneListClear["text"] = "CLEAR"
        SceneListClear.place(x=575, y=688, width=60, height=20)
        SceneListClear["command"] = self.SceneListClear_command

        global GLabel_476
        GLabel_476 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        GLabel_476["font"] = ft
        GLabel_476["fg"] = "#29872d"
        GLabel_476["justify"] = "left"
        GLabel_476["text"] = ""
        GLabel_476.place(x=370, y=190, width=100, height=30)

        global startdate
        global enddate

        startdate = DateEntry(root,width=30,bg="darkblue",fg="white", date_pattern='dd/MM/yyyy')
        startdate.place(x=320, y=430, width=80, height=30)
        startdate.bind("<<DateEntrySelected>>", startdate_validation)
        startdate.delete(0, "end")

        enddate = DateEntry(root,width=30,bg="darkblue",fg="white", date_pattern='dd/MM/yyyy')
        enddate.place(x=320, y=470, width=80, height=30)
        enddate.bind("<<DateEntrySelected>>", enddate_validation)
        enddate.delete(0, "end")

        DateText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        DateText["font"] = ft
        DateText["fg"] = "#333333"
        DateText["justify"] = "right"
        DateText["text"] = "Search by date"
        DateText.place(x=130, y=450, width=120, height=30)

        DimensionText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        DimensionText["font"] = ft
        DimensionText["fg"] = "#333333"
        DimensionText["justify"] = "right"
        DimensionText["text"] = "Search by size"
        DimensionText.place(x=120, y=540, width=120, height=30)

        DimensionWidthText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        DimensionWidthText["font"] = ft
        DimensionWidthText["fg"] = "#333333"
        DimensionWidthText["justify"] = "right"
        DimensionWidthText["text"] = "Width"
        DimensionWidthText.place(x=260, y=520, width=60, height=30)

        global WidthText
        WidthText = tk.Entry(root, validate='key', vcmd=vcmd)
        WidthText["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        WidthText["font"] = ft
        WidthText["fg"] = "#333333"
        WidthText["justify"] = "center"
        WidthText["text"] = "Entry3"
        WidthText.place(x=320, y=525, width=70, height=25)

        DimensionHeightText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        DimensionHeightText["font"] = ft
        DimensionHeightText["fg"] = "#333333"
        DimensionHeightText["justify"] = "right"
        DimensionHeightText["text"] = "Height"
        DimensionHeightText.place(x=260, y=560, width=60, height=30)

        global HeightText
        HeightText = tk.Entry(root, validate='key', vcmd=vcmd)
        HeightText["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        HeightText["font"] = ft
        HeightText["fg"] = "#333333"
        HeightText["justify"] = "center"
        HeightText["text"] = "Entry4"
        HeightText.place(x=320, y=565, width=70, height=25)

        global var_smallerthan
        var_smallerthan = tk.IntVar()
        global SmallerThan
        SmallerThan=tk.Checkbutton(root, variable=var_smallerthan)
        ft = tkFont.Font(family='Times',size=10)
        SmallerThan["font"] = ft
        SmallerThan["fg"] = "#333333"
        SmallerThan["justify"] = "left"
        SmallerThan["text"] = "Smaller than"
        SmallerThan.place(x=400,y=525,width=100,height=25)
        SmallerThan["offvalue"] = "0"
        SmallerThan["onvalue"] = "1"
        SmallerThan["command"] = self.SmallerThan_command

        global var_exactly
        var_exactly = tk.IntVar()
        global Exactly
        Exactly=tk.Checkbutton(root, variable=var_exactly)
        ft = tkFont.Font(family='Times',size=10)
        Exactly["font"] = ft
        Exactly["fg"] = "#333333"
        Exactly["justify"] = "left"
        Exactly["text"] = "Exactly"
        Exactly.place(x=396,y=545,width=80,height=25)
        Exactly["offvalue"] = "0"
        Exactly["onvalue"] = "1"
        Exactly["command"] = self.Exactly_command

        global var_largerthan
        var_largerthan = tk.IntVar()
        global LargerThan
        LargerThan=tk.Checkbutton(root, variable=var_largerthan)
        ft = tkFont.Font(family='Times',size=10)
        LargerThan["font"] = ft
        LargerThan["fg"] = "#333333"
        LargerThan["justify"] = "left"
        LargerThan["text"] = "Larger than"
        LargerThan.place(x=407,y=565,width=80,height=25)
        LargerThan["offvalue"] = "0"
        LargerThan["onvalue"] = "1"
        LargerThan["command"] = self.LargerThan_command

        DateFromText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        DateFromText["font"] = ft
        DateFromText["fg"] = "#333333"
        DateFromText["justify"] = "right"
        DateFromText["text"] = "From"
        DateFromText.place(x=260, y=430, width=60, height=30)

        DateToText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        DateToText["font"] = ft
        DateToText["fg"] = "#333333"
        DateToText["justify"] = "right"
        DateToText["text"] = "To"
        DateToText.place(x=260, y=470, width=60, height=30)

        ObjectText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        ObjectText["font"] = ft
        ObjectText["fg"] = "#333333"
        ObjectText["justify"] = "right"
        ObjectText["text"] = "Search by object"
        ObjectText.place(x=5, y=620, width=140, height=30)

        global sel
        sel=list()

        def CurSelect(evt):
            global sel
            temp=list()
            for i in ObjectsListBox.curselection():
                temp.append(ObjectsListBox.get(i))

            allitems=list()
            for i in range(ObjectsListBox.size()):
                allitems.append(ObjectsListBox.get(i))

            for i in sel:
                if i in allitems:
                    if i not in temp:
                        sel.remove(i)

            for x in ObjectsListBox.curselection():
                if ObjectsListBox.get(x) not in sel:
                    sel.append(ObjectsListBox.get(x))

        def update_list():
            global sel
            global l
            search_term = search_var.get()

            # Just a generic list to populate the listbox
            lbox_list = ObjectNames

            ObjectsListBox.delete(0, 'end')

            for item in lbox_list:
                if search_term.lower() in item.lower():
                    ObjectsListBox.insert('end', item)

            allitems=list()
            for i in range(ObjectsListBox.size()):
                allitems.append(ObjectsListBox.get(i))

            for i in sel:
                if i in allitems:
                    ObjectsListBox.select_set(ObjectsListBox.get(0, "end").index(i))

        search_var = tk.StringVar()
        search_var.trace("w", lambda name, index, mode: update_list())

        global entry
        entry = tk.Entry(root, textvariable=search_var, width=13)
        entry.place(x=150, y=625, width=60, height=20)

        global ObjectsListBox
        ObjectsListBox = tk.Listbox(root, selectmode='multiple',width=45, height=15)
        ObjectsListBox.bind('<<ListboxSelect>>',CurSelect)
        #entry.grid(row=0, column=0, padx=10, pady=3)
        ObjectsListBox.grid(row=1, column=0, padx=10, pady=3)
        cursel=tk.StringVar()
        lb1=tk.Label(root,textvariable=cursel)
        lb1.grid(row=100,column=0,padx=10,pady=3)
        ObjectsListBox.place(x=225, y=610, width=80, height=75)

        # Function for updating the list/doing the search.
        # It needs to be called here to populate the listbox.
        update_list()

        global sel_scene
        sel_scene=list()

        def CurSelect_scene(evt):
            global sel_scene
            temp_scene=list()
            for i in ScenesListBox.curselection():
                temp_scene.append(ScenesListBox.get(i))

            allitems_scene=list()
            for i in range(ScenesListBox.size()):
                allitems_scene.append(ScenesListBox.get(i))

            for i in sel_scene:
                if i in allitems_scene:
                    if i not in temp_scene:
                        sel_scene.remove(i)

            for x in ScenesListBox.curselection():
                if ScenesListBox.get(x) not in sel_scene:
                    sel_scene.append(ScenesListBox.get(x))

            if len(sel_scene) > 1:
                sel_scene = [sel_scene.pop()]

        def update_list_scene():
            global sel_scene
            global l_scene
            search_term_scene = search_scene_var.get()

            # Just a generic list to populate the listbox
            lbox_list_scene = SceneNames

            ScenesListBox.delete(0, 'end')

            for item in lbox_list_scene:
                if search_term_scene.lower() in item.lower():
                    ScenesListBox.insert('end', item)

            allitems_scene=list()
            for i in range(ScenesListBox.size()):
                allitems_scene.append(ScenesListBox.get(i))

            for i in sel_scene:
                if i in allitems_scene:
                    ScenesListBox.select_set(ScenesListBox.get(0, "end").index(i))

        SceneText = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        SceneText["font"] = ft
        SceneText["fg"] = "#333333"
        SceneText["justify"] = "right"
        SceneText["text"] = "Search by scene"
        SceneText.place(x=320, y=620, width=140, height=30)

        global search_scene_var
        search_scene_var = tk.StringVar()
        search_scene_var.trace("w", lambda name, index, mode: update_list_scene())

        global entry_scene
        entry_scene = tk.Entry(root, textvariable=search_scene_var, width=13)
        entry_scene.place(x=460, y=625, width=60, height=20)

        global ScenesListBox
        ScenesListBox = tk.Listbox(root, selectmode='browse',width=45, height=15)
        ScenesListBox.bind('<<ListboxSelect>>', CurSelect_scene)
        #entry.grid(row=0, column=0, padx=10, pady=3)
        ScenesListBox.grid(row=1, column=0, padx=10, pady=3)
        cursel_scene=tk.StringVar()
        lb1_scene=tk.Label(root,textvariable=cursel_scene)
        lb1_scene.grid(row=100,column=0,padx=10,pady=3)
        ScenesListBox.place(x=530, y=610, width=140, height=75)

        update_list_scene()

        GazeText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GazeText["font"] = ft
        GazeText["fg"] = "#333333"
        GazeText["justify"] = "right"
        GazeText["text"] = "Search by gaze"
        GazeText.place(x=700,y=80,width=155,height=30)

        global var_left
        var_left = tk.IntVar()
        global var_direct
        var_direct = tk.IntVar()
        global var_right
        var_right=tk.IntVar()
        global var_away
        var_away=tk.IntVar()
        global Left
        Left=tk.Checkbutton(root, variable=var_left)
        global Right
        Right=tk.Checkbutton(root, variable=var_right)
        global Direct
        Direct=tk.Checkbutton(root, variable=var_direct)
        global Away
        Away=tk.Checkbutton(root, variable=var_away)

        ft = tkFont.Font(family='Times',size=10)
        Left["font"] = ft
        Left["fg"] = "#333333"
        Left["justify"] = "left"
        Left["text"] = "Left"
        Left.place(x=830,y=55,width=100,height=25)
        Left["offvalue"] = "0"
        Left["onvalue"] = "1"
        Left["command"] = self.Left_command

        ft = tkFont.Font(family='Times',size=10)
        Direct["font"] = ft
        Direct["fg"] = "#333333"
        Direct["justify"] = "left"
        Direct["text"] = "Direct"
        Direct.place(x=835,y=75,width=100,height=25)
        Direct["offvalue"] = "0"
        Direct["onvalue"] = "1"
        Direct["command"] = self.Direct_command

        ft = tkFont.Font(family='Times',size=10)
        Right["font"] = ft
        Right["fg"] = "#333333"
        Right["justify"] = "left"
        Right["text"] = "Right"
        Right.place(x=834,y=95,width=100,height=25)
        Right["offvalue"] = "0"
        Right["onvalue"] = "1"
        Right["command"] = self.Right_command

        ft = tkFont.Font(family='Times',size=10)
        Away["font"] = ft
        Away["fg"] = "#333333"
        Away["justify"] = "left"
        Away["text"] = "Away"
        Away.place(x=835,y=115,width=100,height=25)
        Away["offvalue"] = "0"
        Away["onvalue"] = "1"
        Away["command"] = self.Away_command

        global var_mainchar
        var_mainchar = tk.IntVar()
        global MainChar
        MainChar=tk.Checkbutton(root, variable=var_mainchar)
        ft = tkFont.Font(family='Times',size=10)
        MainChar["font"] = ft
        MainChar["fg"] = "#333333"
        MainChar["justify"] = "left"
        MainChar["text"] = "Main Character"
        MainChar.place(x=1040,y=245,width=100,height=25)
        MainChar["offvalue"] = "0"
        MainChar["onvalue"] = "1"
        MainChar["command"] = self.MainChar_command

        global var_everyone
        var_everyone = tk.IntVar()
        global Everyone
        Everyone=tk.Checkbutton(root, variable=var_everyone)
        ft = tkFont.Font(family='Times',size=10)
        Everyone["font"] = ft
        Everyone["fg"] = "#333333"
        Everyone["justify"] = "left"
        Everyone["text"] = "Anyone"
        Everyone.place(x=1020,y=265,width=100,height=25)
        Everyone["offvalue"] = "0"
        Everyone["onvalue"] = "1"
        Everyone["command"] = self.Everyone_command

        global var_everyone_2
        var_everyone_2 = tk.IntVar()
        global Everyone_2
        Everyone_2=tk.Checkbutton(root, variable=var_everyone_2)
        ft = tkFont.Font(family='Times',size=10)
        Everyone_2["font"] = ft
        Everyone_2["fg"] = "#333333"
        Everyone_2["justify"] = "left"
        Everyone_2["text"] = "Everyone"
        Everyone_2.place(x=1025,y=285,width=100,height=25)
        Everyone_2["offvalue"] = "0"
        Everyone_2["onvalue"] = "1"
        Everyone_2["command"] = self.Everyone_2_command

        AgeText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        AgeText["font"] = ft
        AgeText["fg"] = "#333333"
        AgeText["justify"] = "right"
        AgeText["text"] = "Search by age"
        AgeText.place(x=700,y=200,width=155,height=30)

        global var_child
        var_child = tk.IntVar()
        global var_youth
        var_youth = tk.IntVar()
        global var_adult
        var_adult=tk.IntVar()
        global var_senior
        var_senior=tk.IntVar()
        global Child
        Child=tk.Checkbutton(root, variable=var_child)
        global Youth
        Youth=tk.Checkbutton(root, variable=var_youth)
        global Adult
        Adult=tk.Checkbutton(root, variable=var_adult)
        global Senior
        Senior=tk.Checkbutton(root, variable=var_senior)

        ft = tkFont.Font(family='Times',size=10)
        Child["font"] = ft
        Child["fg"] = "#333333"
        Child["justify"] = "left"
        Child["text"] = "Children (0-14)"
        Child.place(x=860,y=170,width=100,height=25)
        Child["offvalue"] = "0"
        Child["onvalue"] = "1"
        Child["command"] = self.Child_command

        ft = tkFont.Font(family='Times',size=10)
        Youth["font"] = ft
        Youth["fg"] = "#333333"
        Youth["justify"] = "left"
        Youth["text"] = "Youth (15-24)"
        Youth.place(x=857,y=190,width=100,height=25)
        Youth["offvalue"] = "0"
        Youth["onvalue"] = "1"
        Youth["command"] = self.Youth_command

        ft = tkFont.Font(family='Times',size=10)
        Adult["font"] = ft
        Adult["fg"] = "#333333"
        Adult["justify"] = "left"
        Adult["text"] = "Adult (25-60)"
        Adult.place(x=856,y=210,width=100,height=25)
        Adult["offvalue"] = "0"
        Adult["onvalue"] = "1"
        Adult["command"] = self.Adult_command

        ft = tkFont.Font(family='Times',size=10)
        Senior["font"] = ft
        Senior["fg"] = "#333333"
        Senior["justify"] = "left"
        Senior["text"] = "Senior (60+)"
        Senior.place(x=854,y=230,width=100,height=25)
        Senior["offvalue"] = "0"
        Senior["onvalue"] = "1"
        Senior["command"] = self.Senior_command

        GenderText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GenderText["font"] = ft
        GenderText["fg"] = "#333333"
        GenderText["justify"] = "right"
        GenderText["text"] = "Search by gender"
        GenderText.place(x=695,y=300,width=160,height=30)

        global var_male
        var_male = tk.IntVar()
        global var_female
        var_female = tk.IntVar()
        global Male
        Male=tk.Checkbutton(root, variable=var_male)
        global Female
        Female=tk.Checkbutton(root, variable=var_female)

        ft = tkFont.Font(family='Times',size=10)
        Male["font"] = ft
        Male["fg"] = "#333333"
        Male["justify"] = "left"
        Male["text"] = "Male"
        Male.place(x=855,y=290,width=70,height=25)
        Male["offvalue"] = "0"
        Male["onvalue"] = "1"
        Male["command"] = self.Male_command

        ft = tkFont.Font(family='Times',size=10)
        Female["font"] = ft
        Female["fg"] = "#333333"
        Female["justify"] = "left"
        Female["text"] = "Female"
        Female.place(x=860,y=310,width=70,height=25)
        Female["offvalue"] = "0"
        Female["onvalue"] = "1"
        Female["command"] = self.Female_command

        EmotionText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        EmotionText["font"] = ft
        EmotionText["fg"] = "#333333"
        EmotionText["justify"] = "right"
        EmotionText["text"] = "Search by emotion"
        EmotionText.place(x=695,y=400,width=160,height=30)

        ShapeText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        ShapeText["font"] = ft
        ShapeText["fg"] = "#333333"
        ShapeText["justify"] = "right"
        ShapeText["text"] = "Search by shape"
        ShapeText.place(x=695,y=510,width=160,height=30)

        global var_square
        var_square = tk.IntVar()
        global Square
        Square=tk.Checkbutton(root, variable=var_square)
        ft = tkFont.Font(family='Times',size=10)
        Square["font"] = ft
        Square["fg"] = "#333333"
        Square["justify"] = "left"
        Square["text"] = "Square"
        Square.place(x=845,y=510,width=100,height=25)
        Square["offvalue"] = "0"
        Square["onvalue"] = "1"
        Square["command"] = self.Square_command

        global var_horizontal
        var_horizontal = tk.IntVar()
        global Horizontal
        Horizontal=tk.Checkbutton(root, variable=var_horizontal)
        ft = tkFont.Font(family='Times',size=10)
        Horizontal["font"] = ft
        Horizontal["fg"] = "#333333"
        Horizontal["justify"] = "left"
        Horizontal["text"] = "Horizontal"
        Horizontal.place(x=925,y=510,width=100,height=25)
        Horizontal["offvalue"] = "0"
        Horizontal["onvalue"] = "1"
        Horizontal["command"] = self.Horizontal_command

        global var_vertical
        var_vertical = tk.IntVar()
        global Vertical
        Vertical=tk.Checkbutton(root, variable=var_vertical)
        ft = tkFont.Font(family='Times',size=10)
        Vertical["font"] = ft
        Vertical["fg"] = "#333333"
        Vertical["justify"] = "left"
        Vertical["text"] = "Vertical"
        Vertical.place(x=1015,y=510,width=80,height=25)
        Vertical["offvalue"] = "0"
        Vertical["onvalue"] = "1"
        Vertical["command"] = self.Vertical_command

        ColorText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        ColorText["font"] = ft
        ColorText["fg"] = "#333333"
        ColorText["justify"] = "right"
        ColorText["text"] = "Search by color"
        ColorText.place(x=700,y=550,width=160,height=30)

        global var_color
        var_color = tk.IntVar()
        global Color
        Color=tk.Checkbutton(root, variable=var_color)
        ft = tkFont.Font(family='Times',size=10)
        Color["font"] = ft
        Color["fg"] = "#333333"
        Color["justify"] = "left"
        Color["text"] = "Color"
        Color.place(x=842,y=550,width=100,height=25)
        Color["offvalue"] = "0"
        Color["onvalue"] = "1"
        Color["command"] = self.Color_command

        global var_greyscale
        var_greyscale = tk.IntVar()
        global Greyscale
        Greyscale=tk.Checkbutton(root, variable=var_greyscale)
        ft = tkFont.Font(family='Times',size=10)
        Greyscale["font"] = ft
        Greyscale["fg"] = "#333333"
        Greyscale["justify"] = "left"
        Greyscale["text"] = "Greyscale"
        Greyscale.place(x=924,y=550,width=100,height=25)
        Greyscale["offvalue"] = "0"
        Greyscale["onvalue"] = "1"
        Greyscale["command"] = self.Greyscale_command

        TonalityText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        TonalityText["font"] = ft
        TonalityText["fg"] = "#333333"
        TonalityText["justify"] = "right"
        TonalityText["text"] = "Search by contrast"
        TonalityText.place(x=670,y=590,width=190,height=30)

        global var_dark
        var_dark = tk.IntVar()
        global Dark
        Dark=tk.Checkbutton(root, variable=var_dark)
        ft = tkFont.Font(family='Times',size=10)
        Dark["font"] = ft
        Dark["fg"] = "#333333"
        Dark["justify"] = "left"
        Dark["text"] = "Low"
        Dark.place(x=840,y=590,width=100,height=25)
        Dark["offvalue"] = "0"
        Dark["onvalue"] = "1"
        Dark["command"] = self.Dark_command

        global var_normal
        var_normal = tk.IntVar()
        global Normal
        Normal=tk.Checkbutton(root, variable=var_normal)
        ft = tkFont.Font(family='Times',size=10)
        Normal["font"] = ft
        Normal["fg"] = "#333333"
        Normal["justify"] = "left"
        Normal["text"] = "Normal"
        Normal.place(x=918,y=590,width=100,height=25)
        Normal["offvalue"] = "0"
        Normal["onvalue"] = "1"
        Normal["command"] = self.Normal_command

        DominantText=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        DominantText["font"] = ft
        DominantText["fg"] = "#333333"
        DominantText["justify"] = "right"
        DominantText["text"] = "Dominant colors"
        DominantText.place(x=692,y=650,width=160,height=30)

        global var_red
        var_red = tk.IntVar()
        global Red
        Red=tk.Checkbutton(root, variable=var_red)
        ft = tkFont.Font(family='Times',size=10)
        Red["font"] = ft
        Red["fg"] = "#333333"
        Red["justify"] = "right"
        Red["text"] = "Red"
        Red.place(x=854,y=630,width=70,height=25)
        Red["offvalue"] = "0"
        Red["onvalue"] = "1"
        Red["command"] = self.Red_command

        global var_pink
        var_pink = tk.IntVar()
        global Pink
        Pink=tk.Checkbutton(root, variable=var_pink)
        ft = tkFont.Font(family='Times',size=10)
        Pink["font"] = ft
        Pink["fg"] = "#333333"
        Pink["justify"] = "right"
        Pink["text"] = "Pink"
        Pink.place(x=855,y=650,width=70,height=25)
        Pink["offvalue"] = "0"
        Pink["onvalue"] = "1"
        Pink["command"] = self.Pink_command

        global var_orange
        var_orange = tk.IntVar()
        global Orange
        Orange=tk.Checkbutton(root, variable=var_orange)
        ft = tkFont.Font(family='Times',size=10)
        Orange["font"] = ft
        Orange["fg"] = "#333333"
        Orange["justify"] = "left"
        Orange["text"] = "Orange"
        Orange.place(x=863,y=670,width=70,height=25)
        Orange["offvalue"] = "0"
        Orange["onvalue"] = "1"
        Orange["command"] = self.Orange_command

        global var_yellow
        var_yellow = tk.IntVar()
        global Yellow
        Yellow=tk.Checkbutton(root, variable=var_yellow)
        ft = tkFont.Font(family='Times',size=10)
        Yellow["font"] = ft
        Yellow["fg"] = "#333333"
        Yellow["justify"] = "left"
        Yellow["text"] = "Yellow"
        Yellow.place(x=861,y=690,width=70,height=25)
        Yellow["offvalue"] = "0"
        Yellow["onvalue"] = "1"
        Yellow["command"] = self.Yellow_command

        global var_green
        var_green = tk.IntVar()
        global Green
        Green=tk.Checkbutton(root, variable=var_green)
        ft = tkFont.Font(family='Times',size=10)
        Green["font"] = ft
        Green["fg"] = "#333333"
        Green["justify"] = "left"
        Green["text"] = "Green"
        Green.place(x=930,y=630,width=70,height=25)
        Green["offvalue"] = "0"
        Green["onvalue"] = "1"
        Green["command"] = self.Green_command

        global var_green_blue
        var_green_blue = tk.IntVar()
        global GreenBlue
        GreenBlue=tk.Checkbutton(root, variable=var_green_blue)
        ft = tkFont.Font(family='Times',size=10)
        GreenBlue["font"] = ft
        GreenBlue["fg"] = "#333333"
        GreenBlue["justify"] = "left"
        GreenBlue["text"] = "Green-Blue"
        GreenBlue.place(x=930,y=650,width=100,height=25)
        GreenBlue["offvalue"] = "0"
        GreenBlue["onvalue"] = "1"
        GreenBlue["command"] = self.GreenBlue_command

        global var_blue
        var_blue = tk.IntVar()
        global Blue
        Blue=tk.Checkbutton(root, variable=var_blue)
        ft = tkFont.Font(family='Times',size=10)
        Blue["font"] = ft
        Blue["fg"] = "#333333"
        Blue["justify"] = "left"
        Blue["text"] = "Blue"
        Blue.place(x=928,y=670,width=70,height=25)
        Blue["offvalue"] = "0"
        Blue["onvalue"] = "1"
        Blue["command"] = self.Blue_command

        global var_purple
        var_purple = tk.IntVar()
        global Purple
        Purple=tk.Checkbutton(root, variable=var_purple)
        ft = tkFont.Font(family='Times',size=10)
        Purple["font"] = ft
        Purple["fg"] = "#333333"
        Purple["justify"] = "left"
        Purple["text"] = "Purple"
        Purple.place(x=934,y=690,width=70,height=25)
        Purple["offvalue"] = "0"
        Purple["onvalue"] = "1"
        Purple["command"] = self.Purple_command

        ####################

        global var_neutral
        var_neutral = tk.IntVar()
        global Neutral
        Neutral=tk.Checkbutton(root, variable=var_neutral)
        ft = tkFont.Font(family='Times',size=10)
        Neutral["font"] = ft
        Neutral["fg"] = "#333333"
        Neutral["justify"] = "left"
        Neutral["text"] = "Neutral"
        Neutral.place(x=860,y=390,width=70,height=25)
        Neutral["offvalue"] = "0"
        Neutral["onvalue"] = "1"
        Neutral["command"] = self.Neutral_command

        global var_happy
        var_happy = tk.IntVar()
        global Happy
        Happy=tk.Checkbutton(root, variable=var_happy)
        ft = tkFont.Font(family='Times',size=10)
        Happy["font"] = ft
        Happy["fg"] = "#333333"
        Happy["justify"] = "left"
        Happy["text"] = "Happy"
        Happy.place(x=858,y=410,width=70,height=25)
        Happy["offvalue"] = "0"
        Happy["onvalue"] = "1"
        Happy["command"] = self.Happy_command

        global var_sad
        var_sad = tk.IntVar()
        global Sad
        Sad=tk.Checkbutton(root, variable=var_sad)
        ft = tkFont.Font(family='Times',size=10)
        Sad["font"] = ft
        Sad["fg"] = "#333333"
        Sad["justify"] = "left"
        Sad["text"] = "Sad"
        Sad.place(x=850,y=430,width=70,height=25)
        Sad["offvalue"] = "0"
        Sad["onvalue"] = "1"
        Sad["command"] = self.Sad_command

        global var_surprise
        var_surprise = tk.IntVar()
        global Surprise
        Surprise=tk.Checkbutton(root, variable=var_surprise)
        ft = tkFont.Font(family='Times',size=10)
        Surprise["font"] = ft
        Surprise["fg"] = "#333333"
        Surprise["justify"] = "left"
        Surprise["text"] = "Surprise"
        Surprise.place(x=930,y=390,width=70,height=25)
        Surprise["offvalue"] = "0"
        Surprise["onvalue"] = "1"
        Surprise["command"] = self.Surprise_command

        global var_fear
        var_fear = tk.IntVar()
        global Fear
        Fear=tk.Checkbutton(root, variable=var_fear)
        ft = tkFont.Font(family='Times',size=10)
        Fear["font"] = ft
        Fear["fg"] = "#333333"
        Fear["justify"] = "left"
        Fear["text"] = "Fear"
        Fear.place(x=852,y=450,width=70,height=25)
        Fear["offvalue"] = "0"
        Fear["onvalue"] = "1"
        Fear["command"] = self.Fear_command

        global var_disgust
        var_disgust = tk.IntVar()
        global Disgust
        Disgust=tk.Checkbutton(root, variable=var_disgust)
        ft = tkFont.Font(family='Times',size=10)
        Disgust["font"] = ft
        Disgust["fg"] = "#333333"
        Disgust["justify"] = "left"
        Disgust["text"] = "Disgust"
        Disgust.place(x=930,y=410,width=70,height=25)
        Disgust["offvalue"] = "0"
        Disgust["onvalue"] = "1"
        Disgust["command"] = self.Disgust_command

        global var_anger
        var_anger = tk.IntVar()
        global Anger
        Anger=tk.Checkbutton(root, variable=var_anger)
        ft = tkFont.Font(family='Times',size=10)
        Anger["font"] = ft
        Anger["fg"] = "#333333"
        Anger["justify"] = "left"
        Anger["text"] = "Anger"
        Anger.place(x=925,y=430,width=70,height=25)
        Anger["offvalue"] = "0"
        Anger["onvalue"] = "1"
        Anger["command"] = self.Anger_command

        global var_contempt
        var_contempt = tk.IntVar()
        global Contempt
        Contempt=tk.Checkbutton(root, variable=var_contempt)
        ft = tkFont.Font(family='Times',size=10)
        Contempt["font"] = ft
        Contempt["fg"] = "#333333"
        Contempt["justify"] = "left"
        Contempt["text"] = "Contempt"
        Contempt.place(x=933,y=450,width=70,height=25)
        Contempt["offvalue"] = "0"
        Contempt["onvalue"] = "1"
        Contempt["command"] = self.Contempt_command

    def GButton_290_command(self):

        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        vector_filename = askopenfilename()
        img0 = cv2.imread(vector_filename)

        global similar_image_chosen
        similar_image_chosen = True

        vtr = imgsim.Vectorizer()
        vec0 = vtr.vectorize(img0)

        global similar_filter_filenames
        similar_filter_filenames = []

        for key in vectors_dict:
            dist = imgsim.distance(vec0, vectors_dict[key])
            if dist < 16 and dist != 0:
                similar_filter_filenames.append(key)

        global GMessage_276
        GMessage_276["text"] = "File chosen"

    def GButton_558_command(self):
        global imagesFolder
        imagesFolder = askdirectory()
        global imagesFolderName
        imagesFolderName = imagesFolder.split('/')[-1]
        Directory["text"] = imagesFolderName
        if not os.path.isdir(imagesFolder):
            return
        with open('data/faces.pickle', 'rb') as handle:
            global face_dict
            face_dict = pickle.load(handle)
        with open('data/vectors.pickle', 'rb') as handle:
            global vectors_dict
            vectors_dict = pickle.load(handle)
        with open('data/main_character.pickle', 'rb') as handle:
            global main_character_dict
            main_character_dict = pickle.load(handle)
        with open('data/environments.pickle', 'rb') as handle:
            global environment_dict
            environment_dict = pickle.load(handle)
        with open('data/captions.pickle', 'rb') as handle:
            global captions_dict
            captions_dict = pickle.load(handle)
        with open('data/objects.pickle', 'rb') as handle:
            global objects_dict
            objects_dict = pickle.load(handle)
        with open('data/date.pickle', 'rb') as handle:
            global date_dict
            date_dict = pickle.load(handle)
        with open('data/dimensions.pickle', 'rb') as handle:
            global dimensions_dict
            dimensions_dict = pickle.load(handle)
        with open('data/gazes.pickle', 'rb') as handle:
            global gazes_dict
            gazes_dict = pickle.load(handle)
        with open('data/age_gender.pickle', 'rb') as handle:
            global age_gender_dict
            age_gender_dict = pickle.load(handle)
        with open('data/emotions.pickle', 'rb') as handle:
            global emotion_dict
            emotion_dict = pickle.load(handle)
        with open('data/colors.pickle', 'rb') as handle:
            global color_dict
            color_dict = pickle.load(handle)

        predictor_path = 'shape_predictor_5_face_landmarks.dat'
        face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

        global detector
        global sp
        global facerec
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(predictor_path)
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)

        global sentence_model
        sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

        if not os.path.isdir('search_results/' + imagesFolderName):
            os.mkdir('search_results/' + imagesFolderName)

    def GButton_158_command(self):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        face_filename = askopenfilename()

        img = dlib.load_rgb_image(face_filename)
        imgCV = cv2.imread(face_filename)

        global face_filter_filenames
        face_filter_filenames = []

        if img.shape[0] > 750:
            scale_percent = (750 / img.shape[0]) * 100  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            imgCV = cv2.resize(imgCV, dim, interpolation=cv2.INTER_AREA)

        dets = detector(img, 1)

        for k, rect in enumerate(dets):
            img = cv2.rectangle(img, (rect.tl_corner().x, rect.tl_corner().y), (rect.br_corner().x, rect.br_corner().y),
                                (255, 0, 0), 3)

        # This will display all the available mouse click events
        events = [i for i in dir(cv2) if 'EVENT' in i]

        def mouse_in_face_rect(x, y):
            for k, rect in enumerate(dets):
                if rect.left() < x < rect.right() and rect.top() < y < rect.bottom():
                    return rect
            return False

        # click event function
        def click_event(event, x, y, flags, param):
            global clickedX, clickedY
            if event == cv2.EVENT_LBUTTONDOWN and mouse_in_face_rect(x, y):
                clickedX = x
                clickedY = y
                cv2.destroyAllWindows()

            if event == cv2.EVENT_MOUSEMOVE and mouse_in_face_rect(x, y):
                rect = mouse_in_face_rect(x, y)
                cv2.rectangle(imgCV, (rect.tl_corner().x, rect.tl_corner().y), (rect.br_corner().x, rect.br_corner().y),
                              (0, 255, 0), 3)
                cv2.imshow("image", imgCV)

            if event == cv2.EVENT_MOUSEMOVE and not mouse_in_face_rect(x, y):
                for k, rect in enumerate(dets):
                    cv2.rectangle(imgCV, (rect.tl_corner().x, rect.tl_corner().y),
                                  (rect.br_corner().x, rect.br_corner().y),
                                  (255, 0, 0), 3)
                cv2.imshow("image", imgCV)

        cv2.imshow("image", img)
        # calling the mouse click event
        cv2.setMouseCallback("image", click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        clickedFace = -1

        for k, rect in enumerate(dets):
            if rect.left() < clickedX < rect.right() and rect.top() < clickedY < rect.bottom():
                clickedFace = k

        for k, d in enumerate(dets):
            if k == clickedFace:
                shape = sp(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                chosenFaceDescriptor = face_descriptor
                break

        if clickedFace != -1:
            global face_selected
            face_selected = True

        fileIndex = 0
        for imageWiseDescriptors in face_dict['Descriptors']:
            fname = face_dict['Filename'][fileIndex]
            rectangles = face_dict['Rectangles'][fileIndex]
            fileIndex += 1

            rectIndex = 0
            for desc in imageWiseDescriptors:
                dist = np.linalg.norm(np.array(chosenFaceDescriptor) - np.array(desc))
                if dist < 0.55:
                    matchedRect = rectangles[rectIndex]
                    if not fname in face_filename:
                        face_filter_filenames.append(fname)
                rectIndex += 1
        global GLabel_476
        GLabel_476["text"] = "File chosen"
        print("158")


    def GCheckBox_962_command(self):
        print("962")


    def GCheckBox_134_command(self):
        print("134")

    def GCheckBox_934_command(self):
        print("934")

    def GCheckBox_26_command(self):
        print("26")
        GCheckBox_588.deselect()

    def SmallerThan_command(self):
        Exactly.deselect()
        LargerThan.deselect()

    def Exactly_command(self):
        SmallerThan.deselect()
        LargerThan.deselect()

    def LargerThan_command(self):
        SmallerThan.deselect()
        Exactly.deselect()

    def Left_command(self):
        Right.deselect()
        Direct.deselect()
        Away.deselect()

    def Right_command(self):
        Left.deselect()
        Direct.deselect()
        Away.deselect()

    def Direct_command(self):
        Left.deselect()
        Right.deselect()
        Away.deselect()

    def Away_command(self):
        Left.deselect()
        Right.deselect()
        Direct.deselect()

    def MainChar_command(self):
        Everyone.deselect()
        Everyone_2.deselect()
        if var_child.get() + var_youth.get() + var_adult.get() + var_senior.get() > 1:
            Child.deselect()
            Adult.deselect()
            Youth.deselect()
            Senior.deselect()
        if var_neutral.get() + var_happy.get() + var_sad.get() + var_fear.get() + var_surprise.get() + var_disgust.get() + var_anger.get() + var_contempt.get() > 1:
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()


    def Everyone_command(self):
        MainChar.deselect()
        Everyone_2.deselect()

    def Everyone_2_command(self):
        MainChar.deselect()
        Everyone.deselect()
        if var_child.get() + var_youth.get() + var_adult.get() + var_senior.get() > 1:
            Child.deselect()
            Adult.deselect()
            Youth.deselect()
            Senior.deselect()
        if var_neutral.get() + var_happy.get() + var_sad.get() + var_fear.get() + var_surprise.get() + var_disgust.get() + var_anger.get() + var_contempt.get() > 1:
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Child_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Adult.deselect()
            Youth.deselect()
            Senior.deselect()

    def Adult_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Child.deselect()
            Youth.deselect()
            Senior.deselect()

    def Youth_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Child.deselect()
            Adult.deselect()
            Senior.deselect()

    def Senior_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Child.deselect()
            Adult.deselect()
            Youth.deselect()

    def Male_command(self):
        Female.deselect()

    def Female_command(self):
        Male.deselect()

    def GCheckBox_588_command(self):
        print("588")
        GCheckBox_26.deselect()

    def Neutral_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Happy_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Sad_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Fear_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Surprise_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Disgust.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Disgust_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Anger.deselect()
            Contempt.deselect()

    def Anger_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Contempt.deselect()

    def Contempt_command(self):
        if var_mainchar.get() or var_everyone_2.get():
            Neutral.deselect()
            Happy.deselect()
            Sad.deselect()
            Fear.deselect()
            Surprise.deselect()
            Disgust.deselect()
            Anger.deselect()

    def Color_command(self):
        Greyscale.deselect()
        if var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 2:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def Greyscale_command(self):
        Color.deselect()
        Red.deselect()
        Pink.deselect()
        Orange.deselect()
        Yellow.deselect()
        Green.deselect()
        GreenBlue.deselect()
        Blue.deselect()
        Purple.deselect()

    def Red_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def Pink_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def Orange_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def Yellow_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def Green_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            GreenBlue.deselect()
            Blue.deselect()
            Purple.deselect()

    def GreenBlue_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            Blue.deselect()
            Purple.deselect()

    def Blue_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Purple.deselect()

    def Purple_command(self):
        if (var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 1) or var_greyscale.get() == 1:
            Red.deselect()
            Pink.deselect()
            Orange.deselect()
            Yellow.deselect()
            Green.deselect()
            GreenBlue.deselect()
            Blue.deselect()

    def Square_command(self):
        Horizontal.deselect()
        Vertical.deselect()

    def Horizontal_command(self):
        Square.deselect()
        Vertical.deselect()

    def Vertical_command(self):
        Horizontal.deselect()
        Square.deselect()

    def Dark_command(self):
        Normal.deselect()

    def Normal_command(self):
        Dark.deselect()


    def GButton_851_command(self):
        GCheckBox_962.deselect()
        GCheckBox_134.deselect()
        GCheckBox_934.deselect()
        GCheckBox_26.deselect()
        GCheckBox_588.deselect()

        SmallerThan.deselect()
        Exactly.deselect()
        LargerThan.deselect()

        Left.deselect()
        Right.deselect()
        Away.deselect()
        Direct.deselect()

        MainChar.deselect()
        Everyone.deselect()
        Everyone_2.deselect()

        Child.deselect()
        Adult.deselect()
        Youth.deselect()
        Senior.deselect()

        Male.deselect()
        Female.deselect()

        Neutral.deselect()
        Happy.deselect()
        Sad.deselect()
        Fear.deselect()
        Surprise.deselect()
        Disgust.deselect()
        Anger.deselect()
        Contempt.deselect()

        Red.deselect()
        Pink.deselect()
        Orange.deselect()
        Yellow.deselect()
        Green.deselect()
        GreenBlue.deselect()
        Blue.deselect()
        Purple.deselect()

        Dark.deselect()
        Normal.deselect()

        Color.deselect()
        Greyscale.deselect()

        Square.deselect()
        Horizontal.deselect()
        Vertical.deselect()

        global face_selected
        face_selected = False
        global GLabel_476
        GLabel_476["text"] = ""
        global face_filter_filenames
        face_filter_filenames = face_dict["Filename"]

        global similar_image_chosen
        similar_image_chosen = False
        global GMessage_276
        GMessage_276["text"] = ""
        global similar_filter_filenames
        similar_filter_filenames = vectors_dict.keys()

        global GLineEdit_14
        GLineEdit_14.delete(0, tk.END)
        global GLineEdit_549
        GLineEdit_549.delete(0, tk.END)
        num_people_var = ""

        global startdate
        startdate.delete(0, "end")
        global enddate
        enddate.delete(0, "end")

        global WidthText
        WidthText.delete(0, 'end')
        global HeightText
        HeightText.delete(0, 'end')

        global ObjectsListBox
        ObjectsListBox.selection_clear(0, 'end')
        global sel
        sel = []
        global entry
        entry.delete(0, 'end')

        global ScenesListBox
        ScenesListBox.selection_clear(0, 'end')
        global sel_scene
        sel_scene = []
        global entry_scene
        entry_scene.delete(0, 'end')

        print("851")

    def ObjectListClear_command(self):
        global ObjectsListBox
        ObjectsListBox.selection_clear(0, 'end')
        global sel
        sel = []
        global entry
        entry.delete(0, 'end')

    def SceneListClear_command(self):
        global ScenesListBox
        ScenesListBox.selection_clear(0, 'end')
        global sel_scene
        sel_scene = []
        global entry_scene
        entry_scene.delete(0, 'end')

    def GButton_227_command(self):

        global entry
        entry.delete(0, 'end')

        # global entry_scene
        # entry_scene.delete(0, 'end')
        global imagesFolderName
        files = glob.glob('search_results/' + imagesFolderName + '/*')
        for f in files:
            os.remove(f)

        env_filter_filenames = []
        if var_outdoor.get() + var_indoor.get() == 1:
            file_index = 0
            for filename in environment_dict['Filename']:
                if var_indoor.get() and environment_dict['IO'][file_index] == 'indoor':
                    env_filter_filenames.append(filename)
                if var_outdoor.get() and environment_dict['IO'][file_index] == 'outdoor':
                    env_filter_filenames.append(filename)
                file_index +=1
        else:
            env_filter_filenames = environment_dict['Filename']

        shot_filter_filenames = []
        if var_close.get() + var_medium.get() + var_long.get() >= 1:
            file_index = 0
            for filename in main_character_dict['Filename']:
                if var_close.get() and main_character_dict['Shot Type'][file_index] == 'close':
                    shot_filter_filenames.append(filename)
                if var_medium.get() and main_character_dict['Shot Type'][file_index] == 'medium':
                    shot_filter_filenames.append(filename)
                if var_long.get() and main_character_dict['Shot Type'][file_index] == 'long':
                    shot_filter_filenames.append(filename)
                file_index += 1
        else:
            shot_filter_filenames = main_character_dict['Filename']

        person_count_filter_filenames = []
        if GLineEdit_14.get() == '':
            person_count_filter_filenames = main_character_dict['Filename']
        else:
            person_count_entry = int(GLineEdit_14.get())
            file_index = 0
            for filename in objects_dict['Filename']:
                count = 0
                for object in objects_dict['Objects'][file_index]:
                    if 'person' in object:
                        count += 1
                if count == person_count_entry:
                    person_count_filter_filenames.append(filename)
                file_index += 1

        global face_selected
        global face_filter_filenames
        if face_selected == False:
            face_filter_filenames = face_dict['Filename']

        global similar_image_chosen
        global similar_filter_filenames
        if similar_image_chosen == False:
            similar_filter_filenames = vectors_dict.keys()

        global sentence_embeddings
        description_filter_filenames = []
        global GLineEdit_549
        if GLineEdit_549.get() != '':
            sentence = GLineEdit_549.get()
            detector = google_translator()
            detect_result = detector.detect(sentence)
            if detect_result[0] == 'fi': sentence = detector.translate(sentence,lang_tgt='en')

            input_description_encoding = sentence_model.encode(sentence)
            similarity_array1 = cosine_similarity(
                [input_description_encoding],
                captions_dict["Embeddings1"]
            )
            similarity_array2 = cosine_similarity(
                [input_description_encoding],
                captions_dict["Embeddings2"]
            )
            similarity_array3 = cosine_similarity(
                [input_description_encoding],
                captions_dict["Embeddings3"]
            )
            similar_sentence_indexes1 = np.argwhere(similarity_array1 > 0.70)[:,1]
            similar_sentence_indexes2 = np.argwhere(similarity_array2 > 0.70)[:,1]
            similar_sentence_indexes3 = np.argwhere(similarity_array3 > 0.70)[:,1]

            for similar_sentences_index in similar_sentence_indexes1:
                description_filter_filenames.append(captions_dict["Filename"][similar_sentences_index])
            for similar_sentences_index in similar_sentence_indexes2:
                if captions_dict["Filename"][similar_sentences_index] not in description_filter_filenames:
                    description_filter_filenames.append(captions_dict["Filename"][similar_sentences_index])
            for similar_sentences_index in similar_sentence_indexes3:
                if captions_dict["Filename"][similar_sentences_index] not in description_filter_filenames:
                    description_filter_filenames.append(captions_dict["Filename"][similar_sentences_index])
        else:
            description_filter_filenames = captions_dict["Filename"]

        global startdate
        global enddate
        date_filter_filenames = []
        #if both dates are filled and they are the same
        if (startdate.get() != '' and enddate.get() != '') and (startdate.get_date() == enddate.get_date()):
            date = startdate.get_date()
            file_index = 0
            for filename in date_dict['Filename']:
                if len(date_dict['Date'][file_index]) == 10:
                    date_of_file = date_dict['Date'][file_index]
                    date_split = date_of_file.split(':')
                    day = date_split[0]
                    month = date_split[1]
                    year = date_split[2]
                    dateoffile = datetime.datetime(int(year), int(month), int(day)).date()
                    if dateoffile == date:
                        date_filter_filenames.append(filename)
                file_index += 1
        #if only one of the dates are filled
        if (startdate.get() == '' and enddate.get() != '') or (startdate.get() != '' and enddate.get() == ''):
            file_index = 0
            for filename in date_dict['Filename']:
                if len(date_dict['Date'][file_index]) == 10:
                    date_of_file = date_dict['Date'][file_index]
                    date_split = date_of_file.split(':')
                    day = date_split[0]
                    month = date_split[1]
                    year = date_split[2]
                    dateoffile = datetime.datetime(int(year), int(month), int(day)).date()
                    if startdate.get() == '':
                        if dateoffile <= enddate.get_date():
                            date_filter_filenames.append(filename)
                    if enddate.get() == '':
                        if dateoffile >= startdate.get_date():
                            date_filter_filenames.append(filename)
                file_index += 1

        #if both dates are picked
        if (startdate.get() != '' and enddate.get() != '') and (startdate.get_date() != enddate.get_date()):
            file_index = 0
            for filename in date_dict['Filename']:
                if len(date_dict['Date'][file_index]) == 10:
                    date_of_file = date_dict['Date'][file_index]
                    date_split = date_of_file.split(':')
                    day = date_split[0]
                    month = date_split[1]
                    year = date_split[2]
                    dateoffile = datetime.datetime(int(year), int(month), int(day)).date()
                    if dateoffile >= startdate.get_date() and dateoffile <= enddate.get_date():
                        date_filter_filenames.append(filename)
                file_index += 1

        if (startdate.get() == '' and enddate.get() == ''):
            date_filter_filenames = date_dict["Filename"]

        dimension_filter_filenames = []
        if var_smallerthan.get() + var_exactly.get() + var_largerthan.get() >= 1 and (len(WidthText.get()) > 0 and len(HeightText.get()) > 0):
            file_index = 0
            for filename in dimensions_dict['Filename']:
                width = dimensions_dict['Dimensions'][file_index].split(':')[0]
                height = dimensions_dict['Dimensions'][file_index].split(':')[1]
                if var_smallerthan.get() and int(width) <= int(WidthText.get()) and int(height) <= int(HeightText.get()):
                    dimension_filter_filenames.append(filename)
                if var_exactly.get() and int(width) == int(WidthText.get()) and int(height) == int(HeightText.get()):
                    dimension_filter_filenames.append(filename)
                if var_largerthan.get() and int(width) >= int(WidthText.get()) and int(height) >= int(HeightText.get()):
                    dimension_filter_filenames.append(filename)
                file_index += 1
        else:
            dimension_filter_filenames = dimensions_dict['Filename']

        object_filter_filenames = []
        global ObjectsListBox
        global ObjectNames
        if len(ObjectsListBox.curselection()) > 0:
            chosenObjects = [e for i, e in enumerate(ObjectNames) if i in list(ObjectsListBox.curselection())]

            file_index = 0
            for filename in objects_dict['Filename']:
                objectsInImageWithConf = objects_dict['Objects'][file_index]
                objectsInImage = []
                for objectInImage in objectsInImageWithConf:
                    objInImageConfSplit = objectInImage.split(' ')
                    if len(objInImageConfSplit) == 2:
                        objectInImage = objInImageConfSplit[1]
                    if len(objInImageConfSplit) == 3:
                        objectInImage = objInImageConfSplit[1] + ' ' + objInImageConfSplit[2]
                    conf = objInImageConfSplit[0]
                    if objectInImage not in objectsInImage and float(conf) > 0.2:
                        objectsInImage.append(objectInImage)
                #if all of the chosen objects are present in the image
                if(set(chosenObjects).issubset(set(objectsInImage))):
                    object_filter_filenames.append(filename)
                file_index += 1
        else:
            object_filter_filenames = objects_dict['Filename']

        scene_filter_filenames = []
        global ScenesListBox
        global SceneNames
        if len(ScenesListBox.curselection()) > 0:
            chosenScene = [e for i, e in enumerate(SceneNames) if i in list(ScenesListBox.curselection())]

            file_index = 0
            for filename in environment_dict['Filename']:
                sceneInfo = environment_dict['Environments'][file_index]
                top_scenes = [ sceneInfo[0].split(' ')[1], sceneInfo[1].split(' ')[1]]

                if chosenScene[0] in top_scenes:
                    scene_filter_filenames.append(filename)
                file_index += 1
        else:
            scene_filter_filenames = environment_dict['Filename']

        gaze_filter_filenames = []
        if var_right.get() + var_left.get() + var_away.get() + var_direct.get() + var_mainchar.get() + var_everyone.get() + var_everyone_2.get() >= 2:
            file_index = 0
            for filename in gazes_dict['Filename']:
                if var_everyone.get():
                    if var_right.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze == 'right':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_left.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze == 'left':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_direct.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze == 'direct':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_away.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze == 'away':
                                gaze_filter_filenames.append(filename)
                                break
                if var_mainchar.get():
                    MainCharFaceRectangles = main_character_dict['Main Character Face Rectangles'][file_index]
                    MainCharGazes = []

                    FaceRectangles = gazes_dict['Face Rectangles'][file_index]

                    MainCharFaceRectangleIndexes = []
                    for MainCharFaceRectangle in MainCharFaceRectangles:
                        MainCharFaceRectangleIndexes.append(FaceRectangles.index(MainCharFaceRectangle))

                    for MainCharFaceRectangleIndex in MainCharFaceRectangleIndexes:
                        MainCharGazes.append(gazes_dict['Gaze'][file_index][MainCharFaceRectangleIndex])

                    if var_right.get():
                        for gaze in MainCharGazes:
                            if gaze == 'right':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_left.get():
                        for gaze in MainCharGazes:
                            if gaze == 'left':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_direct.get():
                        for gaze in MainCharGazes:
                            if gaze == 'direct':
                                gaze_filter_filenames.append(filename)
                                break
                    if var_away.get():
                        for gaze in MainCharGazes:
                            if gaze == 'away':
                                gaze_filter_filenames.append(filename)
                                break
                if var_everyone_2.get():
                    everyone_gaze_satisfied = True
                    if var_right.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze != 'right':
                                everyone_gaze_satisfied = False
                    if var_left.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze != 'left':
                                everyone_gaze_satisfied = False
                    if var_direct.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze != 'direct':
                                everyone_gaze_satisfied = False
                    if var_away.get():
                        for gaze in gazes_dict['Gaze'][file_index]:
                            if gaze != 'away':
                                everyone_gaze_satisfied = False
                    if len(gazes_dict['Gaze'][file_index]) == 0: everyone_gaze_satisfied = False
                    if everyone_gaze_satisfied:
                        gaze_filter_filenames.append(filename)

                file_index += 1
        else:
            gaze_filter_filenames = gazes_dict['Filename']

        age_filter_filenames = []
        if (var_child.get() + var_youth.get() + var_adult.get() + var_senior.get())>0 and (var_mainchar.get() + var_everyone.get() + var_everyone_2.get()) == 1:
            file_index = 0
            for filename in age_gender_dict['Filename']:
                if var_everyone.get():
                    existing_age_groups = [False, False, False, False] #child, youth, adult, senior
                    for age in age_gender_dict['Age'][file_index]:
                        if 0.1 < age and age < 15: existing_age_groups[0] = True
                        if 15 <= age and age < 25: existing_age_groups[1] = True
                        if 25 <= age and age < 60: existing_age_groups[2] = True
                        if 60 <= age: existing_age_groups[3] = True

                    age_selections_satisfied = True
                    if var_child.get() and existing_age_groups[0] == False: age_selections_satisfied = False
                    if var_youth.get() and existing_age_groups[1] == False: age_selections_satisfied = False
                    if var_adult.get() and existing_age_groups[2] == False: age_selections_satisfied = False
                    if var_senior.get() and existing_age_groups[3] == False: age_selections_satisfied = False

                    if age_selections_satisfied:
                        age_filter_filenames.append(filename)

                if var_mainchar.get():
                    MainCharFaceRectangles = main_character_dict['Main Character Face Rectangles'][file_index]
                    MainCharAges = []
                    FaceRectangles = age_gender_dict['Face Rectangles'][file_index]

                    MainCharFaceRectangleIndexes = []
                    for MainCharFaceRectangle in MainCharFaceRectangles:
                        MainCharFaceRectangleIndexes.append(FaceRectangles.index(MainCharFaceRectangle))

                    for MainCharFaceRectangleIndex in MainCharFaceRectangleIndexes:
                        MainCharAges.append(age_gender_dict['Age'][file_index][MainCharFaceRectangleIndex])

                    age_selections_satisfied = False
                    for MainCharAge in MainCharAges:
                        if 0.1 < MainCharAge and MainCharAge < 15 and var_child.get(): age_selections_satisfied = True
                        if 15 <= MainCharAge and MainCharAge < 25 and var_youth.get(): age_selections_satisfied = True
                        if 25 <= MainCharAge and MainCharAge < 60 and var_adult.get(): age_selections_satisfied = True
                        if 60 <= MainCharAge and var_senior.get(): age_selections_satisfied = True

                    if age_selections_satisfied:
                        age_filter_filenames.append(filename)

                if var_everyone_2.get():
                    everyone_age_selections_satisfied = True #child, youth, adult, senior
                    for age in age_gender_dict['Age'][file_index]:
                        if age == 0: everyone_age_selections_satisfied = False
                        if 0.1 < age and age < 15 and var_child.get() == 0: everyone_age_selections_satisfied = False
                        if 15 <= age and age < 25 and var_youth.get() == 0: everyone_age_selections_satisfied = False
                        if 25 <= age and age < 60 and var_adult.get() == 0: everyone_age_selections_satisfied = False
                        if 60 <= age and var_senior.get() == 0: everyone_age_selections_satisfied = False
                    if len(age_gender_dict['Age'][file_index]) == 0: everyone_age_selections_satisfied = False
                    if everyone_age_selections_satisfied:
                        age_filter_filenames.append(filename)

                file_index += 1
        else:
            age_filter_filenames = age_gender_dict['Filename']

        gender_filter_filenames = []
        if (var_male.get() + var_female.get())>0 and (var_mainchar.get() + var_everyone.get() + var_everyone_2.get()) == 1:
            file_index = 0
            for filename in age_gender_dict['Filename']:
                if var_everyone.get():
                    existing_gender_groups = [False, False] #male, female
                    for gender in age_gender_dict['Gender'][file_index]:
                        if gender == 'M': existing_gender_groups[0] = True
                        if gender == 'F': existing_gender_groups[1] = True

                    gender_selection_satisfied = True
                    if var_male.get() and existing_gender_groups[0] == False: gender_selection_satisfied = False
                    if var_female.get() and existing_gender_groups[1] == False: gender_selection_satisfied = False

                    if gender_selection_satisfied:
                        gender_filter_filenames.append(filename)

                if var_mainchar.get():
                    MainCharFaceRectangles = main_character_dict['Main Character Face Rectangles'][file_index]
                    MainCharGenders = []
                    FaceRectangles = age_gender_dict['Face Rectangles'][file_index]

                    MainCharFaceRectangleIndexes = []
                    for MainCharFaceRectangle in MainCharFaceRectangles:
                        MainCharFaceRectangleIndexes.append(FaceRectangles.index(MainCharFaceRectangle))

                    for MainCharFaceRectangleIndex in MainCharFaceRectangleIndexes:
                        MainCharGenders.append(age_gender_dict['Gender'][file_index][MainCharFaceRectangleIndex])

                    gender_selection_satisfied = False
                    for MainCharGender in MainCharGenders:
                        if MainCharGender == 'M' and var_male.get(): gender_selection_satisfied = True
                        if MainCharGender == 'F' and var_female.get(): gender_selection_satisfied = True


                    if gender_selection_satisfied:
                        gender_filter_filenames.append(filename)

                if var_everyone_2.get():
                    everyone_gender_selection_satisfied = True #male, female
                    for gender in age_gender_dict['Gender'][file_index]:
                        if gender == 'M' and var_female.get() == 1: everyone_gender_selection_satisfied = False
                        if gender == 'F' and var_male.get() == 1: everyone_gender_selection_satisfied = False
                    if len(age_gender_dict['Gender'][file_index]) == 0: everyone_gender_selection_satisfied = False
                    if everyone_gender_selection_satisfied:
                        gender_filter_filenames.append(filename)

                file_index += 1
        else:
            gender_filter_filenames = age_gender_dict['Filename']

        emotion_filter_filenames = []
        if (var_neutral.get() + var_happy.get() + var_sad.get() + var_fear.get() + var_surprise.get() + var_disgust.get() + var_anger.get() + var_contempt.get())>0 and (var_mainchar.get() + var_everyone.get() + var_everyone_2.get()) == 1:
            file_index = 0
            for filename in emotion_dict['Filename']:
                if var_everyone.get():
                    existing_emotion_groups = [False, False, False, False, False, False, False, False] #neutral, happy, sad, fear, surprise, disgust, anger, contempt
                    for emotion in emotion_dict['Emotion'][file_index]:
                        if emotion == 'neutral': existing_emotion_groups[0] = True
                        if emotion == 'happy': existing_emotion_groups[1] = True
                        if emotion == 'sad': existing_emotion_groups[2] = True
                        if emotion == 'fear': existing_emotion_groups[3] = True
                        if emotion == 'surprise': existing_emotion_groups[4] = True
                        if emotion == 'disgust': existing_emotion_groups[5] = True
                        if emotion == 'anger': existing_emotion_groups[6] = True
                        if emotion == 'contempt': existing_emotion_groups[7] = True

                    emotion_selection_satisfied = True
                    if var_neutral.get() and existing_emotion_groups[0] == False: emotion_selection_satisfied = False
                    if var_happy.get() and existing_emotion_groups[1] == False: emotion_selection_satisfied = False
                    if var_sad.get() and existing_emotion_groups[2] == False: emotion_selection_satisfied = False
                    if var_fear.get() and existing_emotion_groups[3] == False: emotion_selection_satisfied = False
                    if var_surprise.get() and existing_emotion_groups[4] == False: emotion_selection_satisfied = False
                    if var_disgust.get() and existing_emotion_groups[5] == False: emotion_selection_satisfied = False
                    if var_anger.get() and existing_emotion_groups[6] == False: emotion_selection_satisfied = False
                    if var_contempt.get() and existing_emotion_groups[7] == False: emotion_selection_satisfied = False

                    if emotion_selection_satisfied:
                        emotion_filter_filenames.append(filename)

                if var_mainchar.get():
                    MainCharFaceRectangles = main_character_dict['Main Character Face Rectangles'][file_index]
                    MainCharEmotions = []
                    FaceRectangles = emotion_dict['Face Rectangles'][file_index]

                    MainCharFaceRectangleIndexes = []
                    for MainCharFaceRectangle in MainCharFaceRectangles:
                        MainCharFaceRectangleIndexes.append(FaceRectangles.index(MainCharFaceRectangle))

                    for MainCharFaceRectangleIndex in MainCharFaceRectangleIndexes:
                        MainCharEmotions.append(emotion_dict['Emotion'][file_index][MainCharFaceRectangleIndex])

                    emotion_selection_satisfied = False
                    for MainCharEmotion in MainCharEmotions:
                        if MainCharEmotion == 'neutral' and var_neutral.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'happy' and var_happy.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'sad' and var_sad.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'fear' and var_fear.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'surprise' and var_surprise.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'disgust' and var_disgust.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'anger' and var_anger.get(): emotion_selection_satisfied = True
                        if MainCharEmotion == 'contempt' and var_contempt.get(): emotion_selection_satisfied = True


                    if emotion_selection_satisfied:
                        emotion_filter_filenames.append(filename)

                if var_everyone_2.get():
                    everyone_emotion_satisfied = True #neutral, happy, sad, fear, surprise, disgust, anger, contempt
                    for emotion in emotion_dict['Emotion'][file_index]:
                        if emotion == 'neutral' and var_neutral.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'happy' and var_happy.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'sad' and var_sad.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'fear' and var_fear.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'surprise' and var_surprise.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'disgust' and var_disgust.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'anger' and var_anger.get() == 0: everyone_emotion_satisfied = False
                        if emotion == 'contempt' and var_contempt.get() == 0: everyone_emotion_satisfied = False
                    if len(emotion_dict['Emotion'][file_index]) == 0: everyone_emotion_satisfied = False
                    if everyone_emotion_satisfied:
                        emotion_filter_filenames.append(filename)

                file_index += 1
        else:
            emotion_filter_filenames = emotion_dict['Filename']

        shape_filter_filenames = []
        if var_square.get() + var_horizontal.get() + var_vertical.get() > 0:
            file_index = 0
            for filename in color_dict['Filename']:
                image_shape = color_dict['Shape'][file_index]
                if var_square.get() == 1 and image_shape == 'square':
                    shape_filter_filenames.append(filename)
                if var_horizontal.get() == 1 and image_shape == 'horizontal':
                    shape_filter_filenames.append(filename)
                if var_vertical.get() == 1 and image_shape == 'vertical':
                    shape_filter_filenames.append(filename)
                file_index += 1
        else:
            shape_filter_filenames = color_dict['Filename']

        color_filter_filenames = []
        if var_color.get() + var_greyscale.get() > 0:
            file_index = 0
            for filename in color_dict['Filename']:
                image_greyscale = color_dict['Greyscale'][file_index]
                if var_greyscale.get() == 1 and image_greyscale == True:
                    color_filter_filenames.append(filename)
                if var_color.get() == 1 and image_greyscale == False:
                    color_filter_filenames.append(filename)
                file_index += 1
        else:
            color_filter_filenames = color_dict['Filename']

        tonality_filter_filenames = []
        if var_dark.get() + var_normal.get() > 0:
            file_index = 0
            for filename in color_dict['Filename']:
                image_dark = color_dict['TonalityDark'][file_index]
                if var_dark.get() == 1 and image_dark == True:
                    tonality_filter_filenames.append(filename)
                if var_normal.get() == 1 and image_dark == False:
                    tonality_filter_filenames.append(filename)
                file_index += 1
        else:
            tonality_filter_filenames = color_dict['Filename']

        dominant_color_filter_filenames = []
        if var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() > 0:
            file_index = 0
            for filename in color_dict['Filename']:
                image_dominant_color_1 = color_dict['DominantColor1'][file_index]
                image_dominant_color_2 = color_dict['DominantColor2'][file_index]
                image_greyscale = color_dict['Greyscale'][file_index]
                if var_red.get() + var_pink.get() + var_orange.get() + var_yellow.get() + var_green.get() + var_green_blue.get() + var_blue.get() + var_purple.get() == 1 and not image_greyscale:
                    if var_red.get() and (image_dominant_color_1 == 'red'): dominant_color_filter_filenames.append(filename)
                    if var_pink.get() and (image_dominant_color_1 == 'pink'): dominant_color_filter_filenames.append(filename)
                    if var_orange.get() and (image_dominant_color_1 == 'orange'): dominant_color_filter_filenames.append(filename)
                    if var_yellow.get() and (image_dominant_color_1 == 'yellow'): dominant_color_filter_filenames.append(filename)
                    if var_green.get() and (image_dominant_color_1 == 'green'): dominant_color_filter_filenames.append(filename)
                    if var_green_blue.get() and (image_dominant_color_1 == 'green-blue'): dominant_color_filter_filenames.append(filename)
                    if var_blue.get() and (image_dominant_color_1 == 'blue'): dominant_color_filter_filenames.append(filename)
                    if var_purple.get() and (image_dominant_color_1 == 'purple'): dominant_color_filter_filenames.append(filename)
                file_index += 1
        else:
            dominant_color_filter_filenames = color_dict['Filename']

        search_result_list = set.intersection(*[set(list) for list in [env_filter_filenames, shot_filter_filenames, person_count_filter_filenames, face_filter_filenames, similar_filter_filenames, description_filter_filenames, date_filter_filenames, dimension_filter_filenames,object_filter_filenames, scene_filter_filenames, gaze_filter_filenames, age_filter_filenames, gender_filter_filenames, emotion_filter_filenames, shape_filter_filenames, color_filter_filenames, tonality_filter_filenames, dominant_color_filter_filenames]])

        if len(search_result_list) > 0:
            os.startfile(os.path.normpath('search_results' + '/' + imagesFolderName))
            time.sleep(0.2)
            hwnd = win32gui.GetForegroundWindow()
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

        for filtered_filename in search_result_list:
            copyfile('images/' + filtered_filename, 'search_results/' + imagesFolderName + '/' + filtered_filename)


if __name__ == "__main__":
    # with open('data/faces.pickle', 'rb') as handle:
    #     global face_dict
    #     face_dict = pickle.load(handle)
    # with open('data/vectors.pickle', 'rb') as handle:
    #     global vectors_dict
    #     vectors_dict = pickle.load(handle)
    # with open('data/main_character.pickle', 'rb') as handle:
    #     global main_character_dict
    #     main_character_dict = pickle.load(handle)
    # with open('data/environments.pickle', 'rb') as handle:
    #     global environment_dict
    #     environment_dict = pickle.load(handle)
    # with open('data/captions.pickle', 'rb') as handle:
    #     global captions_dict
    #     captions_dict = pickle.load(handle)
    # with open('data/objects.pickle', 'rb') as handle:
    #     global objects_dict
    #     objects_dict = pickle.load(handle)
    # with open('data/date.pickle', 'rb') as handle:
    #     global date_dict
    #     date_dict = pickle.load(handle)
    # with open('data/dimensions.pickle', 'rb') as handle:
    #     global dimensions_dict
    #     dimensions_dict = pickle.load(handle)
    # with open('data/gazes.pickle', 'rb') as handle:
    #     global gazes_dict
    #     gazes_dict = pickle.load(handle)
    # with open('data/age_gender.pickle', 'rb') as handle:
    #     global age_gender_dict
    #     age_gender_dict = pickle.load(handle)
    # with open('data/emotions.pickle', 'rb') as handle:
    #     global emotion_dict
    #     emotion_dict = pickle.load(handle)
    # with open('data/colors.pickle', 'rb') as handle:
    #     global color_dict
    #     color_dict = pickle.load(handle)
    #
    # predictor_path = 'shape_predictor_5_face_landmarks.dat'
    # face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    #
    # global detector
    # global sp
    # global facerec
    # detector = dlib.get_frontal_face_detector()
    # sp = dlib.shape_predictor(predictor_path)
    # facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    #
    # global sentence_model
    # sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

    root = tk.Tk()
    app = App(root)
    root.mainloop()


