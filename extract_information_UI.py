import tkinter as tk
import tkinter.font as tkFont
from tkinter.filedialog import askdirectory
import os
import extract_information

class App:
    def __init__(self, root):
        #setting title
        root.title("Extract Image Information")
        #setting window size
        width=150
        height=440
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        global GLabel_476
        GLabel_476 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        GLabel_476["font"] = ft
        GLabel_476["fg"] = "#29872d"
        GLabel_476["justify"] = "left"
        GLabel_476["text"] = ""
        GLabel_476.place(x=1, y=55, width=149, height=25)

        GButton_558=tk.Button(root)
        GButton_558["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_558["font"] = ft
        GButton_558["fg"] = "#000000"
        GButton_558["justify"] = "center"
        GButton_558["text"] = "Choose Folder"
        GButton_558.place(x=35,y=30,width=90,height=25)
        GButton_558["command"] = self.GButton_558_command

        Face=tk.Button(root)
        Face["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Face["font"] = ft
        Face["fg"] = "#000000"
        Face["justify"] = "center"
        Face["text"] = "Face"
        Face.place(x=35,y=85,width=90,height=25)
        Face["command"] = self.Face_command

        Similarity=tk.Button(root)
        Similarity["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Similarity["font"] = ft
        Similarity["fg"] = "#000000"
        Similarity["justify"] = "center"
        Similarity["text"] = "Similarity"
        Similarity.place(x=35,y=110,width=90,height=25)
        Similarity["command"] = self.Similarity_command

        Gazes=tk.Button(root)
        Gazes["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Gazes["font"] = ft
        Gazes["fg"] = "#000000"
        Gazes["justify"] = "center"
        Gazes["text"] = "Gazes"
        Gazes.place(x=35,y=135,width=90,height=25)
        Gazes["command"] = self.Gazes_command

        Captions=tk.Button(root)
        Captions["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Captions["font"] = ft
        Captions["fg"] = "#000000"
        Captions["justify"] = "center"
        Captions["text"] = "Captions"
        Captions.place(x=35,y=160,width=90,height=25)
        Captions["command"] = self.Captions_command

        Colors=tk.Button(root)
        Colors["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Colors["font"] = ft
        Colors["fg"] = "#000000"
        Colors["justify"] = "center"
        Colors["text"] = "Colors"
        Colors.place(x=35,y=185,width=90,height=25)
        Colors["command"] = self.Colors_command

        AgeGender=tk.Button(root)
        AgeGender["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        AgeGender["font"] = ft
        AgeGender["fg"] = "#000000"
        AgeGender["justify"] = "center"
        AgeGender["text"] = "Age-Gender"
        AgeGender.place(x=35,y=210,width=90,height=25)
        AgeGender["command"] = self.AgeGender_command

        Date=tk.Button(root)
        Date["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Date["font"] = ft
        Date["fg"] = "#000000"
        Date["justify"] = "center"
        Date["text"] = "Date"
        Date.place(x=35,y=235,width=90,height=25)
        Date["command"] = self.Date_command

        Dimensions=tk.Button(root)
        Dimensions["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Dimensions["font"] = ft
        Dimensions["fg"] = "#000000"
        Dimensions["justify"] = "center"
        Dimensions["text"] = "Dimensions"
        Dimensions.place(x=35,y=260,width=90,height=25)
        Dimensions["command"] = self.Dimensions_command

        Emotions=tk.Button(root)
        Emotions["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Emotions["font"] = ft
        Emotions["fg"] = "#000000"
        Emotions["justify"] = "center"
        Emotions["text"] = "Emotions"
        Emotions.place(x=35,y=285,width=90,height=25)
        Emotions["command"] = self.Emotions_command

        Environments=tk.Button(root)
        Environments["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Environments["font"] = ft
        Environments["fg"] = "#000000"
        Environments["justify"] = "center"
        Environments["text"] = "Environments"
        Environments.place(x=35,y=310,width=90,height=25)
        Environments["command"] = self.Environments_command

        Objects=tk.Button(root)
        Objects["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        Objects["font"] = ft
        Objects["fg"] = "#000000"
        Objects["justify"] = "center"
        Objects["text"] = "Objects"
        Objects.place(x=35,y=335,width=90,height=25)
        Objects["command"] = self.Objects_command

        MainCharacters=tk.Button(root)
        MainCharacters["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        MainCharacters["font"] = ft
        MainCharacters["fg"] = "#000000"
        MainCharacters["justify"] = "center"
        MainCharacters["text"] = "Main Characters"
        MainCharacters.place(x=35,y=360,width=90,height=25)
        MainCharacters["command"] = self.MainCharacters_command

        GLabel_71=tk.Label(root)
        GLabel_71["activebackground"] = "#ee2f2f"
        ft = tkFont.Font(family='Times',size=12)
        GLabel_71["font"] = ft
        GLabel_71["fg"] = "#333333"
        GLabel_71["justify"] = "center"
        GLabel_71["text"] = "Extract Information"
        GLabel_71.place(x=20,y=1,width=120,height=30)

        GButton_741=tk.Button(root)
        GButton_741["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_741["font"] = ft
        GButton_741["fg"] = "#000000"
        GButton_741["justify"] = "center"
        GButton_741["text"] = "Extract All"
        GButton_741.place(x=45,y=400,width=70,height=25)
        GButton_741["command"] = self.GButton_741_command

    def GButton_741_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "all")

    def Face_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "face")

    def Similarity_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "similarity")

    def Gazes_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "gazes")

    def Captions_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "captions")

    def Colors_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "colors")

    def AgeGender_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "age_gender")

    def Date_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "date")

    def Dimensions_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "dimensions")

    def Emotions_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "emotions")

    def Environments_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "scenes")

    def Objects_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "objects")

    def MainCharacters_command(self):
        global imagesFolder
        extract_information_csv.main(imagesFolder, "main_characters")

    def GButton_558_command(self):
        global imagesFolder
        imagesFolder = askdirectory()
        images_folder_name = imagesFolder.split('/')[-1]
        GLabel_476["text"] = images_folder_name
        if not os.path.isdir(imagesFolder):
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
