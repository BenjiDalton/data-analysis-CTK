#----- import python modules -----+
from tkinter import ttk
import customtkinter
import pandas as pd
import matplotlib.colors as mcolors

#----- import custom modules -----+
import frontEnd
from classes import Colors

app_width = 1400
app_height = 700
app_bg_color = Colors.DarkGray
button_color = Colors.AuroraColor
dark_color = tuple(val * 0.7 for val in mcolors.hex2color(button_color))
button_hover_color = mcolors.rgb2hex(dark_color)
ui_frame_height = 50
button_width = 100

button_padding = 20


models = ['Single Fibre', 'In Vivo']
invivo_tests = ['Torque-Frequency', 'Torque-Velocity', 'Fatigue', 'Recovery', 'All']
singlefibre_tests = ['pCa', 'ktr', 'Power', 'rFE', 'Binta', 'Makenna']

class Button(customtkinter.CTkButton):
    def __init__(
        self, 
        master: customtkinter.CTkFrame = None, 
        state: str = 'normal', 
        width: int = button_width, 
        fg_color = button_color,
        hover_color = button_hover_color,
        text_color: Colors = Colors.White,
        padx: int = button_padding, 
        pady: int = button_padding, 
        text: str = '', 
        anchor: str = 'n', 
        side = 'left', 
        command = None,
        visible: bool = True,
        *args, **kwargs):
        super().__init__(
            *args, 
            master = master, 
            state = state, 
            width = width, 
            fg_color = fg_color,
            text_color=text_color,
            hover_color=hover_color,
            text = text, 
            command = command,
            *args, **kwargs)
        
        if visible==True:
            self.pack(in_ = master, padx = padx, pady = pady, anchor = anchor, side = side)

class DropDownMenu(customtkinter.CTkOptionMenu):
    def __init__(
        self, *args,
        master: customtkinter.CTkFrame, 
        width: int = 140,
        fg_color: Colors = button_color, 
        button_color: Colors = button_color,
        hover_color = button_hover_color,
        text_color: Colors = Colors.White,
        values: list = None, 
        state: str = "normal", 
        hover: bool = True, 
        command = None,
        anchor: str = "center",
        pack_side: str = "left",
        visible: bool=True,
        **kwargs):
        super().__init__(
            *args, 
            master = master, 
            width = width, 
            fg_color = fg_color, 
            button_color=button_color,
            button_hover_color=hover_color,
            text_color=text_color,
            values = values, 
            state = state, 
            hover = hover, 
            command = command,
            anchor = anchor,
            **kwargs)

        if visible==True:
            self.pack(in_ = master, padx = 20, pady = 20, anchor = anchor, side = pack_side)

class Frame(customtkinter.CTkFrame):
    def __init__(
        self, 
        master: any, 
        width: int = app_width,
        height: int = 100, 
        corner_radius: int = 0,
        fg_color: Colors = Colors.DarkGray,
        fill = 'both',
        anchor = 'n',
        *args, **kwargs):
        super().__init__(
            master = master, 
            width = width, 
            height = height, 
            corner_radius = corner_radius, 
            fg_color = fg_color,
            *args, **kwargs)

        self.pack(in_ = master, anchor = anchor, fill = fill)

class Table(customtkinter.CTkScrollableFrame):
    def __init__(self, master = None, dataframe: pd.DataFrame = None, orientation='horizontal', **kwargs):
        super().__init__(master, **kwargs)
        self.dataframe: pd.DataFrame = dataframe
        self.table = ttk.Treeview(self, show = 'headings', height = 27)
        if self.dataframe:
            self.table["columns"] = list(self.dataframe.columns)
        
            self.update_dataframe(dataframe=self.dataframe)
        

    def update_dataframe(self, dataframe: pd.DataFrame):
        for child in self.table.get_children():
            self.table.delete(child)

        self.columns = list(dataframe.columns)
        self.table["columns"] = list(dataframe.columns)

        for column in self.table["columns"]:
            for level in column:
                for index, row in dataframe.iterrows():
                    self.table.heading(column = column, text=level)

            # min_columnwidth = 80 if len(column) < 10 else 200
            min_columnwidth = 8 * len(column) if 8 * len(column) >= 60 else 60
            self.table.column(column, minwidth = min_columnwidth, stretch=True)
            self.table.heading(column, text = column)
            self.table.tag_configure(column, anchor = "center")

        for index, row in dataframe.iterrows():
            try:
                self.table.insert("", "end", values=[('{:.3f}'.format(float(v)) if isinstance(v, float) else str(v).upper()) for v in row])
            except:
                self.table.insert("", "end", values = list(row))
        
        self.table.pack()

class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, app: frontEnd.App, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("200x180")
        self.title('Settings Menu')
        self.grid_columnconfigure(0, weight=1)

        self.colorChooser = Button(
            master=self,
            text="Change Button Color",
            command=app.changeUIColor)

        self.colorChooser.grid(row=0, column=0, pady=20)

        self.textColorLabel = customtkinter.CTkLabel(
            master=self,
            text = 'Text Color')

        self.textColorChooser = customtkinter.CTkSegmentedButton(
            master=self,
            values=['White', 'Black'],
            fg_color=Colors.AuroraColor,
            command=app.change_text_color)
        self.textColorChooser.set('White')
        self.textColorLabel.grid(row=1, column=0)
        self.textColorChooser.grid(row=2, column = 0, pady=0)
