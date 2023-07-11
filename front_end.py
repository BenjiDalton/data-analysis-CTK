#----- import python modules -----+
import os
import numpy as np
import pandas as pd
from tkinter import filedialog, colorchooser
import customtkinter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#----- import custom modules -----+
import Plotting, ui
import Analysis_via_options as Analysis
from classes import Colors
from OptionsDictionary import options
plt.style.use('custom_style.mplstyle')

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #----- configure window -----+
        self.title("Neuromechanical Performance Research Lab")
        self.geometry(f"{ui.app_width}x{ui.app_height}")
        self.update()

        self.first_row_frame = ui.Frame(
            master = self)
        self.second_row_frame = ui.Frame(
            master = self)
        self.second_row_frame.grid_rowconfigure(0, weight=1)
        self.tableFrame = ui.Frame(
            master = self,
            height = 1100,
            anchor = 's')
        
        self.settings_button = ui.Button(
            master=self.first_row_frame,
            text='Settings',
            command=self.open_toplevel)
        self.FileDialogButton = ui.Button(
            master = self.first_row_frame, 
            text = 'Choose Files', 
            command = self.open_file_dialog)
        self.ChooseModel = ui.DropDownMenu(
            master = self.first_row_frame, 
            values = ui.models, 
            command = self.update_model)
        self.ChooseTest = ui.DropDownMenu(
            master = self.first_row_frame, 
            values = ui.singlefibre_tests, 
            width = 160)
        self.ChooseTimepoint = ui.DropDownMenu(
                master = self.first_row_frame,
                values = ['D40', 'D80', 'D120', 'D176'],
                width = 80,
                visible=False)
        
        self.ExportButton = ui.Button(
            master = self.first_row_frame, 
            text = 'Export', 
            command = self.ExportData, 
            side = 'right', 
            anchor = 'ne')
        self.RunButton = ui.Button(
            master = self.first_row_frame, 
            text = 'Run', 
            command = self.RunStuff, 
            side = 'right', 
            anchor = 'ne')

        self.graphbool = False
        self.GraphCheckbox = customtkinter.CTkCheckBox(
            master = self.first_row_frame,
            text = "Graph?",
            border_color=Colors.AuroraColor,
            hover_color=Colors.AuroraColor,
            fg_color=Colors.AuroraColor,
            command = self.update_checkboxvalue)
        self.GraphCheckbox.pack(
            padx = 0,
            pady = ui.button_padding,
            side = 'right',
            anchor = 'ne')

        makenna_testing_protocols = ['ISO', 'CONCENTRIC', 'SSC']
        self.choose_protocol = ui.DropDownMenu(
            master = self.second_row_frame,
            values = makenna_testing_protocols,
            width = 160,
            visible=False,
            command = self.update_protocol)
        self.new_dropdown = ui.DropDownMenu(
            master = self.second_row_frame,
            values = options[self.ChooseModel.get()][self.choose_protocol.get()]['variables'],
            width = 160,
            visible=False)
        self.plotmeans_button = ui.Button(
            master = self.second_row_frame,
            text = "Graph means",
            visible=False,
            command = self.plot_means)

        self.table = ui.Table(
            master = self.tableFrame, 
            orientation = "horzitonal",
            width = 1200, 
            height = ui.app_height, 
            label_text = 'Data')

        self.toplevel_window = None

    #----- update UI elements -----+
    def change_text_color(self, choice):
        color = Colors.Black if choice == 'Black' else Colors.White
        for frame in self.first_row_frame, self.second_row_frame:
            for widget in frame.winfo_children():
                widget.configure(text_color = color)
        self.update()

    def change_ui_elements_color(self):
        color = colorchooser.askcolor()
        for frame in self.first_row_frame, self.second_row_frame:
            for widget in frame.winfo_children():
                for attribute in ['fg_color', 'button_color', 'border_color']:
                    try:
                        widget.cget(attribute)
                    except ValueError:
                        continue
                    widget.configure(**{attribute: color[1]})
                
                for attribute in ['hover_color', 'button_hover_color']:
                    try:
                        dark_color = tuple((val * 0.7)/255 for val in color[0])
                        widget.configure(**{attribute: mcolors.rgb2hex(dark_color)})
                    except ValueError:
                        continue
        self.update()

    def update_checkboxvalue(self):
        if self.graphbool == False:
            self.graphbool = True
        else:
            self.graphbool = False
        
    def update_model(self, choice):
        options_dict = {
            'Single Fibre': {
                'values': ui.singlefibre_tests,
                'width': 160
                },
            'In Vivo': {
                'values': ['Torque-Frequency', 'Torque-Velocity', 'Fatigue', 'Recovery'],
                'width': 160,
                'dropdown_y': 140,
                'timepoint_values': ['D40', 'D80', 'D120', 'D176'],
                'timepoint_width': 80
                }
            }

        model = self.ChooseModel.get()
        current_options = options_dict[model]
        self.ChooseTest.configure(values = current_options['values'])
        self.ChooseTest.set(current_options['values'][0])

        if model == 'Single Fibre' and self.ChooseTimepoint != None:
            self.ChooseTimepoint.destroy()

        if model == 'In Vivo':
            self.ChooseTimepoint.pack(
                padx=ui.button_padding,
                pady=ui.button_padding,
                anchor='center',
                side='left')

    def update_protocol(self, choice):
        self.DrawTable()
        self.update()

    def create_second_row(self):
        for idx, widget in enumerate(self.second_row_frame.winfo_children()):
            widget.grid(row=0, column=idx, padx=10, pady=10)
        self.update()
    
    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ui.ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()

    def DrawTable(self):
        self.create_second_row()
        chosen_option = options[self.ChooseModel.get()][self.choose_protocol.get()]
        self.new_dropdown.configure(values = chosen_option['variables'])
        self.new_dropdown.set(chosen_option['variables'][0]) 
            
        self.table.update_dataframe(dataframe=self.all_results)

        self.table.pack(
            padx = ui.button_padding,
            fill = 'both')
        self.update()
      
    #----- working with files -----+
    def open_file_dialog(self):
        # if self.excelbool == False:
            Directory = filedialog.askdirectory()

            AllFiles = []
            for root, subdirs, files in os.walk(Directory):
                for file in files:
                    if file.lower().__contains__('store') or file.lower().__contains__('directory'):
                        continue
                    if file.startswith('.'):
                        continue

                    AllFiles.append(os.path.join(root, file))
            self.AllFiles = sorted(AllFiles)
 
    def RunStuff(self):
        if self.ChooseTimepoint:
            time = self.ChooseTimepoint.get()
        else:
            time = 'N/A'
            
        self.all_results = Analysis.Run(
            Directory = self.AllFiles,
            model = self.ChooseModel.get(), 
            test = self.ChooseTest.get(), 
            Graph = self.graphbool)

        self.DrawTable()
    
    def ExportData(self):
        export_filename = filedialog.asksaveasfilename(defaultextension="xlsx")
        self.all_results.to_excel(str(export_filename), index = False)
        
    def open_excelfile(self):
        file = filedialog.askopenfile()
        self.all_results = pd.read_excel(file.name)
        self.DrawTable()

    #----- plot stuff -----+
    def plot_means(self):
        variable = self.new_dropdown.get()

        chosen_option = options[self.ChooseModel.get()][self.choose_protocol.get()]

        matched_columns = [
            col for col in self.all_results.columns 
            if chosen_option["col basenames"][chosen_option["variables"].index(variable)].format('') in col]

        for idx, column in enumerate(matched_columns):
            x_val = column
            if variable == 'ISO Force' or variable == 'ISO Stiffness':
                x_val = column.split(' ')[3]
            if ' - ' in column:
                x_val = '-'.join([column.split(' ')[0], column.split(' ')[2]])
            if self.choose_protocol.get() == 'CONCENTRIC':
                x_val = column.split(' ')[2]
            
            plt.bar(
                x_val, 
                np.mean(self.all_results[column]), 
                yerr=np.std(self.all_results[column]),
                edgecolor=Plotting.edge_rgb,
                color=Plotting.hex_to_rgb(Colors.bar_graph_colors[idx], 0.5))

            for index, row in self.all_results.iterrows():
                animal_id = row['Animal']
                plt.plot(x_val, row[column], marker=r"${}$".format(animal_id), markerfacecolor=Colors.Black)
            
        try:
            plt.xlabel(chosen_option["graphing"]["x label"], fontsize=16)
            plt.ylabel(chosen_option["graphing"]["y label"], fontsize=16)
        except:
            pass
        plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()

