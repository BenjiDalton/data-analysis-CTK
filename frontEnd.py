#----- import python modules -----+
import os
import numpy as np
import pandas as pd
from tkinter import filedialog, colorchooser
import customtkinter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#----- import custom modules -----+
import plotting, ui
import Analysis as Analysis
from classes import Colors
from optionsDictionary import options
plt.style.use('custom_style.mplstyle')

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #----- configure window -----+
        self.title("Neuromechanical Performance Research Lab")
        self.geometry(f"{ui.app_width}x{ui.app_height}")
        self.update()

        self.firstRowFrame=ui.Frame(
            master=self)
        self.secondRowFrame=ui.Frame(
            master=self)
        self.secondRowFrame.grid_rowconfigure(0, weight=1)
        self.tableFrame=ui.Frame(
            master=self,
            height=1100,
            anchor='s')
        
        self.settingsButton=ui.Button(
            master=self.firstRowFrame,
            text='Settings',
            command=self.openTopLevel)
        self.fileDialogButton=ui.Button(
            master=self.firstRowFrame, 
            text='Choose Files', 
            command=self.openFileDialog)
        self.chooseModel=ui.DropDownMenu(
            master=self.firstRowFrame, 
            values=ui.models, 
            command=self.updateModel)
        self.chooseTest=ui.DropDownMenu(
            master=self.firstRowFrame, 
            values=ui.singlefibre_tests, 
            width=160)
        self.chooseTimepoint=ui.DropDownMenu(
                master=self.firstRowFrame,
                values=['D40', 'D80', 'D120', 'D176'],
                width=80,
                visible=False)
        
        self.exportButton=ui.Button(
            master=self.firstRowFrame, 
            text='Export', 
            command=self.exportData, 
            side='right', 
            anchor='ne')
        self.runButton=ui.Button(
            master=self.firstRowFrame, 
            text='Run', 
            command=self.runFiles, 
            side='right', 
            anchor='ne')

        self.graphBool=False
        self.graphCheckbox=customtkinter.CTkCheckBox(
            master=self.firstRowFrame,
            text="Graph?",
            border_color=Colors.AuroraColor,
            hover_color=Colors.AuroraColor,
            fg_color=Colors.AuroraColor,
            command=self.updateCheckboxValue)
        self.graphCheckbox.pack(
            padx=0,
            pady=ui.button_padding,
            side='right',
            anchor='ne')

        makennaProtocols=['ISO', 'CONCENTRIC', 'SSC']
        self.chooseProtocol=ui.DropDownMenu(
            master=self.secondRowFrame,
            values=makennaProtocols,
            width=160,
            visible=False,
            command=self.updateProtocol)
        self.newDropdown=ui.DropDownMenu(
            master=self.secondRowFrame,
            values=options[self.chooseModel.get()][self.chooseProtocol.get()]['variables'],
            width=160,
            visible=False)
        self.plotmeansButton=ui.Button(
            master=self.secondRowFrame,
            text="Graph means",
            visible=False,
            command=self.plotMeans)

        self.table=ui.Table(
            master=self.tableFrame, 
            orientation="horzitonal",
            width=1200, 
            height=ui.app_height, 
            label_text='Data')

        self.topLevelWindow=None

    #----- update UI elements -----+
    def changeTextColor(self, choice):
        color=Colors.Black if choice == 'Black' else Colors.White
        for frame in self.firstRowFrame, self.secondRowFrame:
            for widget in frame.winfo_children():
                widget.configure(text_color=color)
        self.update()

    def changeUIColor(self):
        color=colorchooser.askcolor()
        for frame in self.firstRowFrame, self.secondRowFrame:
            for widget in frame.winfo_children():
                for attribute in ['fg_color', 'button_color', 'border_color']:
                    try:
                        widget.cget(attribute)
                    except ValueError:
                        continue
                    widget.configure(**{attribute: color[1]})
                
                for attribute in ['hover_color', 'button_hover_color']:
                    try:
                        darkColor=tuple((val * 0.7)/255 for val in color[0])
                        widget.configure(**{attribute: mcolors.rgb2hex(darkColor)})
                    except ValueError:
                        continue
        self.update()

    def updateCheckboxValue(self):
        if self.graphBool == False:
            self.graphBool=True
        else:
            self.graphBool=False
        
    def updateModel(self, choice):
        optionsDict={
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

        model=self.chooseModel.get()
        chosenOption=optionsDict[model]
        self.chooseTest.configure(values=chosenOption['values'])
        self.chooseTest.set(chosenOption['values'][0])

        if model == 'Single Fibre' and self.chooseTimepoint != None:
            self.chooseTimepoint.destroy()

        if model == 'In Vivo':
            self.chooseTimepoint.pack(
                padx=ui.button_padding,
                pady=ui.button_padding,
                anchor='center',
                side='left')

    def updateProtocol(self, choice):
        self.drawTable()
        self.update()

    def createSecondRow(self):
        for idx, widget in enumerate(self.secondRowFrame.winfo_children()):
            widget.grid(row=0, column=idx, padx=10, pady=10)
        self.update()
    
    def openTopLevel(self):
        if self.topLevelWindow is None or not self.topLevelWindow.winfo_exists():
            self.topLevelWindow=ui.ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.topLevelWindow.focus()

    def drawTable(self):
        self.createSecondRow()
        chosenOption=options[self.chooseModel.get()][self.chooseProtocol.get()]
        self.newDropdown.configure(values=chosenOption['variables'])
        self.newDropdown.set(chosenOption['variables'][0]) 
            
        self.table.update_dataframe(dataframe=self.allResults)

        self.table.pack(
            padx=ui.button_padding,
            fill='both')
        self.update()
      
    #----- working with files -----+
    def openFileDialog(self):
        # if self.excelbool == False:
            fileDirectory=filedialog.askdirectory()

            allFiles=[]
            for root, subdirs, files in os.walk(fileDirectory):
                for file in files:
                    if file.lower().__contains__('store') or file.lower().__contains__('fileDirectory'):
                        continue
                    if file.startswith('.'):
                        continue

                    allFiles.append(os.path.join(root, file))
            self._allFiles=sorted(allFiles)
 
    def runFiles(self):
        if self.chooseTimepoint:
            time=self.chooseTimepoint.get()
        else:
            time='N/A'
            
        self.allResults=Analysis.Run(
            fileDirectory=self._allFiles,
            model=self.chooseModel.get(), 
            test=self.chooseTest.get(), 
            graphBool=self.graphBool)

        self.drawTable()
    
    def exportData(self):
        exportFilename=filedialog.asksaveasfilename(defaultextension="xlsx")
        self.allResults.to_excel(str(exportFilename), index=False)
        
    def openExcelFile(self):
        file=filedialog.askopenfile()
        self.allResults=pd.read_excel(file.name)
        self.drawTable()

    #----- plot stuff -----+
    def plotMeans(self):
        variable=self.newDropdown.get()

        chosenOption=options[self.chooseModel.get()][self.chooseProtocol.get()]

        matchedColumns=[
            col for col in self.allResults.columns 
            if chosenOption["col basenames"][chosenOption["variables"].index(variable)].format('') in col]

        for idx, column in enumerate(matchedColumns):
            xData=column
            if variable == 'ISO Force' or variable == 'ISO Stiffness':
                xData=column.split(' ')[3]
            if ' - ' in column:
                xData='-'.join([column.split(' ')[0], column.split(' ')[2]])
            if self.chooseProtocol.get() == 'CONCENTRIC':
                xData=column.split(' ')[2]
            
            plt.bar(
                xData, 
                np.mean(self.allResults[column]), 
                yerr=np.std(self.allResults[column]),
                edgecolor=plotting.edgeRGB,
                color=plotting.hex_to_rgb(Colors.barColors[idx], 0.5))

            for index, row in self.allResults.iterrows():
                animal_id=row['Animal']
                plt.plot(xData, row[column], marker=r"${}$".format(animal_id), markerfacecolor=Colors.Black)
            
        try:
            plt.xlabel(chosenOption["graphing"]["x label"], fontsize=16)
            plt.ylabel(chosenOption["graphing"]["y label"], fontsize=16)
        except:
            pass
        plt.show()

if __name__ == "__main__":
    app=App()
    app.mainloop()

