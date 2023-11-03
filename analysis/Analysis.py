#----- import python modules -----+
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from datetime import date

#----- import custom modules -----+
import functions.plotting as plotting
from functions.optionsDictionary import options
plt.style.use('styles/custom_style.mplstyle')

def Run(fileDirectory = None, model: str = None, test: str = None, graphBool: bool = False) -> pd.DataFrame:

	allFiles = []
	if isinstance(fileDirectory, str):
		for root, subdirs, files in os.walk(fileDirectory):
			for file in files:
				if file.lower().__contains__('store') or file.lower().__contains__('directory'):
					continue
				allFiles.append(os.path.join(root, file))
	if isinstance(fileDirectory, list):
		allFiles = fileDirectory

	excelResults = pd.DataFrame()
	for file in allFiles:
		graph = None
		if graphBool == True and test != 'Binta' and test != 'Makenna':
			fig, graph = plotting.initializeFig()
		for protocol in options[model].keys():
			if protocol.upper() in os.path.basename(file).upper():
				
				#----- User inputs ------+
				chosenOption = options[model][test] 
				#----- Read text file and grab info based on protocol selected by user ------+
				data, metaData = options[model]['read file'](file, model, protocol)
				#----- Analyze data based on protocol selected by user ------+
				results = chosenOption['analyze'](data, metaData, graph)

				chosenOption = options[model][protocol]

				#----- Fill dataframe with results ------+
				#----- Sent to front end to fill table with data and let user save results to excel file ------+
				try:
					rowIndex = metaData['filename info']['full filename']
					for subkey in ['filename info', 'characteristics']:
						for column, value in metaData[subkey].items():
							excelResults.loc[rowIndex, column] = value
					for column, value in results.items():
						excelResults.loc[rowIndex, column] = value
				except Exception as e:
					print(f"An error occurred: {str(e)} in {file}")
					excelFilename = filedialog.asksaveasfilename(defaultextension='xlsx')
					excelResults.to_excel(excelFilename, index = True)

				if graphBool == True:
					# plotting.show(
					# 	xlabel = chosenOption['graphing']['x label'], 
					# 	ylabel = chosenOption['graphing']['y label']
					# )
					plt.show()
				plt.close()

	print("completed successfully")

	return excelResults
