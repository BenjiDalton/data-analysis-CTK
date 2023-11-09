#----- import python modules -----+
import os
import re
import pandas as pd
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from datetime import date
import shutil

#----- import custom modules -----+
import functions.plotting as plotting
from functions.optionsDictionary import options
from error.customError import customError
plt.style.use('styles/custom_style.mplstyle')

def Run(fileDirectory = None, model: str = None, test: str = None, graphBool: bool = False) -> pd.DataFrame:
	def fillDataframe():
		"""
		Fill dataframe that will be used to create final excel file.
		"""
		for subkey in ['filename info', 'characteristics']:
			for column, value in metaData[subkey].items():
				excelResults.loc[rowIndex, column] = value
		for column, value in results.items():
			excelResults.loc[rowIndex, column] = value
	def saveData():
		"""
		Save data to an excel file. Print file names of successfully analyzed files.
		"""
		excelFilePath = os.path.dirname(file) + f'/results_{date.today()}.xlsx'
		if os.path.isfile(excelFilePath):
			existingData = pd.read_excel(excelFilePath, index_col=0)
			updatedData = pd.concat([existingData, excelResults])
			updatedData.to_excel(excelFilePath, index=True)
			print(f"Data appended to existing file: {excelFilePath}")
		else:
			print(f"New file created: {excelFilePath}")
			excelResults.to_excel(excelFilePath, index = True)

	allFiles = []
	successfulFiles = []
	fileLimit: int = 1000
	excelResults = pd.DataFrame()
	
	if isinstance(fileDirectory, str):
		for root, subdirs, files in os.walk(fileDirectory):
			for file in files:
				if file.lower().__contains__('store') or file.lower().__contains__('directory'):
					continue
				allFiles.append(os.path.join(root, file))
	if isinstance(fileDirectory, list):
		allFiles = fileDirectory

	
	for idx, file in enumerate(allFiles):
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
				if test == "Binta" or test == "Makenna":
					try:
						# rowIndex = metaData['filename info']['animal'] + '_' + metaData['filename info']['fibre']
						rowIndex = metaData['filename info']['full filename']
						fillDataframe()
					except Exception as error:
						customError(f"An error occurred: {str(error)} in {file}.\nChoose excel file to save completed results to.")
						saveData()
						break
				else:
					try:
						rowIndex = metaData['filename info']['full filename']
						fillDataframe()
					except Exception as error:
						customError(f"An error occurred: {str(error)} in {file}.\nChoose excel file to save completed results to.")
						saveData()
						break

				if graphBool == True:
					# plotting.show(
					# 	xlabel = chosenOption['graphing']['x label'], 
					# 	ylabel = chosenOption['graphing']['y label']
					# )
					plt.show()
				plt.close()
				successfulFiles.append(file)

				#----- move analyzed files to a seperate folder -----+
				if file in successfulFiles:
					try:
						destinationFolder = os.path.join(os.path.dirname(file), "completed")
						if not os.path.exists(destinationFolder):
							os.makedirs(destinationFolder)
						destinationPath = os.path.join(destinationFolder, os.path.basename(file))
						try: 
							shutil.move(file, destinationPath)
						except OSError as error:
							print(f"file: {file} not moved to 'completed' folder.")
							print(error)
							continue
					except Exception as error:
						customError(f"Error occurred while moving the file {file} to the completed folder: {error}")
	
				if idx > fileLimit:
					print(f"Analysis on a subset of {fileLimit} files completed.")
					saveData()
					break
				
	try:
		saveData()
	except OSError:
		excelFilePath = filedialog.asksaveasfilename(defaultextension="xlsx")
		excelResults.to_excel(excelFilePath, index = True)
	print("completed successfully")

	return excelResults
