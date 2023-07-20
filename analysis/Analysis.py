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
				
				chosenOption = options[model][test]
				data, metaData = options[model]['read file'](file, model, protocol)
				results = chosenOption['analyze'](data, metaData, graph)

				chosenOption = options[model][protocol]
				if test == "Binta" or test == "Makenna":
					try:
						rowIndex = metaData['filename info']['animal'] + '_' + metaData['filename info']['fibre']
						for subkey in ['filename info', 'characteristics']:
							for column, value in metaData[subkey].items():
								excelResults.loc[rowIndex, column] = value
						for column, value in results.items():
							excelResults.loc[rowIndex, column] = value
					except Exception as e:
						print(f"An error occurred: {str(e)} in {file}")
						excelFilename = filedialog.asksaveasfilename(defaultextension='xlsx')
						excelResults.to_excel(excelFilename, index = True)
					continue

				# data = options[model]['fill results'](
				# 	model, 
				# 	results,
				# 	metaData, 
				# 	chosenOption['col basenames'], 
				# 	chosenOption['substring']
				# )

				if graphBool == True:
					# plotting.show(
					# 	xlabel = chosenOption['graphing']['x label'], 
					# 	ylabel = chosenOption['graphing']['y label']
					# )
					plt.show()
				plt.close()

	# try:
	# 	excelFilename = filedialog.asksaveasfilename(defaultextension='xlsx')
	# 	excelResults.to_excel(excelFilename, index = True)
	# except:
	# 	excelResults.to_excel(os.path.dirname(file) + f'/results{date.today()}.xlsx', index = True)
	excelResults.to_excel(os.path.dirname(file) + f'/Binta_results{date.today()}.xlsx', index = True)
	print("completed successfully")
	exit()
	sorted_columns = sorted(data.columns, key = lambda column: float(re.search(r'pCa (\d+\.\d+)', column).group(1) if 'pCa' in column else np.nan))
	
	data = data.reindex(columns = sorted_columns)
	
	results = pd.DataFrame(data = data, columns = sorted_columns)

	return results

	# results.to_excel('', index = False)
