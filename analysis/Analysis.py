#----- import python modules -----+
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#----- import custom modules -----+
import functions.plotting as plotting
from classes.classes import FileInfo
from functions.optionsDictionary import options
plt.style.use('styles/custom_style.mplstyle')

def Run(fileDirectory = None, model: str = None, test: str = None, graphBool: bool = False) -> pd.DataFrame:

	allFiles = []
	if isinstance(fileDirectory, str):
		for root, subdirs, files in os.walk(fileDirectory):
			for file in files:
				if file.lower().__contains__('store') or file.lower().__contains__('fileDirectory'):
					continue
				allFiles.append(os.path.join(root, file))
	if isinstance(fileDirectory, list):
		allFiles = fileDirectory

	excel_results = pd.DataFrame()
	for file in allFiles:
		graph = None
		if graphBool == True:
			fig, graph = plotting.initialize_fig()
		for protocol in options[model].keys():
			if protocol.upper() in os.path.basename(file).upper():
				
				chosenOption = options[model][test]
				data, metaData  =  options[model]['read file'](
					file = file, 
					model = model, 
					test = protocol
				)
				results = chosenOption['analyze'](
					data, 
					metaData, 
					graph
				)

				chosenOption = options[model][protocol]
				if test == "Binta" or test == "Makenna":
					try:
						rowIndex = metaData['filename info']['animal'] + '_' + metaData['filename info']['fibre']
						for subkey in ['filename info', 'characteristics']:
							for key, value in metaData[subkey].items():
								excel_results.loc[rowIndex, key] = value
						for column in results.keys():
							excel_results.loc[rowIndex, column] = results[column]
					except:
						excel_results.to_excel('/Volumes/Lexar/Makenna/Data_July17.xlsx', index = True)
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
	excel_results.to_excel('/Volumes/Lexar/Makenna/Data_July17.xlsx', index = True)
	print("completed successfully")
	exit()
	sorted_columns = sorted(data.columns, key = lambda column: float(re.search(r'pCa (\d+\.\d+)', column).group(1) if 'pCa' in column else np.nan))
	
	data = data.reindex(columns = sorted_columns)
	
	results = pd.DataFrame(data = data, columns = sorted_columns)

	return results

	# results.to_excel('', index = False)
