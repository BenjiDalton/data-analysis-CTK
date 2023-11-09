#----- import python modules -----+
import os
import re
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution

#----- import custom modules -----+
import functions.plotting as plotting
from classes.classes import Colors

msToSeconds = 1000

class Error(Exception):
	pass

def readFile(file: str = None, model: str = None, test: str = None, user: str = None) -> tuple[pd.DataFrame, dict]:
	metaData = {'test': test, 'model': model}

	data = None
	protocolInfo = {}
	filenameInfo = defaultdict(str)
	filenameInfo['full filename'] = os.path.basename(file)

	filename = os.path.basename(file).split('_')
	sectionHeaders = {
		'A/D Sampling Rate': int,
		'*** Setup Parameters ***': int,
		'Diameter': int,
		'*** Test Protocol Parameters ***': int,
		'Time (ms)\tControl Function\tOptions': int,
		'Time (ms)': int,
		'*** Force and Length Signals vs Time ***': int
		} 
	characteristics = {
		'Fiber Length': float,
		'Initial Sarcomere Length': float,
		'Diameter': float,
		'Fiber Stiffness (mN/mm)': float
		}
	testParams = [
		'Force-Step', 
		'Length-Step',
		'Length-Ramp',
		'Bath',
		'Data-Enable',
		'Data-Disable'
		]

	with open(file) as tempFile:
		textData = []
		lineNumbers = range(1, 300)
		for idx, line in enumerate(tempFile):
			if idx in lineNumbers:
				for variable in characteristics.keys():
					if variable in line:
						characteristics[variable] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

				textData.append(line.strip())
			for header in sectionHeaders.keys():
				if header in line:
					sectionHeaders[header] = idx
	
			if idx > np.max(lineNumbers):
				break

	characteristics['CSA'] = np.pi*(float(characteristics['Diameter'])/2)**2
	protocolInfo['sample rate'] = float(re.findall(r"[-+]?\d*\.\d+|\d+", textData[sectionHeaders['A/D Sampling Rate']-1])[0])
	
	for param in testParams:
		paramTime = [float(i.split('\t')[0]) for i in textData[sectionHeaders['Time (ms)\tControl Function\tOptions']:sectionHeaders['*** Force and Length Signals vs Time ***']] if param in i]
		paramInfo = [i.split('\t')[3].strip() for i in textData[sectionHeaders['Time (ms)\tControl Function\tOptions']:sectionHeaders['*** Force and Length Signals vs Time ***']] if param in i and len(i.split('\t'))>2]
		protocolInfo[param] = {
			'time': paramTime,
			'info': paramInfo} 
		
		if param == 'Bath':
			bathNumber = [i.split(' ')[0] for i in paramInfo]
			delay = [''.join(i.split(' ')[1:]) for i in paramInfo]
			protocolInfo[param] = {
				'info': {
					'bath': bathNumber,
					'delay': delay
				},
				'time': paramTime}
	
	columns = {
		0: 'Time (ms)', 
		1: 'Length in (mm)', 
		2: 'Length out (mm)', 
		3: 'Force in (mN)', 
		4: 'Force out (mN)'}

	data = pd.read_table(
		file, 
		encoding = 'latin1',
		memory_map = True,
		low_memory = False,
		header = None,
		delim_whitespace = True,
		on_bad_lines = 'skip',
		skiprows = sectionHeaders['*** Force and Length Signals vs Time ***']+2,
		usecols = columns.keys(),
		names = columns.values())

	data['Normalized Length'] = data['Length in (mm)'] / characteristics['Fiber Length']

	if test == 'rFD':
		filenameInfo['protocol descriptor'] = filename[3]
	
	if test == 'rFE':
		filenameInfo['starting SL'] = '.'.join(filename[3].split())
		filenameInfo['ending SL'] = '.'.join(filename[4].split())

	if test in ['pCa', 'ktr']:
		filenameInfo['animal'] = filename[0]
		filenameInfo['fibre'] = filename[1]
		filenameInfo['muscle'] = filename[2]
		filenameInfo['pCa'] = filename[3][3:]

	if test == 'Power':
		powerLoads = []
		filenameInfo['animal'] = filename[0]
		filenameInfo['muscle'] = filename[1]
		filenameInfo['fibre'] = filename[2]
		if filename[3].__contains__('LC'):
			filename[3] = filename[3][3:14]

		#----- Split end of filename by comma and append each isotonic load as int to list for future reference -----+
		All_Loads = filename[3].split(',')
		for l in All_Loads: 
			if l.__contains__('.dat'):
				l = l.split('.')
				load = int(l[0])
			else:
				load = int(l)
			powerLoads.append(load)
		filenameInfo['power loads'] = powerLoads

	if test in ['SSC', 'CONCENTRIC', 'ISO', 'rFE', 'rFD']:
		filenameInfo['animal'] = filename[0].upper()
		filenameInfo['fibre'] = filename[1].capitalize()
		filenameInfo['protocol'] = filename[2].capitalize()
	
	if test == 'SSC':
		filenameInfo['stretch speed'] = filename[3].capitalize()
		filenameInfo['shorten speed'] = filename[4].capitalize()
	
	if test == 'CONCENTRIC':
		filenameInfo['shorten speed'] = filename[3].capitalize()

	if test == 'ISO':
		filenameInfo['starting SL'] = filename[3].capitalize()
	
	if test in ['pCa', 'ktr', 'SSC', 'CONCENTRIC', 'ISO', 'Power', 'rFE', 'rFD', 'Makenna']:
		for key, dictionary in zip(['protocol info', 'characteristics','filename info'], [protocolInfo, characteristics, filenameInfo]):
			metaData[key] = dictionary

		# return data, protocolInfo, characteristics, filenameInfo
		return data, metaData

def pCaAnalysis(data: pd.DataFrame = None, metaData: dict = None, graph: plt.Axes = None) -> tuple[pd.DataFrame, float, float]:
	"""
	Define peak force as the highest rolling 500 ms average \n
	Returns:
	 - Orginal dataframe (with baseline force subtracted from force column)
	 - Peak force
	 - Specific force (i.e., peak force / CSA)
	"""

	removeBaseline(data, range(int(1000), int(5000)))
	
	# Find highest rolling 500 ms window in force
	windowLength = 5000
	graphWindow = int(windowLength / 2)
	peakForce, peakIndex = findPeakForce(data['Force in (mN)'], windowLength)

	if graph:
		graph.plot(data['Time (ms)'].div(msToSeconds), data['Force in (mN)'], color = Colors.Black)
		graph.plot(data['Time (ms)'].div(msToSeconds)[peakIndex - graphWindow: peakIndex + graphWindow], data['Force in (mN)'][peakIndex - graphWindow: peakIndex + graphWindow], color = Colors.Firebrick, label = 'Peak Force')
		graph.text(
			x = 0.5, y = 0.1,
			s = f'Peak force = {peakForce:.2f}uN', 
				transform = plt.gca().transAxes,
				horizontalalignment = 'center',
				verticalalignment = 'center')

	return {
			'Absolute Force': peakForce, 
			'Specific Force': peakForce/metaData['characteristics']['CSA']
			}

def ktrAnalysis(data: pd.DataFrame = None, metaData: dict = None, graph: plt.Axes = None) -> tuple[float, float, float, pd.Series, pd.Series, float]:
	"""
	model ktr \n
	Returns:
	 - Stiffness (i.e., \u0394 force / \u0394 normalized length)
	 - ktr
	 - ktr goodness of fit
	 - Estimated x and y data from modeling (can be graphed with real data to visually inspect fit)
	 - Average force over the final 500 ms of test
	"""
	def findClosestIndex(target, numbers):
		return numbers.index(min(numbers, key = lambda x: abs(x - target)))

	def ktrModel(x, a, kt, c):
		return a * (1-np.exp(-kt*x)) + c

	def generateInitialParameters(xData: pd.Series = None, yData: pd.Series = None):
		def sumOfSquaredError(parameterTuple):
			warnings.filterwarnings("ignore")
			val = ktrModel(xData, *parameterTuple)
			return(np.sum((yData - val) ** 2.0))

		parameterBounds = []
		maxY = max(yData)
		minY = min(yData)
		maxForce: float = maxY - minY
		#----- if force doesn't fully recover after ktr, curve fit to ~90% of max force instead -----+
		if yData[:-500].mean() < maxForce * 0.9:
			maxForce = maxForce * 0.9
		#----- force at ktr start -----+
		forceT0: float = yData[0] 

		parameterBounds.append([maxForce, maxForce]) #----- search bounds for a (force when at plateau) -----+
		parameterBounds.append([0, 30]) #----- search bounds for ktr -----+
		parameterBounds.append([forceT0, forceT0]) #----- searh bounds for c (force at t = 0) -----+

		#----- "seed" the numpy random number generator for repeatable results -----+
		result = differential_evolution(sumOfSquaredError, parameterBounds, seed = 3)
		return result.x

	def modelktr(ktrStart: int, ktrEnd: int):
		print("model ktr entered")
		print("ktr start: ", ktrStart)
		print("ktr end: ", ktrEnd)
		xData = np.array(modelData['Time (ms)'][ktrStart:ktrEnd])
		yData = np.array(modelData['Force in (mN)'][ktrStart:ktrEnd])

		#----- Find initial parameters for curve fitting -----+
		ktrParameters = generateInitialParameters(xData, yData)

		maxfev:int = 1000 #----- number of iterations code will attempt to find optimal curve fit -----+
		try:
			fittedParameters, pcov = curve_fit(ktrModel, xData, yData, ktrParameters, maxfev = maxfev)
		except:
			try:
				maxfev = 5000
				fittedParameters, pcov = curve_fit(ktrModel, xData, yData, ktrParameters, maxfev = maxfev)
			except:
				print(Error(f"ktr parameters were not fit after {maxfev} iterations for file: {metaData['full filename']}. Added to 'Files to Check'"))
				return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
				
		#----- Generate model predictions -----+
		modelPredictions = ktrModel(xData, *fittedParameters)

		#----- Calculate error -----+
		maxForceError: float = np.sqrt(pcov[0, 0])
		ktrError: float = np.sqrt(pcov[1, 1])
		forceT0Error: float = np.sqrt(pcov[2, 2])
		ktr: float = fittedParameters[1]
		absError = modelPredictions - yData

		SE: float = np.square(absError)  #---- squared errors -----+
		MSE: float = np.mean(SE)  #----- mean squared errors -----+
		RMSE: float = np.sqrt(MSE)  #----- Root Mean Squared Error, RMSE -----+
		goodnessFit: float = 1.0 - (np.var(absError) / np.var(yData))

		xModel: np.array = np.linspace(min(xData), max(xData), 100)
		yModel: np.array = ktrModel(xModel, *fittedParameters)
		
		ktrForce = data['Force in (mN)'][-5000:].mean()
		
		if graph:
			subset: int = 4000
			graph.plot(modelData['Time (ms)'][:subset], modelData['Force in (mN)'][:subset], color = Colors.Black, label = 'Raw')
			graph.plot(xModel[:subset], yModel[:subset], color = Colors.Firebrick, label = 'Fit')
			graph.text(
				x = 0.5, y = 0.1,
				s = f'ktr = {ktr:.3f}\n'
					f'Goodness of fit = {goodnessFit * 100:.2f}%', 
				transform = plt.gca().transAxes,
				horizontalalignment = 'center',
				verticalalignment = 'center')
		return ktr, goodnessFit, ktrForce
	
	
	protocolInfo = metaData['protocol info']
	#----- find index of length restretch from the protocol defined in text file -----+
	restretchIndex = findClosestIndex(protocolInfo['Length-Ramp']['time'][0], protocolInfo['Length-Step']['time'])
	
	#----- grab actual time associated with restretch and stiffness steps from protocol -----+
	ktrRestretch = protocolInfo['Length-Step']['time'][restretchIndex]
	stiffnessPostKtr = protocolInfo['Length-Step']['time'][restretchIndex+1] 

	#----- find where restretch and stiffness occur in actual data -----+
	initialktrStart = data.index[data['Time (ms)'] == ktrRestretch][0]
	initialktrEnd = data.index[data['Time (ms)'] == stiffnessPostKtr-500][0]
	
	stiffnessResults = stiffnessAnalysis(data, 0.9, metaData['protocol info']['sample rate'], graph)

	modelData = pd.DataFrame(data[['Time (ms)', 'Force in (mN)']][initialktrStart:initialktrEnd])
	#----- time must be in seconds for curve fitting to work properly -----+
	modelData['Time (ms)'] = modelData['Time (ms)'].div(msToSeconds) 
	
	#----- ktr start defined as the min force value + an offset -----+
	ktrStartOffset = 100
	minForceIndex = modelData[['Force in (mN)']].idxmin()
	ktrStart: int = (minForceIndex[0] + ktrStartOffset) - modelData.index[0]
	ktrEnd = ktrStart + 4000
	ktr, goodnessFit, ktrForce = modelktr(ktrStart, ktrEnd)

	coords=plt.ginput(n=2, show_clicks= True, timeout=9999)
	if coords:	
		ktrStartClick=round(coords[0][0] * 10000)
		ktrEndClick=round(coords[1][0] * 10000)
		graph.clear()
		ktr, goodnessFit, ktrForce = modelktr(ktrStartClick - 10000, ktrEndClick - 10000)


	return {
			"ktr": ktr,
			"Goodness of Fit": goodnessFit,
			"Absolute Force after ktr": ktrForce,
			"Specific Force after ktr": ktrForce/metaData['characteristics']['CSA'],
			"Stiffness after ktr": stiffnessResults['Stiffness']
			}

def powerAnalysis(data: pd.DataFrame = None, metaData: dict = None) -> tuple[float, dict]:
	"""
	Four isotonic clamps are performed during each test\n

	During each clamp, the following is calculated:
	- Force (determined as the mean of force-time curve during each clamp)
	- Normalized force (i.e., force / CSA)
	- Velocity (i.e., \u0394 length / \u0394 time during clamp)
	- Normalized velocity (i.e., velocity / fibre length)

	Returns:
	- Max force (i.e., highest single point during 500 ms just before first clamp begins)
	- Force-Velocity data for each clamp
	"""
	def addText(x: int | float = None, y: int | float = None, txt: str = None):
		subplot.text(
			x = x,
			y = y - textOffset,
			s = txt,
			transform = subplot.transAxes,
			horizontalalignment = 'center',
			verticalalignment = 'center',
			fontdict = dict(size = 12, color = lineColor))
	textOffset = 0
	forceVelocityData = {}
	
	clampTimes = metaData['protocol info']['Force-Step']['time']
	powerLoads = metaData['filename info']['power loads']

	maxForce = data['Force'][(195000 - 5000):195000].max()

	for idx, load in enumerate(powerLoads):
		rows = range(int(clampTimes[idx] + 500), int(clampTimes[idx+1]))
		start = rows[0]
		end = rows[-1]

		absoluteForce: float = data['Force'][start:end].mean()
	
		try:
			absoluteVelocity:float = ((data['Length'][end] - data['Length'][start]) / (data['Time'][end] - data['Time'][start]) * -1)
		except:
			absoluteVelocity = 0
		
		forceVelocityData[load] = {
			f'{load}% Active Force': absoluteForce,
			f'{load}% Specific Force': absoluteForce / metaData['characteristics']['CSA'],
			f'{load}% Velocity': absoluteVelocity,
			f'{load}% Normalized Velocity': absoluteVelocity / metaData['characteristics']['Fibre Length']
		}
		
	allColors = [Colors.PowerColors[load] for load in powerLoads]
	
	forceGraph, lengthGraph = plotting.createForceLengthFig()

	for subplot, variable in zip([forceGraph, lengthGraph], ['Force', 'Length']):
		subplot.plot(data['Time'][190000:215000], data[variable][190000:215000], color = Colors.Black, label = variable)
		subplot.set_yticklabels([])
		subplot.set_yticks([])
		subplot.set_xticklabels([])
		subplot.set_xticks([])
		
		#----- highlight data in force and length signals during each clamp -----+
		for lineColor, clamp in zip(allColors, clampTimes):
			calculationStart = int(clamp[0] + 500) #----- ignore first 50 ms of clamp -----+
			calculationEnd = int(clamp[1])

			force: float = (data['Force'][calculationStart:calculationEnd].mean())
			velocity: float = ((data['Length'][calculationEnd] - data['Length'][calculationStart]) / (data['Time'][calculationEnd] - data['Time'][calculationStart]))

			subplot.plot(
				data['Time'][calculationStart:calculationEnd], 
				data[variable][calculationStart:calculationEnd], 
				color = lineColor, 
				label = 'Calculations')

			if variable == 'Force':
				addText(0.3, 0.7, f'{force:.3f}')

			if variable == 'Length':
				addText(0.3, 1.5, f'{velocity * -1:.3f}')
		
			textOffset += .2
	
	for string, xCoord, color in zip(powerLoads, [196500, 201500, 206500, 211500], allColors):
		forceGraph.annotate(
			f'{string}%',
			xy = (data['Time'][xCoord], (data['Force'][190000])),
			xycoords = 'data',
			fontsize = 20,
			fontname = 'Arial',
			color = color)

	forceGraph.axes.spines.bottom.set_visible(False)
	forceGraph.axes.xaxis.set_visible(False)

	return {
			'Max Force': maxForce,
			'Force-Velocity': forceVelocityData
			}

def residualForceAnalysis(data: pd.DataFrame = None, metaData: dict = None, forceGraph: plt.Axes = None) ->dict[float, float, float, float]:
	"""
	Returns:
	 - Peak force (i.e., average force during 500 ms prior to rFE stretch)
	 - Specific force (peak force / CSA)
	 - Passive force (i.e., average force during final 500 ms of test)
	 - Stiffness (i.e., \u0394 force / \u0394 normalized length)
	"""

	removeBaseline(data, range(int(100000), int(105000)))
	passiveWindow = range(len(data) - 5001, len(data)-1)

	stiffnessResults = stiffnessAnalysis(data, findStiffnessTime(data, metaData, 30000), 10000, forceGraph)
	passiveForce = np.mean(data['Force in (mN)'][passiveWindow])

	return {
			f"Asbolute Force": stiffnessResults['Force before Stiffness'],
			f"Specific Force": stiffnessResults['Force before Stiffness'] / metaData['characteristics']['CSA'],
			f"Passive Force": passiveForce,
			f"Stiffness": stiffnessResults['Stiffness'],
			}

def getContractionData(data: pd.DataFrame = None, idx: int = None, protocolInfo: dict = None) -> pd.DataFrame:
	"""
	get subset of data where contraction occurs
	"""
	lengthStart = protocolInfo['Length-Ramp']['time'][idx]
	lengthEnd = lengthStart + (float(protocolInfo['Length-Ramp']['info'][idx].split()[2]) * msToSeconds)
	lengthStartIndex = data.index[data['Time (ms)'] == float(lengthStart)][0]
	lengthEndIndex = data.index[data['Time (ms)'] == float(lengthEnd)][0]
	lengthWindow = range(int(lengthStartIndex), int(lengthEndIndex))

	return data.loc[lengthWindow, ['Time (ms)', 'Length in (mm)', 'Force in (mN)']]

def workCalculation(data: pd.DataFrame, sampleRate: int = 10000, forceGraph: plt.Axes = None, lengthGraph: plt.Axes = None,  graphLinecolor: Colors = None, graphLabel: str = None, annotationX = None, annotationY = None) -> float:
	"""
	calculate work and power
	"""
	lengthChange = (data['Length in (mm)'].iloc[-1] - data['Length in (mm)'].iloc[0]) * -1
	cumForce = np.max(data['Force in (mN)'].cumsum())
	contractionDuration = (data['Time (ms)'].iloc[-1] - data['Time (ms)'].iloc[-0]) / msToSeconds
	work = ((cumForce * lengthChange)/(contractionDuration)) / sampleRate
	power = work / contractionDuration

	forceGraph.plot(data['Time (ms)'].div(msToSeconds), data['Force in (mN)'], color = graphLinecolor, label = graphLabel)
	lengthGraph.plot(data['Time (ms)'].div(msToSeconds), data['Length in (mm)'], color = graphLinecolor, label = graphLabel)
	forceGraph.fill_between(data['Time (ms)'].div(msToSeconds), y1 = data['Force in (mN)'], color = graphLinecolor,  alpha = 0.4)
	forceGraph.annotate(
		f"Work = {work:.3f} \nPower = {power:.3f} \nContraction duration = {contractionDuration:.3f}",
		(annotationX, annotationY),
		size = 14
	)
	return {
			'Work': work,
			'Power': power
			}

def Binta_Analysis(data: pd.DataFrame = None, metaData: dict = None, graph: plt.Axes = None) -> dict:
	def createGraph(colors: list[Colors] = None, labels: list[str] = None):
		for df, graphColor, label in zip([firstFrameData, secondFrameData], colors, labels):
			removeBaseline(df, range(int(100000), int(105000)))
			graphData = df[100000:700000].reset_index(drop = True)
			forceGraph.plot(graphData['Time (ms)'].div(msToSeconds), graphData['Force in (mN)'], color = graphColor, label = label)
			lengthGraph.plot(graphData['Time (ms)'].div(msToSeconds), graphData['Length in (mm)'], color = graphColor, label = label)
	
	analysisResults = {}
	protocolInfo = metaData['protocol info']
	secondFrameIndex = data.index[data['Time (ms)'] == float(protocolInfo['Data-Enable']['time'][1])][0]
	firstFrameData = pd.DataFrame(data[:secondFrameIndex], dtype = float).reset_index(drop = True)
	secondFrameData = pd.DataFrame(data[secondFrameIndex:], dtype = float).reset_index(drop = True)
	firstFrameData['Time (ms)'] = firstFrameData['Time (ms)'] - firstFrameData['Time (ms)'][0]
	secondFrameData['Time (ms)'] = secondFrameData['Time (ms)'] - secondFrameData['Time (ms)'][0]

	forceGraph, lengthGraph = plotting.createForceLengthFig()

	if metaData['filename info']['protocol'].upper() == 'RFD':
		forceDepressionResults, isoResults = map(lambda data: residualForceAnalysis(data, metaData, forceGraph), [firstFrameData, secondFrameData])
		createGraph([Colors.Perrywinkle, Colors.Black], ['rFD', 'ISO'])

		columnHeader = f"{metaData['filename info']['protocol descriptor']}"
		for columnBasename, value in forceDepressionResults.items():
			analysisResults[f"{columnBasename} - {columnHeader} rFD"] = value
		for columnBasename, value in isoResults.items():
			analysisResults[f"{columnBasename} - {columnHeader} ISO"] = value

	if metaData['filename info']['protocol'].upper() == 'RFE':
		isoResults, forceEnhancementResults = map(lambda data: residualForceAnalysis(data, metaData, forceGraph), [firstFrameData, secondFrameData])
		createGraph([Colors.Black, Colors.Perrywinkle], ['ISO', 'rFE'])
		
		columnHeader = f"{metaData['filename info']['starting SL']}-{metaData['filename info']['ending SL']}"
		for columnBasename, value in forceEnhancementResults.items():
			analysisResults[f"{columnBasename} - {columnHeader} rFE"] = value
		for columnBasename, value in isoResults.items():
			analysisResults[f"{columnBasename} - {columnHeader} ISO"] = value
	
	# plt.legend()
	# plt.show()
	# plt.savefig(f"/Volumes/Lexar/Binta/{metaData['filename info']['full filename']}-fig", dpi = 500)
	plt.close()
	return analysisResults

def Makenna_Analysis(data: pd.DataFrame = None, metaData: dict = None, graph: plt.Axes = None) -> dict:
	protocolInfo = metaData['protocol info']
	removeBaseline(data, range(int(100000), int(105000)))

	stretchColor = '#f67e2a'
	shortenColor = '#31d3db'
	forceGraph, lengthGraph = plotting.createForceLengthFig()

	graphData = data[100000:600000]
	forceGraph.plot(graphData['Time (ms)'].div(msToSeconds), graphData['Force in (mN)'], color = Colors.Black, label = 'Force')
	lengthGraph.plot(graphData['Time (ms)'].div(msToSeconds), graphData['Length in (mm)'], color = Colors.Black, label = 'Length')
	forceGraph.set_title(metaData['filename info']['full filename'])
	
	stiffnessResults = stiffnessAnalysis(data, findStiffnessTime(data, metaData, 40000), 10000, forceGraph)
	if metaData['filename info']['protocol'].upper() == 'SSC':
		forceGraph.annotate(
			f"Force after shortening = {stiffnessResults['Force before Stiffness']:.3f}",
			(40, stiffnessResults['Force before Stiffness'] * 0.6),
			size = 14
		)
		stretchData, shortenData = map(lambda number: getContractionData(data, number, protocolInfo), [0, 1])
		peakEccentricForce = np.max(stretchData['Force in (mN)'])
		
		stretchResults, shortenResults = map(
			lambda data, lineColor, label, annotationX, annotationY: workCalculation(
				data, 
				10000,
				forceGraph,
				lengthGraph, 
				lineColor, 
				label, 
				annotationX,
				annotationY), 
					[stretchData, shortenData], 
					[stretchColor, shortenColor], 
					['Stretch', 'Shorten'], 
					[15, 40],
					[np.max(data['Force in (mN)']) / 2, np.max(data['Force in (mN)']) / 2]
		)
		
		netWork = shortenResults['Work'] + stretchResults['Work']
		netPower = shortenResults['Power'] + stretchResults['Power']
		
		columnHeader = f"{metaData['filename info']['stretch speed']}-{metaData['filename info']['shorten speed']}"
		analysisResults = {
			f"Peak Eccentric Force during {columnHeader}": peakEccentricForce, 
			f"Stretch Work during {columnHeader}": stretchResults['Work'],
			f"Shorten Work during {columnHeader}": shortenResults['Work'],
			f"Net Work during {columnHeader}": netWork,
			f"Stretch Power during {columnHeader}": stretchResults['Power'],
			f"Shorten Power during {columnHeader}": shortenResults['Power'],
			f"Net Power during {columnHeader}": netPower,
			f"Absolute Force following {columnHeader}": stiffnessResults['Force before Stiffness'],
			f"Specific Force following {columnHeader}": stiffnessResults['Force before Stiffness'] / metaData['characteristics']['CSA'],
			f"Stiffness following {columnHeader}": stiffnessResults['Stiffness']
		}

	if metaData['filename info']['protocol'].upper() == 'ISO':
		columnHeader = f"{metaData['filename info']['starting SL']}"
		analysisResults = {
			f"Absolute Force @ {columnHeader} SL": stiffnessResults['Force before Stiffness'],
			f"Specific Force @ {columnHeader}": stiffnessResults['Force before Stiffness'] / metaData['characteristics']['CSA'],
			f"Stiffness  @ {columnHeader} SL": stiffnessResults['Stiffness']
		}
		plt.close()
		
	if metaData['filename info']['protocol'].upper() == 'CONCENTRIC':
		forceGraph.annotate(
			f"Force after shortening = {stiffnessResults['Force before Stiffness']:.3f}",
			(45, stiffnessResults['Force before Stiffness'] * 0.6),
			size = 14
		)
		shortenData = getContractionData(data, 0, protocolInfo)
		shortenResults = workCalculation(shortenData, 10000, forceGraph, lengthGraph, shortenColor, 'Shortening', 45, np.min(data['Force in (mN)']))

		columnHeader = f"{metaData['filename info']['shorten speed']}"
		analysisResults = {
			f"Shorten Work during {columnHeader} Shortening": shortenResults['Work'],
			f"Shorten Power during {columnHeader} Shortening": shortenResults['Power'],
			f"Absolute Force following {columnHeader}": stiffnessResults['Force before Stiffness'],
			f"Specific Force following {columnHeader}": stiffnessResults['Force before Stiffness'] / metaData['characteristics']['CSA'],
			f"Stiffness following {columnHeader}": stiffnessResults['Stiffness']
		}
	
	# plt.savefig(f"/Volumes/Lexar/Makenna/{metaData['filename info']['full filename']}-fig", dpi = 500)
	# plt.show()
	plt.close()
	return analysisResults

def findStiffnessTime(data: pd.DataFrame = None, metaData: dict = None, activationDuration: int = None):
	protocolInfo = metaData['protocol info']
	activationStart = protocolInfo['Bath']['time'][protocolInfo['Bath']['info']['bath'].index('4')]
	return data.index[data['Time (ms)'] == activationStart + activationDuration][0]

def stiffnessAnalysis(data: pd.DataFrame, stiffnessTimeSecs: float|int, sampleRate: int = 10000, graph: plt.Axes  =  None):
	if stiffnessTimeSecs < 100:
		stiffnessTime = stiffnessTimeSecs * sampleRate
	else:
		stiffnessTime = stiffnessTimeSecs
	stiffnessWindow = range(int(stiffnessTime) - 100, int(stiffnessTime) + 200)
	forceWindow = range(int(stiffnessTime) - 5001, int(stiffnessTime) - 1)
	dF = (data['Force in (mN)'][stiffnessWindow]).max() - (data['Force in (mN)'][forceWindow]).mean()
	dLo = (data['Normalized Length'][stiffnessWindow]).max() - (data['Normalized Length'][forceWindow]).mean()
	stiffness = dF/dLo
	forceBeforeStiffness = np.mean(data['Force in (mN)'][forceWindow])
	# if graph != None:
	# 	graph.plot(data['Time (ms)'].div(msToSeconds)[forceWindow], data['Force in (mN)'][forceWindow], color = Colors.SkyBlue, label = 'Peak force')
	return {
			'Stiffness': stiffness, 
			'Force before Stiffness': forceBeforeStiffness
			}

def findPeakForce(data: pd.DataFrame, windowLength: int) -> tuple[float, int]:
	"""
	find highest rolling of window length average of force signal
	"""
	temp = pd.DataFrame()
	temp['Rolling Mean'] = pd.DataFrame(data.rolling(window = windowLength, center = True).mean())

	peakForce = temp['Rolling Mean'].max()
	try:
		peakIndex = temp['Rolling Mean'].argmax() + data.index[0] #----- add first index to peak force index to get the true index of peak force -----+
	except: 
		peakForce = data.max()
		peakIndex = 0 

	return peakForce, peakIndex

def removeBaseline(data: pd.DataFrame = None, baselineWindow: range = None):
	"""
	subtract baseline force from force signal
	"""
	data['Force in (mN)'] = data['Force in (mN)'] - np.mean(data['Force in (mN)'][baselineWindow])
	return 