#----- import python modules -----+
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from classes.classes import Colors

plt.style.use('styles/custom_style.mplstyle')

#----- useful strings for defining x/y labels -----+
torqueLabel = fr'Torque (mN$\cdot$m)'
torqueUnits = fr'mN$\cdot$m'
rtdTitle = fr'RTD (mN$\cdot$m$\cdot$s$^\minus$$^1$)'
rtdUnits = fr'(mN$\cdot$m$\cdot$s$^\minus$$^1$)'
jointAngleLabel = fr'Joint Angle (deg$\cdot$s$^\minus$$^1$)'

edgeRGB=mcolors.hex2color(Colors.Black) + (1.0,)

def hex_to_rgb(hex: Colors = None, alpha: int | float = 1):
	"""
	convert hex codes to RGB notation

	optional:
	- input an alpha to control transparency (0 = completely transparent, 1 = no transparency (default))
	"""
	return mcolors.hex2color(hex) + (alpha,)

def keyPress(Event):
	"""
	press "enter" to close any open figures
	"""
	if Event.key == 'enter':
		plt.close('all')
		Continue = False

def initializeFig():
	"""
	inititalize a matplotlib figure and subplot to be drawn on
	"""
	fig, graph = plt.subplots()
	fig.canvas.mpl_connect('key_press_event', keyPress)
	return fig, graph

def show(xlabel: str = None, ylabel: str = None, legend: bool = False):
	"""
	define x and y labels of figure(s) and then show any open figures
	"""
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	if legend == True:
		plt.legend()
	plt.show()

def mouseClickEvent(coords):
	"""
	handle mouse left click event
	"""
	#----- click_x may need to be multiplied by 10, 100, 1000 depending on how the index and "time" data are arranged -----+

	clickX=round(coords[0][0])
	clickY=round(coords[0][1])

	# ax.plot(data['Time (ms)'][click_x:], data['Force in (mN)'][click_x:], color='red')
	# ax.axvline(x=data['Time (ms)'][click_x], ymin = 0, ymax = 1, color = 'red', linestyle = '--', linewidth = 2)
	# fig.canvas.draw()
	# plt.show()

	return clickX, clickY

def createForceLengthFig(forceY: str = 'Force (mN)', lengthY: str = 'Length (mm)', lengthX: str = 'Time (s)', heightRatios: list = [3, 1]):
	"""
	Returns a Gridspec figure for force (top) and length (bottom) subplots
	"""
	fig = plt.figure()
	fig.canvas.mpl_connect('key_press_event', keyPress)
	gs = GridSpec(2, 1, height_ratios = heightRatios)
	forceGraph = plt.subplot(gs[0])
	lengthGraph = plt.subplot(gs[1])

	forceGraph.set_ylabel(forceY)
	forceGraph.set_xticklabels([])
	forceGraph.yaxis.set_major_locator(MaxNLocator(nbins=5))
	lengthGraph.set_ylabel(lengthY)
	lengthGraph.set_xlabel(lengthX)
	return forceGraph, lengthGraph

