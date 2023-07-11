#----- import python modules -----+
import matplotlib.colors as mcolors
from classes import Colors
import matplotlib.pyplot as plt

# plt.rcParams.update({
#                 'font.family': 'times new roman',
#                 'xtick.labelsize': 12,
#                 'ytick.labelsize': 12,
#                 'axes.labelsize': 12,
#                 'axes.titlesize': 12,
#                 'axes.spines.right': False,
#                 'axes.spines.top': False,
#                 'figure.dpi': 500,
#                 'legend.edgecolor': 'white',
#                 'figure.figsize': [9, 6],
#                 'figure.autolayout': True})

plt.style.use('custom_style.mplstyle')

#----- useful strings for defining x/y labels -----+
torqueLabel = fr'Torque (mN$\cdot$m)'
torqueUnits = fr'mN$\cdot$m'
rtdTitle = fr'RTD (mN$\cdot$m$\cdot$s$^\minus$$^1$)'
rtdUnits = fr'(mN$\cdot$m$\cdot$s$^\minus$$^1$)'
jointAngle = fr'Joint Angle (deg$\cdot$s$^\minus$$^1$)'

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

def initialize_fig():
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

def mouseclick_event(coords):
    """
    handle mouse left click event
    """
    #----- click_x may need to be multiplied by 10, 100, 1000 depending on how the index and "time" data are arranged -----+

    click_x=round(coords[0][0])
    click_y=round(coords[0][1])

    # ax.plot(data['Time (ms)'][click_x:], data['Force in (mN)'][click_x:], color='red')
    # ax.axvline(x=data['Time (ms)'][click_x], ymin = 0, ymax = 1, color = 'red', linestyle = '--', linewidth = 2)
    # fig.canvas.draw()
    # plt.show()

    return click_x, click_y

# coords=plt.ginput(n=1, show_clicks=True, timeout=9999)
# if coords:
#     mouseclick_event()

