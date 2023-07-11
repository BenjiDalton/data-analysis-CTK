#----- import python modules -----+
from dataclasses import dataclass
import pandas as pd

@dataclass
class FileInfo:
    """
    Stores information from each file. Allows for easier organization of data
    """
    Filename: str=None
    Subject: dict=None
    Animal: str=None
    Muscle: dict=None
    Fibre: dict=None
    pCa: dict=None
    Test: str=None
    OrganizedData: dict=None

@dataclass
class TwitchResults:
    """
    twitches during PLFFD task
    """
    Name: str=None
    RTDStartIndex: int=None
    RTDEndIndex: int=None
    TimetoRTDEnd: float=None
    RTD: float=None
    PeakTwitchIndex: int=None
    PeakTwitch: float=None
    HRTForceIndex: int=None
    HRT: float=None
    HRTForce: float=None
    TwitchEndIndex: int=None
    PeakTorque: float=None
    PeakTorqueData: pd.DataFrame=None

@dataclass
class Colors:
    """ 
    list of colors defined by hex code
    visit https://htmlcolorcodes.com/color-picker/ to find new colors you want to add
    """
    DeepBlue='#00688B'
    SkyBlue='#87CEEB'
    Firebrick='#B22222'
    SeaGreen='#4EEE94'
    Sienna='#FF8247'
    Charcoal='#525252'
    LightGray='#B0B0B0'
    Black='#000000'
    White='#ffffff'
    Slate='#404040'
    DarkGray='#302f33'
    SteelBlue='#215b76'
    ForestGreen='#168d14'
    AuroraColor='#458c82'
    Perrywinkle='#7E71C8'
    SageGreen='#2E5F4F'
    LemonLime='#A9EB66'

    #----- list of colors to use when creating bar graphs in app -----+
    bar_graph_colors=[
        SteelBlue, 
        Perrywinkle, 
        AuroraColor, 
        Firebrick, 
        SeaGreen, 
        DeepBlue, 
        ForestGreen, 
        Slate, 
        SkyBlue]
