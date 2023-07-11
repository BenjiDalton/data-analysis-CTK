#----- import python modules -----+
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt

#----- import custom modules -----+
import functions.plotting as plotting
from classes.classes import Colors, TwitchResults


ms_to_seconds=1000

class Error(Exception):
    pass

def readFile(file: str=None, model: str=None, test: str=None, user: str=None) -> tuple[pd.DataFrame, dict]:
    print(os.path.basename(file))
    metaData={'test': test, 'model': model}

    data=None
    protocolInfo={}
    filenameInfo=defaultdict(str)
    filenameInfo['full filename']=file

    filename=os.path.basename(file).split('-')

    def ButterworthFilter(data: pd.DataFrame=None, cutoff: int=None, order: int=None) -> pd.Series:

        b, a=butter(order, cutoff, btype='low', analog=False, fs=10000)
        filteredTorque=filtfilt(b, a, data)

        return filteredTorque

    filenameInfo['condition']=filename[0]
    filenameInfo['animal']=f'{filename[0]}_{filename[1]}'
    filenameInfo['timepoint']=filename[2]

    if test == 'Torque-Frequency':
        filenameInfo['frequency']=filename[4]
    
    if test == 'Torque-Velocity':
        filenameInfo['ISO percent']=filename[4]

    if test == 'Recovery':
        recoveryTimepoint=filename[4]
        filenameInfo['recovery timepoint']=f'{recoveryTimepoint} min' if recoveryTimepoint in {'1', '2', '5', '10'} else recoveryTimepoint

    sectionHeaders={
        'Sample Frequency (Hz)': int,
        'Scale (units/V)': int,
        'Protocol Array': int,
        'Test Data in Volts': int,
        'Sample': int}

    with open(file, encoding='latin1') as tempFile:
        textdata=[]
        lineNumbers=range(1, 50)
        for idx, line in enumerate(tempFile):
            for header in sectionHeaders.keys():
                if header in line:
                    sectionHeaders[header]=idx
            textdata.append(line.strip())
    
            if idx > np.max(lineNumbers):
                break
        
    columns ={
        0: 'Time',
        1: 'Length',
        2: 'Torque',
        11: 'Stim'
    }
    
    data=pd.read_table(
        file, 
        encoding='latin1',
        memory_map=True,
        low_memory=False,
        header=None,
        delim_whitespace=True,
        on_bad_lines='skip',
        skiprows=sectionHeaders['Sample']+1,
        usecols=columns.keys(),
        names=columns.values())
        
    
    lengthScale, torqueScale=map(lambda idx: float(textdata[sectionHeaders['Scale (units/V)']].split('\t')[idx]), [1, 2])

    data['Length']=data['Length']*lengthScale
    data['Torque']=data['Torque']*torqueScale
    # data=GetTextData(skiprows=8, columns={0: 'Time', 1: 'Length', 2: 'Raw Torque', 3: 'Other', 11: 'Stim'})
    # Aurora Scientific includes length and torque scale factors in their data files
    lengthScale: float=float(data['Raw Torque'][0])
    torqueScale: float=float(data['Stim'][0])

    # Find start of actual data in text file
    dataStart=data.index[data['Time'] == '0'][0]

    data=pd.DataFrame(data[['Time', 'Length', 'Raw Torque', 'Stim']][dataStart:], dtype=float).reset_index(drop=True)

    data['Time']=data['Time'].div(10000)

    # Baseline torque values 
    if file.__contains__('TF' or 'Isotonic'):
        baselineTorque=data['Raw Torque'].iloc[15000:16000].mean()
    if file.__contains__('PLFFD'):
        baselineTorque=data['Raw Torque'].iloc[21000:22000].mean()
    else:
        baselineTorque=data['Raw Torque'].iloc[0:100].mean()

    data['Raw Torque'] -= baselineTorque

    # Scale length and torque channels based off Aurora Scientific values
    data['Length']=data['Length'] * lengthScale
    data['Raw Torque']=data['Raw Torque'] * torqueScale

    # Filter torque signal with highpass (> 100 Hz) filter to eliminate stim artifacts
    data['Filtered Torque']=ButterworthFilter(data['Raw Torque'], 100, 2)

    return data, filenameInfo

def findPeakTorque(data: pd.DataFrame, windowLength: int) -> tuple[float, int]:
    temp=pd.DataFrame()
    temp['Rolling Mean']=pd.DataFrame(data.rolling(window=windowLength, center=True).mean())

    peakTorque=temp['Rolling Mean'].max()
    try:
        peakTorqueIndex=temp['Rolling Mean'].argmax() + data.index[0] # add first index to peak force index to get the true index of peak force
    except: 
        print(Error(data))
        peakTorque=data.max()
        peakTorqueIndex=0 

    return peakTorque, peakTorqueIndex

def torqueFrequency(data: pd.DataFrame=None, filenameInfo: dict=None, graphbool: bool=False) -> float:
    fig, graph=plotting.initialize_fig()
    # Find index of first stimulation (i.e., contraction start)
    stimIndex=data.index[data['Stim'] == 1][0]
    
    # Find the greatest 500 ms rolling average of torque signal
    # The footplate is moved to starting position at beginning of test so first portion of test (i.e., 20000 samples; 2 sec) is ignored
    if int(filenameInfo['frequency']) >= 80:
        windowLength=500
        graphWindow=int(windowLength / 2)
        peak_torque, peakTorqueIndex=findPeakTorque(data=data['Filtered Torque'], windowLength=windowLength)
        graph.plot(data['Time'], data['Filtered Torque'], color=Colors.Black)
        graph.plot(data['Time'][peakTorqueIndex - graphWindow : peakTorqueIndex + graphWindow], data['Filtered Torque'][peakTorqueIndex - graphWindow : peakTorqueIndex + graphWindow], color=Colors.Firebrick, label='Peak Force')

    else:
        # For non-tetanic contractions, peak torque is highest single point during 500 ms window where stims are delivered
        peak_torque=data['Filtered Torque'][stimIndex : stimIndex + 5000].max()

    if int(filenameInfo['frequency']) >= 100:
        results=RTD_Analysis(data=data, PeakTorque=peak_torque, graphbool=graphbool)
    analysisResults=[peak_torque]+results
    return analysisResults

def isotonicContractions(data: pd.DataFrame=None, filenameInfo: dict=None, UserInput: bool=False) -> pd.DataFrame:
    baselineLength=data['Length'].iloc[0:100].mean()
    if UserInput == False:
        try:
            isoStart=data.index[data['Length'] <= baselineLength * 0.99][0]
        except:
            isoStart: int=0
        
        data=data.iloc[isoStart:]

    #----- Find first sample when footplate crosses end ROM (i.e., -18.99 degrees) -----+
    #----- Define as end of contraction -----+
    endROM=-20 if filenameInfo['timepoint'] == 'D80' or 'D120' else -18.99

    try:
        if data['Length'].min() > endROM:
            isoEnd=data.index[data['Length'] == data['Length'].min()][0]
        #----- If end ROM isn't achieved (possible during higher isotonic loads), then end ROM is final length sample in window -----+
        if data['Length'].min() < endROM:
            isoEnd=data.index[data['Length'] <= endROM][0] - isoStart
        
    except:
        print(Error(f"End ROM not found for {filenameInfo['animal']}, {filenameInfo['timepoint']}"))

    return data.iloc[:isoEnd]

def torqueVelocity(data: pd.DataFrame=None, metaData: dict=None) -> tuple[float, float, float]:
    fig, graph=plotting.initialize_fig()
    contractionData=data[20000:26000].reset_index(drop=True)

    # Isolate data relevant to isotonic contractions 
    try:
        contractionData=isotonicContractions(data=contractionData, filenameInfo=metaData['filename info'])
    except: 
        # contractions where the footplate doesn't move at all have their velocity set at 0
        # May happen when testing against high isotonic loads, following fatigue protocol, etc.
        isoVelocity=0
    else: 
        lengthStart= contractionData['Length'].iloc[1]
        lengthEnd=contractionData['Length'].iloc[-1]

        timeStart=contractionData['Time'].iloc[1]
        timeEnd=contractionData['Time'].iloc[-1]

        isoVelocity=((lengthEnd - lengthStart) / (timeEnd- timeStart) * -1)

    isoTorque=contractionData['Filtered Torque'].mean()
    
    def isoGraphing():
        plt.gca().clear()
        graph.plot(
            data['Time'][20000:28000],
            data['Length'][20000:28000],
            color=Colors.Charcoal)
        graph.plot(
            contractionData['Time'],
            contractionData['Length'], 
            color=Colors.SkyBlue)
    
        graph.text(
            x=0.5, y=0.9,
            s=f'Velocity={isoVelocity:.3f}',
            transform=plt.gca().transAxes,
            horizontalalignment='center',
            verticalalignment='center')
        graph.ylabel(plotting.jointAngleLabel)
        graph.xlabel('Time (s)')
    
    isoGraphing()
    Coords=plt.ginput(n=2, show_clicks= True)
    
    if Coords:
        contractionStart=round(Coords[0][0] * 10000)
        contractionEnd=round(Coords[1][0] * 10000)
        contractionData=data[contractionStart:contractionEnd].reset_index(drop=True)
        
        isoVelocity=((contractionData['Length'].iloc[-1] - contractionData['Length'].iloc[0]) / (contractionData['Time'].iloc[-1] - contractionData['Time'].iloc[0]) * -1) 

        isoGraphing()
        fig.canvas.draw()

    isoPower=isoTorque * isoVelocity
    analysisResults=[
        isoTorque, 
        isoVelocity, 
        isoPower
        ]
    return analysisResults

def Recovery_Isotonics(data: pd.DataFrame=None, metaData: dict=None) -> tuple[float, float, float]:
    def get_isotonic_results(data: pd.DataFrame, metaData: dict):
        try: 
            data=isotonicContractions(data=data, filenameInfo=metaData['filename info'])
        except:
            iso_velocity=0
        else:
            iso_velocity=((data['Length'].iloc[-1] - data['Length'].iloc[1]) / (data['Time'].iloc[-1] - data['Time'].iloc[1]) * -1)
        
        iso_torque=data['Filtered Torque'].mean()
        iso_power=iso_torque * iso_velocity

        return iso_velocity, iso_torque, iso_power
    
    # Grab subset of data where isotonic contractions were performed 
    # Make sure to include at least 1000 samples (i.e., 100 ms) prior to contraction starting so baseline signals can be obtained
    Rec1_Data, Rec2_Data=[data[dataStart:dataStart + 6000] for dataStart in [20000, 55000]]

    rec1, rec2=map(lambda data: get_isotonic_results(data, metaData), [Rec1_Data, Rec2_Data])

    fig, Recovery_ISOs=plt.subplots(nrows=1, ncols=2, figsize=(width:= 6, height:= 4), layout='constrained')
    Recovery_ISOs[0].plot(Rec1_Data['Time'], Rec1_Data['Length'])
    Recovery_ISOs[1].plot(Rec2_Data['Time'], Rec2_Data['Length'])

 
    return rec1 if rec1[2] > rec2[2] else rec2

def Fatigue(data: pd.DataFrame=None, metaData: dict=None) -> int:
    """
    Number of contractions performed during fatigue task is determined by \n
    adding 1 contraction every 20,000 samples (i.e., 2 seconds) until the end of text file
    
    Returns:
    - Number of contractions performed during fatigue protocol
    """
    fig, graph=plotting.initialize_fig()
    IndFatiguecontractions=[]
    # The end of the actual data must be found for fatigue tests
    # The same protocol is used for each subject so there may be several/many extra contractions written into Aurora Scientific protocol
    # If the end of data isn't found, then there could be several thousand rows of N/A values in data file
    # print(Subject, data.index[data['Length'] == 0])
    try:
        data_end=data.index[data['Length'] == 0][0]
        data=data[:data_end]

    except IndexError:
        plt.plot(data['Time'], data['Filtered Torque'], color=Colors.Black)
        plt.show()
        print(Error(f"{metaData['filename info']['animal']} fatigue failed"))
   
    # Find number of contractions performed during fatigue test
    contraction_Start=20000
    contraction_End=contraction_Start + 5000 

    while contraction_End <= data_end:
        contraction=data['Filtered Torque'][contraction_Start:contraction_End].max()
        contraction_Start += 20000
        contraction_End=contraction_Start + 5000 
        IndFatiguecontractions.append(contraction)

    number_of_contractions: int=len(IndFatiguecontractions)

    graph.plot(data['Time'], data['Filtered Torque'], color=Colors.Black)
    return number_of_contractions

def RTD_Analysis(data: pd.DataFrame=None, dataStart: int=None, PeakTorque: float=None):
    """
    RTD start criteria is determined when torque signal exceeds 3 x standard deviations of baseline torque signal\n
    RTD end criteria is when torque signal exceeds 90% of peak torque (to ignore portions of curve that begins to plateau)
    
    Returns:
    - RTD (i.e., \u0394 force / \u0394 time)
    """
    fig, graph=plotting.initialize_fig()
    # contraction onset defined as the point torque exceeds 3 standard deviations of baseline
    data=data[dataStart:].reset_index(drop=True)
    
    BaselineTorque_Mean=data['Filtered Torque'][19000:19500].mean()
    BaselineTorque_STDEV=np.std(data['Filtered Torque'][19000:19500])
    RTDStart_Criteria=BaselineTorque_Mean + (3 * BaselineTorque_STDEV)

    # Find the sample where stimulations began to be delivered
    stimIndex=data.loc[data['Stim'] == 1].index[0]
    windowLength=1000
    graphWindow=int(windowLength / 2)
    
    # Search for the point torque exceeds defined contraction onset only in data following stimulation onset
    # So we are confident contraction onset will be defined properly
    RTDStartIndex=data[stimIndex:].loc[data['Filtered Torque'][stimIndex:] >= RTDStart_Criteria].index[0]

    RTDEndIndex=data[RTDStartIndex:].loc[data['Filtered Torque'][RTDStartIndex:] >= .90 * PeakTorque].index[0]

    RTD, RTD_0_10, RTD_0_25=lambda offset: (data['Filtered Torque'][RTDEndIndex + offset] - data['Filtered Torque'][RTDStartIndex + offset]) / (data['Time'][RTDEndIndex + offset] - data['Time'][RTDStartIndex + offset]), [0, 100, 250]
    Norm_RTD, Norm_RTD_0_10, Norm_RTD_0_25=lambda rtd: rtd/PeakTorque, [RTD, RTD_0_10, RTD_0_25]

    graph.plot(data['Time'][19000:25000], data['Filtered Torque'][19000:25000], color=Colors.Black)
    graph.plot(data['Time'][RTDStartIndex:RTDEndIndex], data['Filtered Torque'][RTDStartIndex:RTDEndIndex], color=Colors.SkyBlue)
    graph.plot(data['Time'][RTDStartIndex:RTDStartIndex + 250], data['Filtered Torque'][RTDStartIndex:RTDStartIndex + 250], color=Colors.Firebrick)
    graph.plot(data['Time'][RTDStartIndex:RTDStartIndex + 100], data['Filtered Torque'][RTDStartIndex:RTDStartIndex + 100], color=Colors.ForestGreen)
    
    graph.text(
        x=0.60, y=0.85,
        s=f'RTD={RTD:.2f} {plotting.RTD_Units}\n'
            f'0-10 ms RTD={RTD_0_10:.2f} {plotting.RTD_Units}\n'
            f'0-25 ms RTD={RTD_0_25:.2f} {plotting.RTD_Units}',
        transform=plt.gca().transAxes,
        fontsize=6,
        bbox=dict(boxstyle='round', facecolor='white'))

    analysisResults=[RTD, RTD_0_10, RTD_0_25, Norm_RTD, Norm_RTD_0_10, Norm_RTD_0_25]
    return analysisResults

def twitchCharacteristics(data: pd.DataFrame=None, dataStart: int=None, contraction: str=None) -> TwitchResults:

    dataRanges={
        'Control Twitch': (dataStart, dataStart + 3000),
        '10 Hz': (dataStart, dataStart + 5000),
        'Tetanus': (dataStart, dataStart + 10000),
        'Potentiated Twitch': (dataStart, dataStart + 3000)}

    dataStart, data_end=dataRanges[contraction]

    SubsetData=data[dataStart:data_end].reset_index(drop=True)
    stimIndex=SubsetData.index[SubsetData['Stim'] == 1][0]

    # contraction onset defined as the point torque exceeds 3 standard deviations of baseline
    BaselineTorque_Mean=SubsetData['Filtered Torque'].iloc[stimIndex - 500:stimIndex - 1].mean()
    BaselineTorque_STDEV=np.std(data['Filtered Torque'].iloc[stimIndex - 500:stimIndex - 1])
    RTDStart_Criteria=BaselineTorque_Mean + (3 * BaselineTorque_STDEV)

    PostStimData=SubsetData[stimIndex:]

    RTD_StartIndex=PostStimData.index[PostStimData['Filtered Torque'] > RTDStart_Criteria][0]
    PeakTwitch=PostStimData['Filtered Torque'].iloc[10:].max()

    AdjustedPeakTwitch=PeakTwitch - PostStimData['Filtered Torque'][RTD_StartIndex]

    PeakTwitchIndex=SubsetData.index[SubsetData['Filtered Torque'] == PeakTwitch][0]
    PostPeakData=SubsetData[PeakTwitchIndex:]

    Post_Twitch_Baseline=PostPeakData['Filtered Torque'].iloc[-300:].mean()

    if Post_Twitch_Baseline > BaselineTorque_Mean:
        TwitchEndIndex=PostPeakData.index[PostPeakData['Filtered Torque'] < (Post_Twitch_Baseline)][0] 
    else:
        TwitchEndIndex=PostPeakData.index[PostPeakData['Filtered Torque'] < (BaselineTorque_Mean)][0]

    HalfForce: float=PeakTwitch / 2

    HalfForceIndex: int=PostPeakData.index[PostPeakData['Filtered Torque'] < (PeakTwitch - (AdjustedPeakTwitch / 2))][0]

    Half_Relaxation_Time=(SubsetData['Time'][HalfForceIndex] - SubsetData['Time'][PeakTwitchIndex]) * 1000

    RTD=(SubsetData['Filtered Torque'][PeakTwitchIndex] - SubsetData['Filtered Torque'][RTD_StartIndex]) / (SubsetData['Time'][PeakTwitchIndex] - SubsetData['Time'][RTD_StartIndex])

    return TwitchResults(
            Name=contraction,
            RTDStartIndex=RTD_StartIndex + dataStart,
            RTD=RTD,
            PeakTwitchIndex=PeakTwitchIndex + dataStart,
            PeakTorque=AdjustedPeakTwitch,
            HRTForceIndex=HalfForceIndex + dataStart,
            HRT=Half_Relaxation_Time,
            HRTForce=HalfForce,
            TwitchEndIndex=TwitchEndIndex + dataStart)

def PLFFD(data: pd.DataFrame=None, filenameInfo: dict=None, graphbool: bool=False) -> tuple[float, float, float, float, float]:
    PLFFD_Data={}
    ControlTwitch=twitchCharacteristics(data, dataStart=29000, contraction='Control Twitch')

    LowHzTwitch=twitchCharacteristics(data, dataStart=39000, contraction ='10 Hz')

    dataStart=46000
    contraction='Tetanus'
    TwitchData=RTD_Analysis(data=data, dataStart=dataStart)
    Tetanus=TwitchResults(
        Name=contraction,
        RTDStartIndex=TwitchData[0] + dataStart,
        RTDEndIndex=TwitchData[1] + dataStart,
        RTD=TwitchData[2],
        TimetoRTDEnd=TwitchData[3],
        PeakTorque=TwitchData[4],
        PeakTorqueData=TwitchData[5]
    )

    PotentiatedTwitch=twitchCharacteristics(data=data, dataStart=59000, contraction='Potentiated Twitch')

    
    if graphbool == True:
        annotate_fontsize=6
        def PLFFD_Graphing():
            graph.plot(
                data['Time'][Twitch.RTDStartIndex - 200:Twitch.TwitchEndIndex], 
                data['Filtered Torque'][Twitch.RTDStartIndex - 200:Twitch.TwitchEndIndex], 
                color=Colors.Black)
            
            graph.set_xlabel('Time (s)')

            graph.set_ylim(subplotYmin, subplotYmax)
                
            graph.annotate(
                'RTD start',
                xy=(data['Time'][Twitch.RTDStartIndex], data['Filtered Torque'][Twitch.RTDStartIndex]),
                xycoords='data',
                xytext=(-50, 40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=annotate_fontsize) 
            graph.annotate(
                f'HRT={Twitch.HRT:.2f} ms',
                xy=(data['Time'][Twitch.HRTForceIndex], data['Filtered Torque'][Twitch.HRTForceIndex]),
                xycoords='data',
                xytext=(0, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=annotate_fontsize)
            
            graph.text(
                x=0.5, y=.98,
                s=Title,
                transform=Graph.transAxes,
                horizontalalignment='center',
                fontsize=annotate_fontsize,
                verticalalignment='center')
            
            graph.text(
                x=0.5, y=0.95,
                s=f'RTD={Twitch.RTD:.2f} {plotting.RTD_Units} \n'
                    f'Peak Torque={Twitch.PeakTorque:.2f} {plotting.TorqueUnits}',
                transform=Graph.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))

        fig=plt.figure()
        fig.canvas.mpl_connect('key_press_event', plotting.keyPress)
        subplotYmax=max([ControlTwitch.peakTorque, LowHzTwitch.peakTorque, PotentiatedTwitch.peakTorque]) * 1.5
        subplotYmin=data['Filtered Torque'].iloc[29000] - (data['Filtered Torque'].iloc[29000] - data['Filtered Torque'].min()) / 2

        gridSpecs=GridSpec(2, 2, fig)
        fullTest=fig.add_subplot(gridSpecs[0, :])
        fullTest.plot(
            data['Time'][25000:65000],
            data['Filtered Torque'][25000:65000], 
            color=Colors.Black
        )
        fullTest.plot(
            data['Time'][Tetanus.RTDStartIndex:Tetanus.RTDEndIndex], 
            data['Filtered Torque'][Tetanus.RTDStartIndex:Tetanus.RTDEndIndex], 
            color=Colors.SkyBlue)
        fullTest.plot(
            Tetanus.peakTorqueData['Time'], 
            Tetanus.peakTorqueData['Filtered Torque'], 
            color=Colors.SeaGreen)
        
        fullTest.set_xlabel('Time (s)')
        fullTest.set_ylabel(plotting.torqueLabel)
        fullTest.set_ylim(-0.5, data['Filtered Torque'].max() * 1.5)
        fullTest.text(
                x=0.60, y=0.85,
                s=f'RTD={Tetanus.RTD:.2f} {plotting.rtdUnits} \n'
                    f'Peak Torque={Tetanus.peakTorque:.2f} {plotting.torqueUnits}',
                transform=fullTest.transAxes,
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))
        fullTest.text(
                x=0.33, y=0.5,
                s=f'Peak Torque={LowHzTwitch.peakTorque:.2f} {plotting.torqueUnits}',
                transform=fullTest.transAxes,
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))
        
        controlTwitchGraph=fig.add_subplot(gridSpecs[1, 0])
        controlTwitchGraph.set_ylabel(plotting.torqueLabel)

        potentiatedTwitchGraph=fig.add_subplot(gridSpecs[1, 1])
        potentiatedTwitchGraph.tick_params
        potentiatedTwitchGraph.axes.spines.left.set_visible(False)
        potentiatedTwitchGraph.axes.yaxis.set_visible(False)

        for graph, Twitch, Title, in zip([controlTwitchGraph, potentiatedTwitchGraph], [ControlTwitch, PotentiatedTwitch], ['Control Twitch', 'Potentiated Twitch']):

            PLFFD_Graphing()

        coords=plt.ginput(n=2, show_clicks= True, timeout=9999)


        CTData=data[ControlTwitch.RTDStartIndex - 200:ControlTwitch.TwitchEndIndex]
        PTData=data[PotentiatedTwitch.RTDStartIndex - 200:PotentiatedTwitch.TwitchEndIndex]

        if coords:
            for TwitchData, Graph, Twitch, Title, in zip([CTData, PTData],[controlTwitchGraph, potentiatedTwitchGraph], [ControlTwitch, PotentiatedTwitch], ['Control Twitch', 'Potentiated Twitch']):
                
                CT_NewX=round(coords[0][0] * 10000)
                PT_Newx=round(coords[1][0] * 10000)
                
                for NewX in [CT_NewX, PT_Newx]:
                    if NewX in TwitchData.index:
                        Twitch.HRTForceIndex=NewX
                    else:
                        NewX=Twitch.HRTForceIndex

                Twitch.HRT=(data['Time'][Twitch.HRTForceIndex] - data['Time'][Twitch.PeakTwitchIndex]) * 1000

                Graph.clear()

                PLFFD_Graphing()

        fig.canvas.draw()
        plt.show()

    for Twitch in [ControlTwitch, LowHzTwitch, Tetanus, PotentiatedTwitch]:
        PLFFD_Data[Twitch.Name]={
            f"{filenameInfo['timepoint']} {filenameInfo['recovery timepoint']} PLFFD {Twitch.Name} TQ": Twitch.peakTorque,
            f"{filenameInfo['timepoint']} {filenameInfo['recovery timepoint']} PLFFD {Twitch.Name} RTD": Twitch.RTD,
            f"{filenameInfo['timepoint']} {filenameInfo['recovery timepoint']} PLFFD {Twitch.Name} HRT": Twitch.HRT}

    return PLFFD_Data
