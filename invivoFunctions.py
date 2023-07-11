#----- import python modules -----+
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt

#----- import custom modules -----+
import plotting
from classes import Colors, TwitchResults


ms_to_seconds=1000

class Error(Exception):
    pass

def ReadFile(file: str=None, model: str=None, test: str=None, user: str=None) -> tuple[pd.DataFrame, dict]:
    print(os.path.basename(file))
    meta_data={'test': test, 'model': model}

    data=None
    protocol_info={}
    filename_info=defaultdict(str)
    filename_info['full filename']=file

    filename=os.path.basename(file).split('-')

    def ButterworthFilter(data: pd.DataFrame=None, cutoff: int=None, order: int=None) -> pd.Series:

        b, a=butter(order, cutoff, btype='low', analog=False, fs=10000)
        FilteredTorque=filtfilt(b, a, data)

        return FilteredTorque

    filename_info['condition']=filename[0]
    filename_info['animal']=f'{filename[0]}_{filename[1]}'
    filename_info['timepoint']=filename[2]

    if test == 'Torque-Frequency':
        filename_info['frequency']=filename[4]
    
    if test == 'Torque-Velocity':
        filename_info['ISO percent']=filename[4]

    if test == 'Recovery':
        recovery_timepoint=filename[4]
        filename_info['recovery timepoint']=f'{recovery_timepoint} min' if recovery_timepoint in {'1', '2', '5', '10'} else recovery_timepoint

    section_headers={
        'Sample Frequency (Hz)': int,
        'Scale (units/V)': int,
        'Protocol Array': int,
        'Test Data in Volts': int,
        'Sample': int}

    with open(file, encoding='latin1') as temp_file:
        textdata=[]
        line_numbers=range(1, 50)
        for idx, line in enumerate(temp_file):
            for header in section_headers.keys():
                if header in line:
                    section_headers[header]=idx
            textdata.append(line.strip())
    
            if idx > np.max(line_numbers):
                break
        
    columns ={
        0: 'Time',
        1: 'Length',
        2: 'Torque',
        11: 'Stim'}
    
    data=pd.read_table(
        file, 
        encoding='latin1',
        memory_map=True,
        low_memory=False,
        header=None,
        delim_whitespace=True,
        on_bad_lines='skip',
        skiprows=section_headers['Sample']+1,
        usecols=columns.keys(),
        names=columns.values())
        
    
    length_scale, torque_scale=map(lambda idx: float(textdata[section_headers['Scale (units/V)']].split('\t')[idx]), [1, 2])

    data['Length']=data['Length']*length_scale
    data['Torque']=data['Torque']*torque_scale
    # data=GetTextData(skiprows=8, columns={0: 'Time', 1: 'Length', 2: 'Raw Torque', 3: 'Other', 11: 'Stim'})
    # Aurora Scientific includes length and torque scale factors in their data files
    length_scale: float=float(data['Raw Torque'][0])
    torque_scale: float=float(data['Stim'][0])

    # Find start of actual data in text file
    data_start=data.index[data['Time'] == '0'][0]

    data=pd.DataFrame(data[['Time', 'Length', 'Raw Torque', 'Stim']][data_start:], dtype=float).reset_index(drop=True)

    data['Time']=data['Time'].div(10000)

    # Baseline torque values 
    if file.__contains__('TF' or 'Isotonic'):
        BaselineTorque=data['Raw Torque'].iloc[15000:16000].mean()
    if file.__contains__('PLFFD'):
        BaselineTorque=data['Raw Torque'].iloc[21000:22000].mean()
    else:
        BaselineTorque=data['Raw Torque'].iloc[0:100].mean()

    data['Raw Torque'] -= BaselineTorque

    # Scale length and torque channels based off Aurora Scientific values
    data['Length']=data['Length'] * length_scale
    data['Raw Torque']=data['Raw Torque'] * torque_scale

    # Filter torque signal with highpass (> 100 Hz) filter to eliminate stim artifacts
    data['Filtered Torque']=ButterworthFilter(data['Raw Torque'], 100, 2)

    return data, filename_info

def Find_PeakForce(data: pd.DataFrame, WindowLength: int) -> tuple[float, int]:
    temp=pd.DataFrame()
    temp['Rolling Mean']=pd.DataFrame(data.rolling(window=WindowLength, center=True).mean())

    PeakForce=temp['Rolling Mean'].max()
    try:
        PeakIndex=temp['Rolling Mean'].argmax() + data.index[0] # add first index to peak force index to get the true index of peak force
    except: 
        print(Error(data))
        PeakForce=data.max()
        PeakIndex=0 

    return PeakForce, PeakIndex

def Torque_Frequency(data: pd.DataFrame=None, filename_info: dict=None, Graph: bool=False) -> float:
    fig, graph=plotting.initialize_fig()
    # Find index of first stimulation (i.e., contraction start)
    StimIndex=data.index[data['Stim'] == 1][0]
    
    # Find the greatest 500 ms rolling average of torque signal
    # The footplate is moved to starting position at beginning of test so first portion of test (i.e., 20000 samples; 2 sec) is ignored
    if int(filename_info['frequency']) >= 80:
        WindowLength=500
        GraphWindow=int(WindowLength / 2)
        peak_torque, PeakIndex=Find_PeakForce(data=data['Filtered Torque'], WindowLength=WindowLength)
        graph.plot(data['Time'], data['Filtered Torque'], color=Colors.Black)
        graph.plot(data['Time'][PeakIndex - GraphWindow : PeakIndex + GraphWindow], data['Filtered Torque'][PeakIndex - GraphWindow : PeakIndex + GraphWindow], color=Colors.Firebrick, label='Peak Force')

    else:
        # For non-tetanic contractions, peak torque is highest single point during 500 ms window where stims are delivered
        peak_torque=data['Filtered Torque'][StimIndex : StimIndex + 5000].max()

    if int(filename_info['frequency']) >= 100:
        results=RTD_Analysis(data=data, PeakTorque=peak_torque, Graph=Graph)
    analysis_results=[peak_torque]+results
    return analysis_results

def Isotonic_contractions(data: pd.DataFrame=None, filename_info: dict=None, UserInput: bool=False) -> pd.DataFrame:
    BaselineLength=data['Length'].iloc[0:100].mean()

    if UserInput == False:
        try:
            ISO_Start=data.index[data['Length'] <= BaselineLength * 0.99][0]
        except:
            ISO_Start: int=0
        
        data=data.iloc[ISO_Start:]

    # Find first sample when footplate crosses end ROM (i.e., -18.99 degrees)
    # Define as end of contraction
    END_ROM=-20 if filename_info['timepoint'] == 'D80' or 'D120' else -18.99

    try:
        if data['Length'].min() > END_ROM:
            ISO_End=data.index[data['Length'] == data['Length'].min()][0]

        # If end ROM isn't achieved (possible during higher isotonic loads), then end ROM is final length sample in window
        if data['Length'].min() < END_ROM:

            ISO_End=data.index[data['Length'] <= END_ROM][0] - ISO_Start
        
    except:
        print(Error(f"End ROM not found for {filename_info['animal']}, {filename_info['timepoint']}"))

    data=data.iloc[:ISO_End]
    # Return dataframe containing only data during contraction
    return data

def Torque_Velocity(data: pd.DataFrame=None, meta_data: dict=None) -> tuple[float, float, float]:
    fig, graph=plotting.initialize_fig()
    contractionData=data[20000:26000].reset_index(drop=True)

    # Isolate data relevant to isotonic contractions 
    try:
        contractionData=Isotonic_contractions(data=contractionData, filename_info=meta_data['filename info'])
    except: 
        # contractions where the footplate doesn't move at all have their velocity set at 0
        # May happen when testing against high isotonic loads, following fatigue protocol, etc.
        ISO_Velocity=0
    else: 
        # Velocity calculated as difference between last and first samples of length and time channels, respectively
        LengthStart= contractionData['Length'].iloc[1]
        LengthEnd=contractionData['Length'].iloc[-1]

        TimeStart=contractionData['Time'].iloc[1]
        TimeEnd=contractionData['Time'].iloc[-1]

        ISO_Velocity=((LengthEnd - LengthStart) / (TimeEnd- TimeStart) * -1)

    ISO_Torque=contractionData['Filtered Torque'].mean()
    
    def IsotonicGraphing():
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
            s=f'Velocity={ISO_Velocity:.3f}',
            transform=plt.gca().transAxes,
            horizontalalignment='center',
            verticalalignment='center')
        graph.ylabel(plotting.JointAngle)
        graph.xlabel('Time (s)')
    
    IsotonicGraphing()
    Coords=plt.ginput(n=2, show_clicks= True)
    
    if Coords:
        contractionStart=round(Coords[0][0] * 10000)
        contractionEnd=round(Coords[1][0] * 10000)
        contractionData=data[contractionStart:contractionEnd].reset_index(drop=True)
        
        ISO_Velocity=((contractionData['Length'].iloc[-1] - contractionData['Length'].iloc[0]) / (contractionData['Time'].iloc[-1] - contractionData['Time'].iloc[0]) * -1) 

        IsotonicGraphing()
        fig.canvas.draw()

    ISO_Power=ISO_Torque * ISO_Velocity
    analysis_results=[ISO_Torque, ISO_Velocity, ISO_Power]
    return analysis_results

def Recovery_Isotonics(data: pd.DataFrame=None, meta_data: dict=None) -> tuple[float, float, float]:
    def get_isotonic_results(data: pd.DataFrame, meta_data: dict):
        try: 
            data=Isotonic_contractions(data=data, filename_info=meta_data['filename info'])
        except:
            iso_velocity=0
        else:
            iso_velocity=((data['Length'].iloc[-1] - data['Length'].iloc[1]) / (data['Time'].iloc[-1] - data['Time'].iloc[1]) * -1)
        
        iso_torque=data['Filtered Torque'].mean()
        iso_power=iso_torque * iso_velocity

        return iso_velocity, iso_torque, iso_power
    
    # Grab subset of data where isotonic contractions were performed 
    # Make sure to include at least 1000 samples (i.e., 100 ms) prior to contraction starting so baseline signals can be obtained
    Rec1_Data, Rec2_Data=[data[data_start:data_start + 6000] for data_start in [20000, 55000]]

    rec1, rec2=map(lambda data: get_isotonic_results(data, meta_data), [Rec1_Data, Rec2_Data])

    fig, Recovery_ISOs=plt.subplots(nrows=1, ncols=2, figsize=(width:= 6, height:= 4), layout='constrained')
    Recovery_ISOs[0].plot(Rec1_Data['Time'], Rec1_Data['Length'])
    Recovery_ISOs[1].plot(Rec2_Data['Time'], Rec2_Data['Length'])

 
    return rec1 if rec1[2] > rec2[2] else rec2

def Fatigue(data: pd.DataFrame=None, meta_data: dict=None) -> int:
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
        print(Error(f"{meta_data['filename info']['animal']} fatigue failed"))
   
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

def RTD_Analysis(data: pd.DataFrame=None, data_start: int=None, PeakTorque: float=None):
    """
    RTD start criteria is determined when torque signal exceeds 3 x standard deviations of baseline torque signal\n
    RTD end criteria is when torque signal exceeds 90% of peak torque (to ignore portions of curve that begins to plateau)
    
    Returns:
    - RTD (i.e., \u0394 force / \u0394 time)
    """
    fig, graph=plotting.initialize_fig()
    # contraction onset defined as the point torque exceeds 3 standard deviations of baseline
    data=data[data_start:].reset_index(drop=True)
    
    BaselineTorque_Mean=data['Filtered Torque'][19000:19500].mean()
    BaselineTorque_STDEV=np.std(data['Filtered Torque'][19000:19500])
    RTDStart_Criteria=BaselineTorque_Mean + (3 * BaselineTorque_STDEV)

    # Find the sample where stimulations began to be delivered
    StimIndex=data.loc[data['Stim'] == 1].index[0]
    WindowLength=1000
    GraphWindow=int(WindowLength / 2)
    
    # Search for the point torque exceeds defined contraction onset only in data following stimulation onset
    # So we are confident contraction onset will be defined properly
    RTDStartIndex=data[StimIndex:].loc[data['Filtered Torque'][StimIndex:] >= RTDStart_Criteria].index[0]

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

    analysis_results=[RTD, RTD_0_10, RTD_0_25, Norm_RTD, Norm_RTD_0_10, Norm_RTD_0_25]
    return analysis_results

def Twitch_Characteristics(data: pd.DataFrame=None, data_start: int=None, contraction: str=None) -> TwitchResults:

    data_ranges={
        'Control Twitch': (data_start, data_start + 3000),
        '10 Hz': (data_start, data_start + 5000),
        'Tetanus': (data_start, data_start + 10000),
        'Potentiated Twitch': (data_start, data_start + 3000)}

    data_start, data_end=data_ranges[contraction]

    SubsetData=data[data_start:data_end].reset_index(drop=True)
    StimIndex=SubsetData.index[SubsetData['Stim'] == 1][0]

    # contraction onset defined as the point torque exceeds 3 standard deviations of baseline
    BaselineTorque_Mean=SubsetData['Filtered Torque'].iloc[StimIndex - 500:StimIndex - 1].mean()
    BaselineTorque_STDEV=np.std(data['Filtered Torque'].iloc[StimIndex - 500:StimIndex - 1])
    RTDStart_Criteria=BaselineTorque_Mean + (3 * BaselineTorque_STDEV)

    PostStimData=SubsetData[StimIndex:]

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
            RTDStartIndex=RTD_StartIndex + data_start,
            RTD=RTD,
            PeakTwitchIndex=PeakTwitchIndex + data_start,
            PeakTorque=AdjustedPeakTwitch,
            HRTForceIndex=HalfForceIndex + data_start,
            HRT=Half_Relaxation_Time,
            HRTForce=HalfForce,
            TwitchEndIndex=TwitchEndIndex + data_start)

def PLFFD(data: pd.DataFrame=None, filename_info: dict=None, Graph: bool=False) -> tuple[float, float, float, float, float]:
    PLFFD_Data={}
    ControlTwitch=Twitch_Characteristics(data, data_start=29000, contraction='Control Twitch')

    LowHzTwitch=Twitch_Characteristics(data, data_start=39000, contraction ='10 Hz')

    data_start=46000
    contraction='Tetanus'
    TwitchData=RTD_Analysis(data=data, data_start=data_start)
    Tetanus=TwitchResults(
        Name=contraction,
        RTDStartIndex=TwitchData[0] + data_start,
        RTDEndIndex=TwitchData[1] + data_start,
        RTD=TwitchData[2],
        TimetoRTDEnd=TwitchData[3],
        PeakTorque=TwitchData[4],
        PeakTorqueData=TwitchData[5]
    )

    PotentiatedTwitch=Twitch_Characteristics(data=data, data_start=59000, contraction='Potentiated Twitch')

    
    if Graph == True:
        annotate_fontsize=6
        def PLFFD_Graphing():
            Graph.plot(
                data['Time'][Twitch.RTDStartIndex - 200:Twitch.TwitchEndIndex], 
                data['Filtered Torque'][Twitch.RTDStartIndex - 200:Twitch.TwitchEndIndex], 
                color=Colors.Black)
            
            Graph.set_xlabel('Time (s)')

            Graph.set_ylim(Subplot_Ymin, Subplot_Ymax)
                
            Graph.annotate(
                'RTD start',
                xy=(data['Time'][Twitch.RTDStartIndex], data['Filtered Torque'][Twitch.RTDStartIndex]),
                xycoords='data',
                xytext=(-50, 40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=annotate_fontsize) 
            Graph.annotate(
                f'HRT={Twitch.HRT:.2f} ms',
                xy=(data['Time'][Twitch.HRTForceIndex], data['Filtered Torque'][Twitch.HRTForceIndex]),
                xycoords='data',
                xytext=(0, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                fontsize=annotate_fontsize)
            
            Graph.text(
                x=0.5, y=.98,
                s=Title,
                transform=Graph.transAxes,
                horizontalalignment='center',
                fontsize=annotate_fontsize,
                verticalalignment='center')
            
            Graph.text(
                x=0.5, y=0.95,
                s=f'RTD={Twitch.RTD:.2f} {plotting.RTD_Units} \n'
                    f'Peak Torque={Twitch.PeakTorque:.2f} {plotting.TorqueUnits}',
                transform=Graph.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))

        fig=plt.figure()
        fig.canvas.mpl_connect('key_press_event', plotting.KeyPress)
        Subplot_Ymax=max([ControlTwitch.PeakTorque, LowHzTwitch.PeakTorque, PotentiatedTwitch.PeakTorque]) * 1.5
        Subplot_Ymin=data['Filtered Torque'].iloc[29000] - (data['Filtered Torque'].iloc[29000] - data['Filtered Torque'].min()) / 2

        GridSpecs=GridSpec(2, 2, fig)
        Fulltest=fig.add_subplot(GridSpecs[0, :])
        Fulltest.plot(
            data['Time'][25000:65000],
            data['Filtered Torque'][25000:65000], 
            color=Colors.Black
        )
        Fulltest.plot(
            data['Time'][Tetanus.RTDStartIndex:Tetanus.RTDEndIndex], 
            data['Filtered Torque'][Tetanus.RTDStartIndex:Tetanus.RTDEndIndex], 
            color=Colors.SkyBlue)
        Fulltest.plot(
            Tetanus.PeakTorqueData['Time'], 
            Tetanus.PeakTorqueData['Filtered Torque'], 
            color=Colors.SeaGreen)
        
        Fulltest.set_xlabel('Time (s)')
        Fulltest.set_ylabel(plotting.Torque)
        Fulltest.set_ylim(-0.5, data['Filtered Torque'].max() * 1.5)
        Fulltest.text(
                x=0.60, y=0.85,
                s=f'RTD={Tetanus.RTD:.2f} {plotting.RTD_Units} \n'
                    f'Peak Torque={Tetanus.PeakTorque:.2f} {plotting.TorqueUnits}',
                transform=Fulltest.transAxes,
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))
        Fulltest.text(
                x=0.33, y=0.5,
                s=f'Peak Torque={LowHzTwitch.PeakTorque:.2f} {plotting.TorqueUnits}',
                transform=Fulltest.transAxes,
                fontsize=annotate_fontsize,
                bbox=dict(boxstyle='round', facecolor='white'))
        
        ControlTwitchGraph=fig.add_subplot(GridSpecs[1, 0])
        ControlTwitchGraph.set_ylabel(plotting.Torque)

        PotentiatedTwitchGraph=fig.add_subplot(GridSpecs[1, 1])
        PotentiatedTwitchGraph.tick_params
        PotentiatedTwitchGraph.axes.spines.left.set_visible(False)
        PotentiatedTwitchGraph.axes.yaxis.set_visible(False)

        for Graph, Twitch, Title, in zip([ControlTwitchGraph, PotentiatedTwitchGraph], [ControlTwitch, PotentiatedTwitch], ['Control Twitch', 'Potentiated Twitch']):

            PLFFD_Graphing()

        coords=plt.ginput(n=2, show_clicks= True, timeout=9999)


        CTData=data[ControlTwitch.RTDStartIndex - 200:ControlTwitch.TwitchEndIndex]
        PTData=data[PotentiatedTwitch.RTDStartIndex - 200:PotentiatedTwitch.TwitchEndIndex]

        if coords:
            for TwitchData, Graph, Twitch, Title, in zip([CTData, PTData],[ControlTwitchGraph, PotentiatedTwitchGraph], [ControlTwitch, PotentiatedTwitch], ['Control Twitch', 'Potentiated Twitch']):
                
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
            f"{filename_info['timepoint']} {filename_info['recovery timepoint']} PLFFD {Twitch.Name} TQ": Twitch.PeakTorque,
            f"{filename_info['timepoint']} {filename_info['recovery timepoint']} PLFFD {Twitch.Name} RTD": Twitch.RTD,
            f"{filename_info['timepoint']} {filename_info['recovery timepoint']} PLFFD {Twitch.Name} HRT": Twitch.HRT}

    return PLFFD_Data
