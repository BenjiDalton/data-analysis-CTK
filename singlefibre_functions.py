#----- import python modules -----+
import os
import re
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit, differential_evolution

#----- import custom modules -----+
import Plotting
from classes import Colors

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

    filename=os.path.basename(file).split('_')
    section_headers={
        'A/D Sampling Rate': int,
        '*** Setup Parameters ***': int,
        'Diameter': int,
        '*** Test Protocol Parameters ***': int,
        'Time (ms)\tControl Function\tOptions': int,
        'Time (ms)': int,
        '*** Force and Length Signals vs Time ***': int} 
    characteristics={
        'Fiber Length': float,
        'Initial Sarcomere Length': float,
        'Diameter': float}
    test_parameters=[
        'Force-Step', 
        'Length-Step',
        'Length-Ramp',
        'Bath',
        'Data-Enable',
        'Data-Disable']

    with open(file) as temp_file:
        textdata=[]
        line_numbers=range(1, 300)
        for idx, line in enumerate(temp_file):
            if idx in line_numbers:
                for variable in characteristics.keys():
                    if variable in line:
                        characteristics[variable]=float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

                textdata.append(line.strip())
            for header in section_headers.keys():
                if header in line:
                    section_headers[header]=idx
    
            if idx > np.max(line_numbers):
                break

    characteristics['CSA']=np.pi*(float(characteristics['Diameter'])/2)**2
    protocol_info['sample rate']=float(re.findall(r"[-+]?\d*\.\d+|\d+", textdata[section_headers['A/D Sampling Rate']-1])[0])

    for param in test_parameters:
        param_time=[float(i.split('\t')[0]) for i in textdata[section_headers['Time (ms)\tControl Function\tOptions']:section_headers['*** Force and Length Signals vs Time ***']] if param in i]
        param_info=[i.split('\t')[3].strip() for i in textdata[section_headers['Time (ms)\tControl Function\tOptions']:section_headers['*** Force and Length Signals vs Time ***']] if param in i and len(i.split('\t'))>2]
        protocol_info[param]={
            'time': param_time,
            'info': param_info} 
        
        if param == 'Bath':
            bath_number=[i.split(' ')[0] for i in param_info]
            delay=[''.join(i.split(' ')[1:]) for i in param_info]
            protocol_info[param]={
                'info': {
                    'bath': bath_number,
                    'delay': delay
                },
                'time': param_time}
    
    columns={
        0: 'Time (ms)', 
        1: 'Length in (mm)', 
        2: 'Length out (mm)', 
        3: 'Force in (mN)', 
        4: 'Force out (mN)'}

    data=pd.read_table(
        file, 
        encoding='latin1',
        memory_map=True,
        low_memory=False,
        header=None,
        delim_whitespace=True,
        on_bad_lines='skip',
        skiprows=section_headers['*** Force and Length Signals vs Time ***']+2,
        usecols=columns.keys(),
        names=columns.values())

    data['Normalized Length']=data['Length in (mm)'] / characteristics['Fiber Length']

    if test == 'rFD':
        filename_info['protocol descriptor']=filename[3]
    
    if test == 'rFE':
        filename_info['starting SL']='.'.join(filename[3].split())
        filename_info['ending SL']='.'.join(filename[4].split())
        # if user == 'Ben':
        #     filename_info['rFE Method']=filename[2]

    if test in ['pCa', 'ktr']:
        filename_info['animal']=filename[0]
        filename_info['fibre']=filename[1]
        filename_info['muscle']=filename[2]
        filename_info['pCa']=filename[3][3:]

    if test == 'Power':
        power_loads=[]
        filename_info['animal']=filename[0]
        filename_info['muscle']=filename[1]
        filename_info['fibre']=filename[2]
        if filename[3].__contains__('LC'):
            filename[3]=filename[3][3:14]

        # Split end of filename by comma and append each isotonic load as int to list for future reference
        All_Loads=filename[3].split(',')
        for l in All_Loads: 
            if l.__contains__('.dat'):
                l=l.split('.')
                Load=int(l[0])
            else:
                Load=int(l)
            power_loads.append(Load)
        filename_info['power loads']=power_loads

    if test in ['SSC', 'CONCENTRIC', 'ISO', 'rFE', 'rFD']:
        filename_info['animal']=filename[0].upper()
        filename_info['fibre']=filename[1].capitalize()
        filename_info['protocol']=filename[2].capitalize()
    
    if test == 'SSC':
        filename_info['stretch speed']=filename[3].capitalize()
        filename_info['shorten speed']=filename[4].capitalize()
    
    if test == 'CONCENTRIC':
        filename_info['shorten speed']=filename[3].capitalize()

    if test == 'ISO':
        filename_info['starting SL']=filename[3].capitalize()

    if test in ['pCa', 'ktr', 'SSC', 'CONCENTRIC', 'ISO', 'Power', 'rFE', 'rFD']:
        for key, dictionary in zip(['protocol info', 'characteristics','filename info'], [protocol_info, characteristics, filename_info]):
            meta_data[key]=dictionary

        # return data, protocol_info, characteristics, filename_info
        return data, meta_data

def StiffnessAnalysis(data: pd.DataFrame, stiffness_time_seconds: float|int, sample_rate: int=10000, graph: plt.Axes = None):
    if stiffness_time_seconds < 100:
        stiffness_time=stiffness_time_seconds * sample_rate
    else:
        stiffness_time=stiffness_time_seconds
    stiffness_window=range(int(stiffness_time) - 100, int(stiffness_time) + 200)
    force_window=range(int(stiffness_time) - 5001, int(stiffness_time) - 1)
    dF=(data['Force in (mN)'][stiffness_window]).max() - (data['Force in (mN)'][force_window]).mean()
    dLo=(data['Normalized Length'][stiffness_window]).max() - (data['Normalized Length'][force_window]).mean()
    Stiffness=dF/dLo
    graph.plot(data['Time (ms)'].div(1000)[force_window], data['Force in (mN)'][force_window], color=Colors.SkyBlue, label='Peak force')

    return Stiffness

def pCaAnalysis(data: pd.DataFrame=None, meta_data: dict=None, graph: plt.Axes=None) -> tuple[pd.DataFrame, float, float]:
    """
    Define peak force as the highest rolling 500 ms average \n
    Returns:
     - Orginal dataframe (with baseline force subtracted from force column)
     - Peak force
     - Specific force (i.e., peak force / CSA)
    """
    # Subtract baseline force (first 50 ms of test) from force signal
    BaselineForce: float=data['Force in (mN)'][5000:5500].mean()
    data['Force in (mN)']=data['Force in (mN)'] - BaselineForce

    # Use a subset of the test (times corresponding to 15-30s following test start) to find peak. 
    # First 15s are ignored so that bath changes aren't captured
    SubsetData=data[150000:300000]
    
    # Find highest rolling 500 ms window in force
    WindowLength=5000
    GraphWindow=int(WindowLength / 2)
    peak_force, PeakIndex=Find_peak_force(data=SubsetData['Force in (mN)'], WindowLength=WindowLength)

    graph.plot(data['Time (ms)'].div(ms_to_seconds), data['Force in (mN)'], color=Colors.Black)
    graph.plot(data['Time (ms)'].div(ms_to_seconds)[PeakIndex - GraphWindow: PeakIndex + GraphWindow], data['Force in (mN)'][PeakIndex - GraphWindow: PeakIndex + GraphWindow], color=Colors.Firebrick, label='Peak Force')
    graph.text(
        x=0.5, y=0.1,
        s=f'Peak force={peak_force:.2f}uN', 
            transform=plt.gca().transAxes,
            horizontalalignment='center',
            verticalalignment='center')
    
    analysis_results=[peak_force, peak_force/meta_data['characteristics']['CSA']]

    return analysis_results

def ktrAnalysis(data: pd.DataFrame=None, meta_data: dict=None, graph: plt.Axes=None) -> tuple[float, float, float, pd.Series, pd.Series, float]:
    """
    model ktr \n
    Returns:
     - Stiffness (i.e., \u0394 force / \u0394 normalized length)
     - ktr
     - ktr goodness of fit
     - Estimated x and y data from modeling (can be graphed with real data to visually inspect fit)
     - Average force over the final 500 ms of test
    """

    def ktr_model(x, a, kt, c):
        return a * (1-np.exp(-kt*x)) + c

    def generate_Initial_Parameters(x_data: pd.Series=None, y_data: pd.Series=None):
        def sumOfSquaredError(parameterTuple):
            warnings.filterwarnings("ignore")
            val=ktr_model(x_data, *parameterTuple)
            return(np.sum((y_data - val) ** 2.0))

        maxY=max(y_data)
        minY=min(y_data)
        
        Max_Force_param: float=maxY - minY
        if y_data[:-500].mean() < Max_Force_param * 0.9:
            Max_Force_param=Max_Force_param * 0.9
        # Force at ktr start
        Force_at_T0: float=y_data[0] 
        
        parameterBounds=[]
        # search bounds for a (force when at plateau)
        parameterBounds.append([Max_Force_param, Max_Force_param])
        # search bounds for kt (range of values software uses to find ktr)
        parameterBounds.append([0, 30])
        # searh bounds for c (force at t=0)
        parameterBounds.append([Force_at_T0, Force_at_T0])

        # "seed" the numpy random number generator for repeatable results
        result=differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        
        return result.x

    def closest_value(target, lst):
        return numbers.index(min(lst, key=lambda x: abs(x - target)))

    ktr_shorten=data.index[data['Time (ms)'] == meta_data['protocol info']['Length-Ramp']['time'][0]]
    target=meta_data['protocol info']['Length-Ramp']['time'][0]
    numbers=meta_data['protocol info']['Length-Step']['time']
    closest_index=closest_value(target, numbers)
    
    ktr_re_sretch=meta_data['protocol info']['Length-Step']['time'][closest_index]
    stiffness_post_ktr=meta_data['protocol info']['Length-Step']['time'][closest_index+1] 

    ktr_start=data.index[data['Time (ms)'] == ktr_re_sretch][0]
    ktr_end=data.index[data['Time (ms)'] == stiffness_post_ktr-500][0]

    data['Time (ms)']=data['Time (ms)'].div(1000)
    Stiffness, graph=StiffnessAnalysis(data=data, stiffness_time_seconds=0.9, sample_rate=meta_data['protocol info']['sample rate'], graph=graph)
    
    model_data=pd.DataFrame(data[['Time (ms)', 'Force in (mN)']][ktr_start:ktr_end])

    # Find min force value after restretch occurs
    # Becomes real start to model_data
    min_force_index=model_data[['Force in (mN)']].idxmin()
    ktr_start: int=(min_force_index[0]+100) - model_data.index[0]
    x_data=np.array(model_data['Time (ms)'][ktr_start:])
    y_data=np.array(model_data['Force in (mN)'][ktr_start:])

    # Find initial parameters for curve fitting
    ktr_parameters=generate_Initial_Parameters(x_data, y_data)

    # maxfev=number of iterations code will attempt to find optimal curve fit
    maxfev:int=1000
    try:
        fitted_parameters, pcov=curve_fit(ktr_model, x_data, y_data, ktr_parameters, maxfev=maxfev)
    except:
        try:
            maxfev=5000
            fitted_parameters, pcov=curve_fit(ktr_model, x_data, y_data, ktr_parameters, maxfev=maxfev)
        except:
            print(Error(f"ktr parameters were not fit after {maxfev} iterations for file: {meta_data['full filename']}. Added to 'Files to Check'"))
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
    # Generate model predictions
    model_predictions=ktr_model(x_data, *fitted_parameters)

    # Calculate error
    Max_Force_error: float=np.sqrt(pcov[0, 0])
    ktr_error: float=np.sqrt(pcov[1, 1])
    Force_at_T0_error: float=np.sqrt(pcov[2, 2])
    ktr: float=fitted_parameters[1]
    abs_error=model_predictions - y_data

    SE: float=np.square(abs_error)  # squared errors
    MSE: float=np.mean(SE)  # mean squared errors
    RMSE: float=np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    GoodnessFit: float=1.0 - (np.var(abs_error) / np.var(y_data))

    x_model: np.array=np.linspace(min(x_data), max(x_data), 100)
    y_model: np.array=ktr_model(x_model, *fitted_parameters)
    
    ktrForce=data['Force in (mN)'][-5000:].mean()
    
    graph.plot(data['Time (ms)'], data['Force in (mN)'], color=Colors.Black, label='Raw')
    graph.plot(x_data, y_data, color=Colors.AuroraColor, label='Data fitted')
    graph.plot(x_model, y_model, color=Colors.Firebrick, label='Fit')
    # graph.plot(data['Time (ms)'][3999:8998], data['Force in (mN)'][3999:8998])
    
    graph.text(
        x=0.5, y=0.1,
        s=f'ktr={ktr:.3f}\n'
            f'Goodness of fit={GoodnessFit * 100:.2f}%', 
        transform=plt.gca().transAxes,
        horizontalalignment='center',
        verticalalignment='center')

    analysis_results=[ktrForce, ktrForce/meta_data['characteristics']['CSA'], ktr, GoodnessFit, Stiffness, Stiffness/meta_data['characteristics']['CSA']]

    return analysis_results

def PowerAnalysis(data: pd.DataFrame=None, meta_data: dict=None) -> tuple[float, dict]:
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
    ForceVelocityData={}
    
    clamp_times=meta_data['protocol info']['Force-Step']['time']
    power_loads=meta_data['filename info']['power loads']

    f_max=data['Force'][(195000 - 5000):195000].max()

    for idx, Load in enumerate(power_loads):

        Rows=range(int(clamp_times[idx] + 500), int(clamp_times[idx+1]))
        Start=Rows[0]
        End=Rows[-1]

        Force: float=data['Force'][Start:End].mean()
    
        try:
            Velocity:float=((data['Length'][End] - data['Length'][Start]) / (data['Time'][End] - data['Time'][Start]) * -1)
        except:
            Velocity=0

        NormForce: float=Force / meta_data['characteristics']['CSA']
        NormVelocity: float=Velocity / meta_data['characteristics']['Fibre Length']
        
        ForceVelocityData[Load]={
            f'{Load}% Active Force': Force,
            f'{Load}% Specific Force': NormForce,
            f'{Load}% Velocity': Velocity,
            f'{Load}% Normalized Velocity': NormVelocity}
        
    TextOffset=0
    AllColors=[Colors.PowerColors[Load] for Load in power_loads]
    
    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', Plotting.KeyPress)
    
    GridSpecs=GridSpec(2, 1, fig)
    ForceGraph=fig.add_subplot(GridSpecs[0, 0])
    LengthGraph=fig.add_subplot(GridSpecs[1, 0])

    for Plot, Variable in zip([ForceGraph, LengthGraph], ['Force', 'Length']):
        Plot.plot(
            data['Time'][190000:215000], 
            data[Variable][190000:215000], 
            color=Colors.Black, 
            label=Variable)
        
        for LineColor, Clamp in zip(AllColors, clamp_times):
            
            CalculationStart=int(Clamp[0] + 500)
            CalculationEnd=int(Clamp[1])

            Force:float=(data['Force'][CalculationStart:CalculationEnd].mean())
            Velocity:float=((data['Length'][CalculationEnd] - data['Length'][CalculationStart]) / (data['Time'][CalculationEnd] - data['Time'][CalculationStart]))

            Plot.plot(
                data['Time'][CalculationStart:CalculationEnd], 
                data[Variable][CalculationStart:CalculationEnd], 
                color=LineColor, 
                label='Calculations')

            if Variable == 'Force':
                # Plot.text(
                #     x=.3,
                #     y=.7 - TextOffset,
                #     s=f'{Force:.3f}',
                #     transform=Plot.transAxes,
                #     horizontalalignment='center',
                #     verticalalignment='center',
                #     fontdict=dict(size=12, color=LineColor))
                
                Plot.set_ylabel('Force (mN)', fontname='Arial')
                Plot.set_yticklabels([])
                Plot.set_yticks([])

            if Variable == 'Length':
                # Plot.text(
                #     x=.3,
                #     y=1.5 - TextOffset, 
                #     s=f'{Velocity * -1:.3f}',
                #     transform=Plot.transAxes,
                #     horizontalalignment='center',
                #     verticalalignment='center',
                #     fontdict=dict(size=12, color=LineColor))
                
                Plot.set_ylabel('Length (mm)', fontname='Arial')
                Plot.set_yticklabels([])
                Plot.set_yticks([])
                Plot.set_xticklabels([])
                Plot.set_xticks([])

            TextOffset += .2
    
    for string, xcoord, color in zip(power_loads, [196500, 201500, 206500, 211500], AllColors):
        # fig.text(
                # x=xcoord,
                # y=.9,
                # s=string,
                # horizontalalignment='center',
                # verticalalignment='center',
                # fontdict=dict(size=12, color=color))

        ForceGraph.annotate(
            f'{string}%',
            xy=(data['Time'][xcoord], (data['Force'][190000])),
            xycoords='data',
            fontsize=20,
            fontname='Arial',
            color=color)

    LengthGraph.set_xlabel('Time (s)', fontname='Arial')
    ForceGraph.axes.spines.bottom.set_visible(False)
    ForceGraph.axes.xaxis.set_visible(False)

    return f_max, ForceVelocityData

def rFEAnalysis(data: pd.DataFrame=None, meta_data: dict=None, graph: plt.Axes=None, graph_color: Colors=None, label: str=None) -> tuple[pd.DataFrame, float, float, float]:
    """
    Returns:
     - Orginal dataframe (with baseline force subtracted from force column)
     - Peak force (i.e., average force during 500 ms prior to rFE stretch)
     - Passive force (i.e., average force during final 500 ms of test)
     - Stiffness (i.e., \u0394 force / \u0394 normalized length)
    """
    protocol_info=meta_data['protocol info']
    activation_start=protocol_info['Bath']['time'][protocol_info['Bath']['info']['bath'].index('4')]

    stiffness_time=data.index[data['Time (ms)']==activation_start + 40000][0]
    baseline_window=range(int(100000), int(105000))
    force_window=range(int(stiffness_time) - 5000, int(stiffness_time) - 1)
    passive_window=range(len(data) - 5001, len(data)-1)

    # Calculadora
    data['Force in (mN)']=data['Force in (mN)'] - np.mean(data['Force in (mN)'][baseline_window])

    graph_data=pd.DataFrame(data[['Time (ms)', 'Force in (mN)']])
    graph_data['Time (ms)']=graph_data['Time (ms)']

    stiffness=StiffnessAnalysis(data=data, stiffness_time_seconds=stiffness_time, graph=graph)

    peak_force=np.mean(data['Force in (mN)'][force_window])
    specific_force=peak_force / meta_data['characteristics']['CSA']

    passive_force=np.mean(data['Force in (mN)'][passive_window])

    graph_linewidth=0.8
    data=data[250000:]    
    
    graph.plot(graph_data['Time (ms)'], graph_data['Force in (mN)'], color=graph_color, linewidth=graph_linewidth, label=label)

    return {
            f"{label} Force": peak_force,
            f"{label} Specific Force": specific_force,
            f"{label} Passive Force": passive_force,
            f"{label} Stiffness": stiffness,
            }

def get_contraction_data(data: pd.DataFrame=None, idx: int=None, protocol_info: dict=None) -> pd.DataFrame:
    length_start=protocol_info['Length-Ramp']['time'][idx]
    length_end=length_start + (float(protocol_info['Length-Ramp']['info'][idx].split()[2]) * 1000)
    length_start_index=data.index[data['Time (ms)']== float(length_start)][0]
    length_end_index=data.index[data['Time (ms)']== float(length_end)][0]
    length_window=range(int(length_start_index), int(length_end_index))

    data=data.loc[length_window, ['Time (ms)', 'Length in (mm)', 'Force in (mN)']]

    return data

def work_calculation(data: pd.DataFrame, sample_rate: int=10000, graph: plt.Axes=None, graph_linecolor: Colors=None, graph_label: str=None, annotation_offset: int=20) -> float:
    ms_to_second=1000
    change_in_length=(data['Length in (mm)'].iloc[-1] - data['Length in (mm)'].iloc[0]) * -1
    cum_force=np.max(data['Force in (mN)'].cumsum())
    contraction_duration=(data['Time (ms)'].iloc[-1] - data['Time (ms)'].iloc[-0]) / ms_to_second
    work=((cum_force * change_in_length)/(contraction_duration)) / sample_rate

    graph.plot(data['Time (ms)'].div(1000), data['Force in (mN)'], color=graph_linecolor, linewidth=0.8, label=graph_label)
    graph.fill_between(data['Time (ms)'].div(1000), y1=data['Force in (mN)'], color=graph_linecolor,  alpha=0.4)
    graph.annotate(
        text=f'Work={work:.4f}',
        xy=(np.mean(data['Time (ms)'].div(1000)), np.mean(data['Force in (mN)'])),
        xycoords='data',
        xytext=(annotation_offset, 20),
        textcoords='offset points',
        arrowprops=dict(facecolor=Colors.Black, headlength=5, width=1, headwidth=5))
    
    return work

def Binta_Analysis(data: pd.DataFrame=None, meta_data: dict=None, graph: plt.Axes=None) -> dict:
    protocol_info=meta_data['protocol info']
    activation_start=protocol_info['Bath']['time'][protocol_info['Bath']['info']['bath'].index('4')]

    second_frame_index=data.index[data['Time (ms)']== float(protocol_info['Data-Enable']['time'][1])][0]
    first_frame_data=pd.DataFrame(data[:second_frame_index], dtype=float).reset_index(drop=True)
    second_frame_data=pd.DataFrame(data[second_frame_index:], dtype=float).reset_index(drop=True)
    first_frame_data['Time (ms)']=first_frame_data['Time (ms)'] - first_frame_data['Time (ms)'][0]
    second_frame_data['Time (ms)']=second_frame_data['Time (ms)'] - second_frame_data['Time (ms)'][0]

    if meta_data['filename info']['protocol'].upper() == 'RFD':
        rfd_results, iso_results=map(lambda data, color, label: rFEAnalysis(data, meta_data, graph, color, label), [first_frame_data, second_frame_data], [Colors.Black, Colors.Firebrick], ['rFD', 'ISO'])
        analysis_results=rfd_results,iso_results
    if meta_data['filename info']['protocol'].upper() == 'RFE':
        iso_results, rfe_results=map(lambda data, color, label: rFEAnalysis(data, meta_data, graph, color, label), [first_frame_data, second_frame_data], [Colors.Black, Colors.Firebrick], ['ISO', 'rFE'])
        analysis_results=iso_results, rfe_results

    return analysis_results

def Makenna_Analysis(data: pd.DataFrame=None, meta_data: dict=None, graph: plt.Axes=None) -> dict:
    protocol_info=meta_data['protocol info']
    activation_start=protocol_info['Bath']['time'][protocol_info['Bath']['info']['bath'].index('4')]
    graph_linewidth=0.8

    baseline_window=range(int(100000), int(105000))
    data['Force in (mN)']=data['Force in (mN)'] - np.mean(data['Force in (mN)'][baseline_window])

    stiffness_time=data.index[data['Time (ms)']==activation_start + 40000][0]

    data=data[:600000]    
    graph.plot(data['Time (ms)'].div(1000), data['Force in (mN)'], color=Colors.Black, linewidth=graph_linewidth, label='Full test')

    stiffness=StiffnessAnalysis(data=data, stiffness_time_seconds=stiffness_time, graph=graph)

    if meta_data['filename info']['protocol'].upper() == 'SSC':
        stretch_data, shorten_data=map(lambda number: get_contraction_data(data, number, protocol_info), [0, 1])
        peak_ECC_force=np.max(stretch_data['Force in (mN)'])
        stretch_work, shorten_work=map(
            lambda data, linecolor, label, annotation_offset: work_calculation(data, graph=graph, graph_linecolor=linecolor, graph_label=label, annotation_offset=annotation_offset), 
                [stretch_data, shorten_data], 
                [Colors.Firebrick, Colors.DeepBlue], 
                ['Stretch', 'Shorten'], 
                [-100, 20])
        net_work=shorten_work + stretch_work
        analysis_results=[peak_ECC_force, stretch_work, shorten_work, net_work, stiffness]

    if meta_data['filename info']['protocol'].upper() == 'ISO':
        peak_force=np.mean(data['Force in (mN)'].loc[stiffness_time - 5001:stiffness_time-1])

        analysis_results=[peak_force, stiffness]

    if meta_data['filename info']['protocol'].upper() == 'CONCENTRIC':
        shorten_data=get_contraction_data(data=data, idx=0, protocol_info=protocol_info)
        shorten_work=work_calculation(data=shorten_data, graph=graph, graph_linecolor=Colors.Firebrick, graph_label='Shortening')
        force_following_shortening=np.mean(data['Force in (mN)'].loc[stiffness_time - 5001:stiffness_time-1])

        analysis_results=[shorten_work, force_following_shortening, stiffness] 

    return analysis_results

def Find_peak_force(data: pd.DataFrame, WindowLength: int) -> tuple[float, int]:
    temp=pd.DataFrame()
    temp['Rolling Mean']=pd.DataFrame(data.rolling(window=WindowLength, center=True).mean())

    peak_force=temp['Rolling Mean'].max()
    try:
        PeakIndex=temp['Rolling Mean'].argmax() + data.index[0] # add first index to peak force index to get the true index of peak force
    except: 
        print(Error(data))
        peak_force=data.max()
        PeakIndex=0 

    return peak_force, PeakIndex
