import singlefibre_functions as singlefibre
import invivo_functions as invivo
import pandas as pd
from classes import FileInfo
# define lists used to generate dictionary that is used to assist with specific functions based on model and test the user is analzying
# create dicts with key (used to define 'col basenames') and value (used to define 'graph y label')

pCa_basenames = {
    'Active Force': 'Force (mN)',
    'Specific Force': 'Specific Force (mN / CSA)'}
    
ktr_basenames = {
    'ktr Force': 'Force (uN)', 
    'ktr Specific Force': 'Specific Force (mN / CSA)',
    'ktr': 'ktr (s-1)', 
    'Goodness of fit': 'Goodness of fit (r2)', 
    'Active Stiffness': [], 
    'Specific Stiffness': []}

power_basenames = {
    'Fmax': 'Force (mN)',
    'Active Force': 'Force (mN)',
    'Specific Force': 'Specific Force (mN / CSA)',
    'Velocity': 'Velocity (FL/s)', 
    'Normalized Velocity': 'Normalized Velocity (s-1)'}


filename_info=None
characteristics=None
columns=None
File=None
AnalyzedFiles: list[FileInfo]=[]


def fill_results(model: str = None, analysis_results = None, meta_data: dict = None, cols: list = None, substring: list = None) -> pd.DataFrame:
    OrganizedData={}
    columns=[]
    def fill_organized_data():
        for idx, var in enumerate(cols):

            if len(substring)>1:
                col=cols[idx].format('-'.join(meta_data['filename info'][string]for string in substring))

            else:
                col=cols[idx].format(meta_data['filename info'][substring[0]])
            if isinstance(analysis_results, dict):
                OrganizedData[col]=analysis_results[col]
            if isinstance(analysis_results, list):
                OrganizedData[col]=analysis_results[idx]

    def grab_filename_and_characteristic_info():
        if model == 'Single Fibre':
            OrganizedData['Animal']=meta_data['filename info']['animal'] 
            OrganizedData['Fibre']=meta_data['filename info']['fibre']
            for key, value in meta_data['characteristics'].items():
                OrganizedData[key]=value
        if model == 'In Vivo':
            OrganizedData['Condition']=meta_data['filename info']['condition']
            if 'CON' in meta_data['filename info']['animal']:
                OrganizedData['Condition int']=0
            elif filename_info['animal'] in ['VCD_02', 'VCD_03', 'VCD_04', 'VCD_05', 'VCD_09', 'VCD_10', 'VCD_15', 'VCD_16', 'VCD_19', 'VCD_20']:
                OrganizedData['Condition int']=1
            else:
                OrganizedData['Condition int']=2
            OrganizedData['Animal']=meta_data['filename info']['animal'] 
            OrganizedData['Timepoint']=meta_data['filename info']['timepoint']

    def generate_FileInfo_object():
        for key in OrganizedData.keys():
            if key in columns:
                continue
            columns.append(key)
        if model=='Single Fibre':
            EachFileInfo=FileInfo(
                Animal=meta_data['filename info']['animal'],
                Fibre=meta_data['filename info']['fibre'],
                OrganizedData=OrganizedData)
        if model=='In Vivo':
            EachFileInfo=FileInfo(
                Animal=meta_data['filename info']['animal'],
                Timepoint=meta_data['filename info']['timepoint'],
                OrganizedData=OrganizedData)
        AnalyzedFiles.append(EachFileInfo)

    grab_filename_and_characteristic_info()
    fill_organized_data()
    generate_FileInfo_object()

    results=pd.DataFrame(columns=OrganizedData.keys())
    for File in AnalyzedFiles:
        for column in File.OrganizedData.keys():
            results.loc[File.Animal+File.Fibre, column]=File.OrganizedData[column]
    return results

options = {
    'Single Fibre': {
        'read file': singlefibre.ReadFile,
        'fill results': fill_results,
        'pCa': {
            'analyze': singlefibre.pCaAnalysis
        },
        'ktr': {
            'analyze': singlefibre.ktrAnalysis
        },
        'rFE': {
            'col basenames':[
                '{} rFE Force',
                '{} rFE Specific Force',
                '{} rFE Passive Force',
                '{} rFE Stiffness',
                '{} ISO Force',
                '{} ISO Specific Force',
                '{} ISO Passive Force',
                '{} ISO Stiffness'],
                'graphing': {
                    'x label': 'Time (ms)',
                    'y label': 'Force (mN)'},
                'substring': [
                    'starting SL', 
                    'ending SL']
                },
        'rFD': {
            'col basenames': [
                '{} rFD Force',
                '{} rFD Specific Force',
                '{} rFD Passive Force',
                '{} rFD Stiffness',
                '{} ISO Force',
                '{} ISO Specific Force',
                '{} ISO Passive Force',
                '{} ISO Stiffness'],
                'graphing': {
                    'x label': 'Time (ms)',
                    'y label': 'Force (mN)'},
                'substring': [
                    'protocol descriptor', 
                    'protocol']
                },
        'ISO': {
            'col basenames': [
                'ISO Force at {}', 
                'ISO Stiffness at {}'],
            'variables':[
                'ISO Force', 
                'ISO Stiffness'],
            'substring': [
                'starting SL'],
            'graphing': {
                'x label': 'Time (s)',
                'y label': 'Force (mN)'}
            },
        'CONCENTRIC': {
            'col basenames': [
                'Work during {}', 
                'Force following {}', 
                'Stiffness following {}'],
            'variables': [
                'Work', 
                'Force', 
                'Stiffness'],
            'substring': [
                'shorten speed'],
            'graphing': {
                'x label': 'Time (s)',
                'y label': 'Force (mN)'}},
        'SSC': {
            'col basenames': [
                '{} Peak ECC Force', 
                '{} Stretch work', 
                '{} Shorten work', 
                '{} Net work', 
                '{} SSC Stiffness'],
            'variables': [
                'Peak ECC Force', 
                'Stretch work', 
                'Shorten work', 
                'Net work', 
                'Force after shortening', 
                'SSC Stiffness'],
            'substring': [
                'stretch speed', 
                'shorten speed'],
            'graphing': {
                'x label': 'Time (s)',
                'y label': 'Force (mN)'}},
        'Makenna': {
            'analyze': singlefibre.Makenna_Analysis
        },
        'Binta': {
            'analyze': singlefibre.Binta_Analysis
        },
    },
    'In Vivo': {
        'read file': invivo.ReadFile,
        'fill results': fill_results,
        'Torque-Frequency':{
            'analyze': invivo.Torque_Frequency
        },
        'Torque-Velocity':{
            'analyze': invivo.Torque_Velocity
        },
        'Fatigue': {
            'analyze': invivo.Fatigue
        },
        'Recovery': {
            'analyze': invivo.PLFFD
        }
    }
}

for protocol, basename in zip(['pCa', 'ktr'], [pCa_basenames, ktr_basenames]):
    options['Single Fibre'][protocol].update({
        'col basenames': [*[key+' (pCa {})' for key in basename.keys()]],
        'graphing': {
            'x label': 'Time (s)',
            'y label': 'Force (mN)'
        },
        'substring': ['pCa']
    })


