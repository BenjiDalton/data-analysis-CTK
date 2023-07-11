import singlefibreFunctions as singlefibre
import invivoFunctions as invivo
import pandas as pd
from classes import FileInfo
# define lists used to generate dictionary that is used to assist with specific functions based on model and test the user is analzying
# create dicts with key (used to define 'col basenames') and value (used to define 'graph y label')

pcaBasenames = {
    'Active Force': 'Force (mN)',
    'Specific Force': 'Specific Force (mN / CSA)'}
    
ktrBasenames = {
    'ktr Force': 'Force (uN)', 
    'ktr Specific Force': 'Specific Force (mN / CSA)',
    'ktr': 'ktr (s-1)', 
    'Goodness of fit': 'Goodness of fit (r2)', 
    'Active Stiffness': [], 
    'Specific Stiffness': []}

powerBasenames = {
    'Fmax': 'Force (mN)',
    'Active Force': 'Force (mN)',
    'Specific Force': 'Specific Force (mN / CSA)',
    'Velocity': 'Velocity (FL/s)', 
    'Normalized Velocity': 'Normalized Velocity (s-1)'}


filenameInfo=None
characteristics=None
columns=None
file=None
analyzedFiles: list[FileInfo]=[]


def fillResults(model: str = None, analysisResults = None, metaData: dict = None, cols: list = None, substring: list = None) -> pd.DataFrame:
    organizedData={}
    columns=[]
    def fillOrganizedData():
        for idx, var in enumerate(cols):

            if len(substring)>1:
                col=cols[idx].format('-'.join(metaData['filename info'][string]for string in substring))

            else:
                col=cols[idx].format(metaData['filename info'][substring[0]])
            if isinstance(analysisResults, dict):
                organizedData[col]=analysisResults[col]
            if isinstance(analysisResults, list):
                organizedData[col]=analysisResults[idx]

    def grabFilename_Characteristcs():
        if model == 'Single Fibre':
            organizedData['Animal']=metaData['filename info']['animal'] 
            organizedData['Fibre']=metaData['filename info']['fibre']
            for key, value in metaData['characteristics'].items():
                organizedData[key]=value
        if model == 'In Vivo':
            organizedData['Condition']=metaData['filename info']['condition']
            if 'CON' in metaData['filename info']['animal']:
                organizedData['Condition int']=0
            elif filenameInfo['animal'] in ['VCD_02', 'VCD_03', 'VCD_04', 'VCD_05', 'VCD_09', 'VCD_10', 'VCD_15', 'VCD_16', 'VCD_19', 'VCD_20']:
                organizedData['Condition int']=1
            else:
                organizedData['Condition int']=2
            organizedData['Animal']=metaData['filename info']['animal'] 
            organizedData['Timepoint']=metaData['filename info']['timepoint']

    def generateFileInfo():
        for key in organizedData.keys():
            if key in columns:
                continue
            columns.append(key)
        if model=='Single Fibre':
            fileInfo=FileInfo(
                Animal=metaData['filename info']['animal'],
                Fibre=metaData['filename info']['fibre'],
                organizedData=organizedData)
        if model=='In Vivo':
            fileInfo=FileInfo(
                Animal=metaData['filename info']['animal'],
                Timepoint=metaData['filename info']['timepoint'],
                organizedData=organizedData)
        analyzedFiles.append(fileInfo)

    grabFilename_Characteristcs()
    fillOrganizedData()
    generateFileInfo()

    results=pd.DataFrame(columns=organizedData.keys())
    for file in analyzedFiles:
        for column in file.organizedData.keys():
            results.loc[file.Animal+file.Fibre, column]=file.organizedData[column]
    return results

options = {
    'Single Fibre': {
        'read file': singlefibre.readFile,
        'fill results': fillResults,
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
        'fill results': fillResults,
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

for protocol, basename in zip(['pCa', 'ktr'], [pcaBasenames, ktrBasenames]):
    options['Single Fibre'][protocol].update({
        'col basenames': [*[key+' (pCa {})' for key in basename.keys()]],
        'graphing': {
            'x label': 'Time (s)',
            'y label': 'Force (mN)'
        },
        'substring': ['pCa']
    })


