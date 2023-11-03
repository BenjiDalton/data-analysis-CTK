#----- import python modules -----+
import os
import pandas as pd
import functions.plotting as plotting
import matplotlib.pyplot as plt

#----- import custom modules -----+
import functions.singlefibreFunctions as singlefibre
import functions.invivoFunctions as invivo
from classes.classes import Colors, FileInfo
from functions.optionsDictionary import options
# plt.style.use('custom_style.mplstyle')


class Analysis:
    def __init__(self) -> None:
        pass

    def Run(Directory=None, AnalyzeTimepoint: str=None, model: str=None, test: str=None, graphBool: bool=False) -> tuple[list[FileInfo], pd.DataFrame]:

        def grab_filename_and_characteristic_info():
            if model == 'Single Fibre':
                OrganizedData['Animal']=meta_data['filename info']['animal'] 
                OrganizedData['Fibre']=meta_data['filename info']['fibre']
                for key, value in meta_data['characteristics'].items():
                    OrganizedData[key]=value
            if model == 'In Vivo':
                OrganizedData['Condition']=meta_data['filename info']['condition']
                if 'CON' in filename_info['animal']:
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
                    Filename=os.path.basename(file),
                    Animal=meta_data['filename info']['animal'],
                    Fibre=meta_data['filename info']['fibre'],
                    organizedData=OrganizedData)
            if model=='In Vivo':
                EachFileInfo=FileInfo(
                    Filename=os.path.basename(file),
                    Animal=meta_data['filename info']['animal'],
                    Timepoint=meta_data['filename info']['timepoint'],
                    organizedData=OrganizedData)
            AnalyzedFiles.append(EachFileInfo)

        def fill_results():
            results=pd.DataFrame(columns=OrganizedData.keys())
            for file in AnalyzedFiles:
                for column in file.organizedData.keys():
                    results.loc[file.Animal+file.Fibre, column]=file.organizedData[column]
            return results

        AllFiles=[]
        AnalyzedFiles: list[FileInfo]=[]
        if isinstance(Directory, str):
            for root, subdirs, files in os.walk(Directory):
                for file in files:
                    if file.lower().__contains__('store') or file.lower().__contains__('directory'):
                        continue
                    AllFiles.append(os.path.join(root, file))
        if isinstance(Directory, list):
            AllFiles=Directory
        
        
        if model=='Single Fibre': 
            if test=='pCa':
                columns=[]
                for file in AllFiles:
                    print("file: ", file, "\n test: ", test)
                    OrganizedData={}
                    chosen_option=options[model][test]
                    # if test in os.path.basename(file).lower():
                        # Read text file, grab information from filename, and returns dataframe with necessary raw data for further analyses
                    data, meta_data =singlefibre.readFile(file=file, model=model, test=test)

                    grab_filename_and_characteristic_info()

                    results=singlefibre.pCaAnalysis(data=data, metaData=meta_data)
                    print("results: ", results)
                    # def fill_organized_data(substring: str = None):
                    #     for idx, var in enumerate(chosen_option['col basenames']):
                    #         col=chosen_option['col basenames'][idx].format(substring)
                    #         print("col: ", col)
                    #         OrganizedData[col]=results[idx]
                    
                    # fill_organized_data(meta_data['filename info']['pCa'])

                    generate_FileInfo_object()

            if test=='ktr':
                columns=[]
                for file in AllFiles:
                    OrganizedData={}
                    chosen_option=options[model][test]
                    if test in os.path.basename(file).lower():
                        # Read text file, grab information from filename, and returns dataframe with necessary raw data for further analyses
                        data, meta_data, characteristics, filename_info=singlefibre.ReadFile(file=file, model=model, test=test)
                        
                        grab_filename_and_characteristic_info()



                        results=singlefibre.ktrAnalysis(data=data, Filename=os.path.basename(file), characteristics=characteristics, meta_data=meta_data, graphBool=graphBool)

                        for idx, var in enumerate(chosen_option['col basenames']):
                            col=chosen_option['col basenames'][idx].format(filename_info['pCa'])
                            OrganizedData[col]=results[idx]
                    
                        generate_FileInfo_object()

            if test=='Power':
                columns=[]
                for file in AllFiles:
                    OrganizedData={}
                    chosen_option=options[model][test]
                    data, meta_data=singlefibre.ReadFile(file=file, model=model, test=test) 


                    grab_filename_and_characteristic_info()

                    OrganizedData['Fmax'], OrganizedData['Force Velocity Dict']=singlefibre.PowerAnalysis(data=data, meta_data=meta_data)
                    
                    generate_FileInfo_object()

            if test=='rFE':
                columns=[]
                for file in AllFiles:
                    OrganizedData={}
                    chosen_option=options[model][test]

                    plotting.initialize_fig()
                    data, meta_data, characteristics, filename_info, first_frame_data, second_frame_data=singlefibre.ReadFile(file=file, model=model, test=test, user='Ben')

                    grab_filename_and_characteristic_info()

                    iso_results, rfe_results=map(lambda data: singlefibre.rFEAnalysis(data, meta_data=meta_data, CSA=characteristics['CSA'], graphBool=graphBool), [first_frame_data, second_frame_data])
                    iso_graph, rfe_graph=map(lambda result, color, label: plotting.show(result[0]['Time (ms)'], result[0]['Force in (mN)'], color, label), [iso_results, rfe_results], [Colors.Black, Colors.Firebrick], ['ISO', 'rFE'])
                    
                    for descriptor, results in zip(['ISO', 'rFE'], [iso_results, rfe_results]):
                        for idx, var in enumerate(chosen_option['col basenames']):
                            col=chosen_option['col basenames'][idx].format(filename_info['rFE Method'], descriptor, '')
                            OrganizedData[col]=results[idx+1]
                    
                    OrganizedData[f"{filename_info['rFE Method']} % rFE"]=((rfe_results[1] - iso_results[1]) / (iso_results[1]) * 100)

                    if graphBool==True: 
                        plotting.show(ylabel='Force (mN)', xlabel='Time (ms)')

                    generate_FileInfo_object()
            
            if test=='Binta':
                all_tests=['rFE', 'rFD']
                columns=[]
                for file in AllFiles:
                    OrganizedData={}
                    for specific_test in all_tests:
                        chosen_option=options[model][specific_test]
                        if specific_test.lower() in file.lower():
                            
                            plotting.initialize_fig()

                            data, meta_data, characteristics, filename_info, first_frame_data, second_frame_data=singlefibre.ReadFile(file=file, model=model, test=specific_test)
                            
                            grab_filename_and_characteristic_info()

                            if specific_test=='rFD':
                                rfd_results, iso_results=map(lambda data: singlefibre.rFEAnalysis(data, graphBool=graphBool), [first_frame_data, second_frame_data])
                                rfd_graph, iso_graph=map(lambda result, color, label: plotting.show(result[0]['Time (ms)'], result[0]['Force in (mN)'], color, label), [rfd_results, iso_results], [Colors.Black, Colors.Firebrick], ['rFD', 'ISO'])

                                for idx, var in enumerate(chosen_option['col basenames']):
                                    col=chosen_option['col basenames'][idx].format(specific_test, filename_info['protocol descriptor'])
                                    OrganizedData[col]=[results[idx+1] for results in [rfd_results, iso_results]]

                            if specific_test=='rFE':
                                iso_results, rfe_results=map(lambda data: singlefibre.rFEAnalysis(data, graphBool=graphBool), [first_frame_data, second_frame_data])
                                iso_graph, rfe_graph=map(lambda result, color, label: plotting.show(result[0]['Time (ms)'], result[0]['Force in (mN)'], color, label), [iso_results, rfe_results], [Colors.Black, Colors.Firebrick], ['ISO', 'rFE'])
                                for idx, var in enumerate(chosen_option['col basenames']):
                                    col=chosen_option['col basenames'][idx].format(specific_test, f"{filename_info['starting SL']}-{filename_info['ending SL']}")
                                    OrganizedData[col]=[results[idx+1] for results in [rfe_results, iso_results]]
                            
                            if graphBool==True:
                                plotting.show(ylabel='Force (mN)', xlabel='Time (ms)')
                                
                            generate_FileInfo_object()
                        
            if test=='Makenna':
                all_tests=['ISO', 'CONCENTRIC', 'SSC']
                columns=[]
                for file in AllFiles:
                    OrganizedData={}
                    for specific_test in all_tests:
                        chosen_option=options[model][specific_test]
                        if specific_test.lower() in file.lower():
                            data, meta_data=singlefibre.ReadFile(file=file, model=model, test=specific_test) 
                            
                            grab_filename_and_characteristic_info()

                            analysis_results=singlefibre.Makenna_Analysis(data=data, meta_data=meta_data)

                            if specific_test=='ISO':
                                sub_string=meta_data['filename info']['starting SL']

                            if specific_test=='CONCENTRIC':
                                sub_string=meta_data['filename info']['shorten speed']

                            if specific_test=='SSC':
                                sub_string=f"{meta_data['filename info']['stretch speed']} - {meta_data['filename info']['shorten speed']}"
                            
                            for idx, var in enumerate(chosen_option['col basenames']):
                                col=chosen_option['col basenames'][idx].format(sub_string)
                                OrganizedData[col]=analysis_results[idx]

                            generate_FileInfo_object()
                            if graphBool==True:
                                plotting.show(chosen_option['graphing']['x label'], chosen_option['graphing']['y label'])
                            plt.close()

        if model=='In Vivo':
            if test=='Torque-Frequency':
                AnalyzeTimepoint=['D40', 'D80', 'D120', 'D176']
                # in vivo torque-frequency
                for time in AnalyzeTimepoint:
                    for file in AllFiles:
                        OrganizedData={}
                        chosen_option=options[model][test]
                        if 'TF' in file and time in file:
                            # Read text file, grab information from filename, and returns dataframe with necessary raw data for further analyses
                            data, filename_info=invivo.ReadFile(file=file, model=model, test=test)
                            
                            grab_filename_and_characteristic_info()
                            
                            results=invivo.Torque_Frequency(data=data, filename_info=filename_info, graphBool=graphBool)
                            
                            for idx, var in enumerate(chosen_option['col basenames']):
                                col=chosen_option['col basenames'][idx].format(filename_info['timepoint'], filename_info['frequency'])
                                OrganizedData[col]=results[idx]

                            generate_FileInfo_object()

            if test=='Torque-Velocity':
                # in vivo Torque-Velocity
                for file in AllFiles:
                    OrganizedData={}
                    chosen_option=options[model][test]
                    if file.__contains__('Isotonic'):

                        data, filename_info=invivo.ReadFile(file=file, model=model, test=test)
                            
                        grab_filename_and_characteristic_info()
                        
                        results=invivo.Torque_Velocity(data=data, filename_info=filename_info, graphBool=graphBool)
                        
                        for idx, var in enumerate(chosen_option['col basenames']):
                            col=chosen_option['col basenames'][idx].format(filename_info['timepoint'], filename_info['ISO Percent'])
                            OrganizedData[col]=results[idx]

                        generate_FileInfo_object()

            if test=='Fatigue':
                # in vivo fatigue
                for file in AllFiles:
                    if file.__contains__('Fatigue'):
                        data, filename_info=invivo.ReadFile(file=file, model=model, test=test)
                            
                        grab_filename_and_characteristic_info()
                        
                        results=invivo.Fatigue(data=data, filename_info=filename_info, graphBool=graphBool)
                        
                        for idx, var in enumerate(chosen_option['col basenames']):
                            col=chosen_option['col basenames'][idx].format(filename_info['timepoint'])
                            OrganizedData[col]=results[idx]

                        generate_FileInfo_object()

            if test=='Recovery':
                # in vivo recovery
                for file in AllFiles:
                    if "PLFFD" in file and AnalyzeTimepoint in file:
                        if "ISOs" in file:
                            continue

                        data, filename_info=invivo.ReadFile(file=file, model=model, test=test)
                            
                        grab_filename_and_characteristic_info()
                        
                        results=invivo.PLFFD(data=data, filename_info=filename_info, graphBool=graphBool)
                        
                        for idx, var in enumerate(chosen_option['col basenames']):
                            col=chosen_option['col basenames'][idx].format(filename_info['timepoint'])
                            OrganizedData[col]=results[idx]

                        generate_FileInfo_object()

        FinalResults=fill_results()

        return AnalyzedFiles, FinalResults
