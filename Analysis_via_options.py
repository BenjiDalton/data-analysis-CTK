#----- import python modules -----+
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#----- import custom modules -----+
import Plotting
from classes import FileInfo
from OptionsDictionary import options

plt.style.use('custom_style.mplstyle')

def Run(Directory=None, model: str=None, test: str=None, Graph: bool=False) -> pd.DataFrame:

    AllFiles=[]
    if isinstance(Directory, str):
        for root, subdirs, files in os.walk(Directory):
            for file in files:
                if file.lower().__contains__('store') or file.lower().__contains__('directory'):
                    continue
                AllFiles.append(os.path.join(root, file))
    if isinstance(Directory, list):
        AllFiles=Directory

    for file in AllFiles:
        fig, graph=Plotting.initialize_fig()
        for protocol in options[model].keys():
            if protocol.upper() in os.path.basename(file).upper():
                
                chosen_option=options[model][test]
                data, meta_data = options[model]['read file'](file=file, model=model, test=protocol)
                results=chosen_option['analyze'](data, meta_data, graph)

                chosen_option=options[model][protocol]

                data = options[model]['fill results'](model, results, meta_data, chosen_option['col basenames'], chosen_option['substring'])

                if Graph == True:
                    Plotting.show(xlabel=chosen_option['graphing']['x label'], ylabel=chosen_option['graphing']['y label'])
                plt.close()
    
    sorted_columns=sorted(data.columns, key=lambda column: float(re.search(r'pCa (\d+\.\d+)', column).group(1) if 'pCa' in column else np.nan))
    
    data=data.reindex(columns=sorted_columns)
    
    results=pd.DataFrame(data=data, columns=sorted_columns)

    return results

    # results.to_excel('', index=False)
