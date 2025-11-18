import os
import numpy as np

mainfolder = '/gdata/fewahab/data/WSJO-3rdPaper-dataset/data/tt-individual/Babble' 
f1 = open('/ghome/fewahab/Transformer_3rd_Paper/MHSA/ShfAtn-SelfAtn-Variants/DFSA6L-HybScaled0.5-hub0.5/scripts/tt_babble.txt', 'a')   # file name
folderlist = os.listdir(mainfolder)

for index,folder in enumerate(folderlist):
    path2write = os.path.join(mainfolder,folder)
    f1.write(path2write+'\n')
    print(path2write)