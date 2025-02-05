import numpy as np
import cv2
import random

from skimage.io import imread
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pydicom as pyd
from torchvision import datasets, models, transforms
import os
from glob import glob

class Dataset(torch.utils.data.Dataset):

    '''
        age, height, weight, BMI, psoas_area, sat_area
    mean 47.85952381, 165.7666667, 64.46119048, 23.4655, 15.85136948, 113.4905268
    std	16.89252224, 7.327424755, 11.86174507, 4.350804001, 4.898369335, 52.79408107
    Max	86	190	104	33.78119437	394.8296484
    Min	19	146.9	32.7	3.959148	25.67129975
    
    max	97	190	104	33.78119437	394.8296484
    min	18	134.8	32.7	3.959148	22.38922119

    
    '''

    def __init__(self, datasetPath):
        self.dataset = pd.read_csv(datasetPath)
        x = self.dataset.drop(columns=['MUSCLE_DENSITY','SAT_DENSITY','ID_DATE','PNI','PSOAS_AREA','PSOAS_DENSITY','PARA_AREA','PARA_DENSITY','VAT_AREA','VAT_DENSITY','WEIGHT','SEX']) 
        y = self.dataset[['PNI']]
        x = (x - x.mean())/x.std()
        x['SEX'] = self.dataset['SEX']
        print(x)
        self.posNum = 0
        self.negNum = 0
        threshold = 45
        for i in range(len(y['PNI'])): 
            if y['PNI'].iloc[i]<threshold: 
                y['PNI'].iloc[i]= 1
                self.posNum += 1
            elif y['PNI'].iloc[i]>=threshold: 
                y['PNI'].iloc[i]= 0
                self.negNum += 1 
        self.y = y['PNI'].to_numpy()
        self.x = x.to_numpy()
        #self.max = [97,190,33.78119437,394.8296484]
        #self.min = [18,134.8,3.959148,22.38922119]
        #self.max = [97,190,33.78119437,511.441]
        #self.min = [18,134.8,3.959148,3.648]
        #self.max = [97,190,33.78119437,3.485755]
        #self.min = [18,134.8,3.959148,0.0059116]
        #self.mean = [52.475,164.071,15.060,109.468]
        #self.std = [17.651,8.036,5.060,52.301]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #target = np.zeros((2), dtype='float32')
        target = torch.tensor(self.y[idx],dtype=torch.long)
        id = str(self.dataset['ID_DATE'].iloc[idx])
        #tmpList = patDate[1:2] + patDate[3:7]
        #array = [float(i) for i in tmpList]
        #for i in range(len(self.mean)):
            #array[i] = (array[i] - self.mean[i])/self.std[i]
            #array[i] = (array[i] - self.min[i])/(self.max[i] - self.min[i])
        array = torch.tensor(self.x[idx],dtype=torch.float)
        return id, array, target 



class inferDataset(torch.utils.data.Dataset):

    '''
        age, height, weight, BMI, psoas_area, sat_area
    mean 47.85952381, 165.7666667, 64.46119048, 23.4655, 15.85136948, 113.4905268
    std	16.89252224, 7.327424755, 11.86174507, 4.350804001, 4.898369335, 52.79408107
    Max	86	190	104	33.78119437	394.8296484
    Min	19	146.9	32.7	3.959148	25.67129975
    
    max	97	190	104	33.78119437	394.8296484
    min	18	134.8	32.7	3.959148	22.38922119

    
    '''

    def __init__(self, datasetPath):
        self.dataset = pd.read_csv(datasetPath)
        x = self.dataset.drop(columns=['MUSCLE_DENSITY','SAT_DENSITY','ID','ID_DATE','PNI','PSOAS_AREA','PSOAS_DENSITY','PARA_AREA','PARA_DENSITY','VAT_AREA','VAT_DENSITY','SEX']) 
        
        y = self.dataset[['PNI']]
        x = (x - x.mean())/x.std()
        x['SEX'] = self.dataset['SEX']
        print(x)
        self.pni_value = self.dataset[['PNI']].copy()
        threshold= 45
        for i in range(len(y['PNI'])): 
            if y['PNI'].iloc[i]<threshold: 
                y['PNI'].iloc[i]= 1
            elif y['PNI'].iloc[i]>= threshold: 
                y['PNI'].iloc[i]= 0 
        self.y = y['PNI'].to_numpy()
        self.x = x.to_numpy()
        #self.max = [97,190,33.78119437,394.8296484]
        #self.min = [18,134.8,3.959148,22.38922119]
        #self.max = [97,190,33.78119437,511.441]
        #self.min = [18,134.8,3.959148,3.648]
        #self.max = [97,190,33.78119437,3.485755]
        #self.min = [18,134.8,3.959148,0.0059116]
        #self.mean = [52.475,164.071,15.060,109.468]
        #self.std = [17.651,8.036,5.060,52.301]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #target = np.zeros((2), dtype='float32')
        target = torch.tensor(self.y[idx],dtype=torch.long)
        id = str(self.dataset['ID_DATE'].iloc[idx])
        pni_value = str(self.pni_value.iloc[idx].to_numpy()[0])
      
        #tmpList = patDate[1:2] + patDate[3:7]
        #array = [float(i) for i in tmpList]
        #for i in range(len(self.mean)):
            #array[i] = (array[i] - self.mean[i])/self.std[i]
            #array[i] = (array[i] - self.min[i])/(self.max[i] - self.min[i])
        array = torch.tensor(self.x[idx],dtype=torch.float)
        return id, array, target, pni_value 