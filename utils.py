import torch
from torch.utils.data import Dataset
import pandas as pd


class CodeData(Dataset): 
    def __init__(self, 
                 data: pd.DataFrame,   
                 is_filter: bool = True):
        
        self.data = data
        if is_filter: 
            condition = (self.data['Error'].isnull()) & (self.data['Correct'] == True)
            self.data = self.data[condition]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return {'operation': self.data['Op_Name'].iloc[index], 
                'kernel': self.data['Kernel_Name'].iloc[index], 
                'cuda': self.data['CUDA_Code'].iloc[index], 
                'torch':self.data['PyTorch_Code_Module'].iloc[index]
                }
    