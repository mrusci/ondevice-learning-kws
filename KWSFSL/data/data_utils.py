import torch

            
            
class SetDataset:
    def __init__(self, dl_list):
        self.sub_dataloader = dl_list

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)            