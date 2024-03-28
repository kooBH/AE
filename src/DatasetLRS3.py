import os, glob
import torch
from torchvision import transforms

class DatasetLRS3(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root

        # preprocessed lip crop data
        self.list_data = glob.glob(os.path.join(root,"**","*.pt"),
                                   recursive=True)
        
        self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        ])

    def __getitem__(self, index):
        data_item = self.list_data[index]
        lip = torch.load(data_item)()
        L = lip.shape[0]

        # average : 86
        if L  > 86:
            # randomly sample 86 frames
            start = torch.randint(0, L - 86, (1,)).item()
            lip = lip[start:start+86]
        else :
            # duplicate the data until 86 frames
            while L < 86:
                lip = torch.cat([lip]*2,0)
                L = lip.shape[0]
            
            if L > 86:
                lip = lip[:86]
        lip = lip/256
        lip = self.transform(lip)

        return lip

    def __len__(self):
        return len(self.list_data)


