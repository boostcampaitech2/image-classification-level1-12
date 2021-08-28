from torch.utils.data import Dataset
from PIL import Image

class MaskDataset(Dataset):
    def __init__(self, data, transform= None, train=True):
        self.data = data
        self.classes = self.data.columns.values.tolist()
        self.transform = transform
        self.train = train

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):
        X = Image.open(self.data['locate'].iloc[idx])

        if self.transform is not None:
            X = self.transform(X)
    
        if self.train:
            y = self.data['label'].iloc[idx]
            return X,y
        
        else:
            return X