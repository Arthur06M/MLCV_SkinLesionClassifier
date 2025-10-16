import os
from torch.utils.data import Dataset
from PIL import Image

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        self.classes = sorted(dataframe['dx'].unique())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        for img_dir in self.img_dirs:
            path = f"{img_dir}/{row['image_id']}.jpg"
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                break
        
        # Get label as integer
        label = self.classes.index(row['dx'])
        
        if self.transform:
            img = self.transform(img)
        
        return img, label