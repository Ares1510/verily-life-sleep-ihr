import numpy as np
import pandas as pd
from utils.sliding_windows import sliding_window
from torch.utils.data import Dataset, DataLoader

class MESADataset(Dataset):
    def __init__(self, sub_ids):
        self.sub_files = [f'data/{sub_id}_processed.csv' for sub_id in sub_ids]
        self.X, self.y = self.get_data()

    def get_data(self):
        X, y = [], []
        for file in self.sub_files:
            df = pd.read_csv(file)
            X.append(df.iloc[:, 0].values)
            y.append(df.iloc[:, 1].values)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X_windowed = sliding_window(X, 256, 60)
        y_windowed = [[i[-1]] for i in sliding_window(y, 256, 60)]
        return np.asarray(X_windowed), np.asarray(y_windowed).squeeze()

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

           
class SleepDataLoader():
    def __init__(self, sub_ids):
        # split subject ids into train, val, and test in the ratio 0.8, 0.1, 0.1
        np.random.seed(42)
        np.random.shuffle(sub_ids)
        train_ids = sub_ids[:int(0.8 * len(sub_ids))]
        val_ids = sub_ids[int(0.8 * len(sub_ids)):int(0.9 * len(sub_ids))]
        test_ids = sub_ids[int(0.9 * len(sub_ids)):]
        self.train_dataset = MESADataset(train_ids)
        self.val_dataset = MESADataset(val_ids)
        self.test_dataset = MESADataset(test_ids)

    def train(self, batch_size=256):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
    
    def val(self, batch_size=256):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
    
    def test(self, batch_size=256):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
