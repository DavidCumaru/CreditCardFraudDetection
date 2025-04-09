import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self):
        self.data = pd.read_csv(self.file_path)
        print(self.data.head())
        return self.data
