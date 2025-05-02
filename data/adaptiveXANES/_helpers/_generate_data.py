import glob
import os
import pandas as pd
import numpy as np

# From Ming Du's auto_beamline_ops 
# https://github.com/AdvancedPhotonSource/auto_beamline_ops
class SpectroscopyDataset:

    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.data = None
        self.energies_ev = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
class LTORawDataset(SpectroscopyDataset):
    def __init__(self, *args, filename_pattern='*', **kwargs):
        super().__init__(*args, **kwargs)
        filelist = glob.glob(os.path.join(self.path, filename_pattern))
        filelist.sort()
        self.data = []
        for i, fname in enumerate(filelist):
            table = pd.read_table(fname, comment='#', header=None, sep='\s+')
            if i == 0:
                self.energies_ev = table[0].to_numpy()
            else:
                assert len(table[0].to_numpy()) == len(self.energies_ev), \
                        "Inconsistent number of points at {} ({} vs {}).".format(
                            i, len(table[0].to_numpy()), len(self.energies_ev))
            data = np.log(table[2] / table[3]).to_numpy()
            self.data.append(data)
        self.data = np.stack(self.data)

def extract_data(input_path, data_file):
    dataset = LTORawDataset(input_path, filename_pattern="LTOsample3.[0-9]*")
    data_all_spectra = dataset.data
    energies = dataset.energies_ev
    data = data_all_spectra[0]
    df = pd.DataFrame({'Energy_eV': energies, 'Absorption': data})
    df.to_csv(data_file, index=False)

if __name__ == "__main__":
    input_path = "./raw_data" 
    data_file = "../data/xanes_spectrum.csv"

    if not os.path.exists("../data"):
        os.makedirs("../data")
    
    extract_data(input_path, data_file)