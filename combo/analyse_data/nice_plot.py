import os

import numpy as np
import pandas as pd
import seaborn as sns

from . import analyse_data

class SaveData(analyse_data.ProcessData):
    """
    Save the combo_galaxies. 
    """

    def __init__(self, folder_name: str, selection_name: str, data_path: str = '.data') -> None:
        super().__init__(folder_name, selection_name, data_path)

    def _write_errors(self) -> None:
        for z_method in ('z_int', 'z_mean', 'z_chi2'):
            zred = np.asarray(self.df_combo['z_red'])
            z = np.asarray(self.df_combo[z_method])

            difference = z - zred
            error = difference / (1 + zred)

            loc = self.df_combo.columns.get_loc(z_method) + 1
            colname = f"err_{z_method.split('_')[-1]}"
            self.df_combo.insert(loc, colname, error)
    
    def save_dataframe(self) -> None:
        """
        Save the dataframe to `selection_folder_path/combo_galaxies.csv` and set `dataframe_path` as such.

        Note this does not save the chi2 values.
        """
        self.combine()

        self.df_combo = self.combo_galaxies[['z_red', 'combo', 'combo_frac_err', 'z_int', 'z_mean', 'z_chi2']]
        self.df_combo.rename(columns={'combo_frac_err': 'frac_err'}, inplace=True)
        self.df_combo.index.name = 'idx'

        self._write_errors()

        self.dataframe_path = f'{self.selection_folder_path}/combo_galaxies.csv'
        self.df_combo.to_csv(self.dataframe_path)


class NicePlots():
    """
    Produce nice plots with seaborn for reports etc.

    Parameters
    ----------
    combo_galaxies_file_path : str
        File path to the `combo_galaxies.csv` file.
    figure_save_path : str, default None
        Where to save produced figures.
        If default of None, will save in same directory as `combo_galaxies_file_path` in folder called `figures`.
        If provided, expects to have been created.
        (Doesn't include final `/`).
    """

    def __init__(self, combo_galaxies_file_path:str, figure_save_path:str=None) -> None:
        # extract relevant filepaths
        self.combo_galaxies_file_path = combo_galaxies_file_path
        self.folder_path = '/'.join(combo_galaxies_file_path.split('/')[:-1])
        
        # get the dataframe
        self.df = pd.read_csv(self.combo_galaxies_file_path)

        # set figure save path
        if figure_save_path == None:
            self.figure_save_path = f'{self.folder_path}/figures'
            os.makedirs(self.figure_save_path, exist_ok=True)
        else:
            self.figure_save_path = figure_save_path

