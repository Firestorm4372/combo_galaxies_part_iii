import csv

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

class Analyse():
    def __init__(self, folder_name:str, selection_name:str, data_path:str='.data') -> None:
        self.folder_name = folder_name
        self.selection_name = selection_name
        self.data_path = data_path

        # path to the overall catalog, and individual selection
        self.folder_path = f'{data_path}/{folder_name}'
        self.selection_folder_path = f'{self.folder_path}/{selection_name}'

        # catalog description
        self.catalog_name = folder_name.split('_', maxsplit=1)[1]
        with open(f'{self.folder_path}/description') as f:
            self.catalog_description = f.readline()
        
        # selection properties
        with open(f'{self.selection_folder_path}/properties.csv') as f:
            lines = [row for row in csv.reader(f)]
            self.selection_description = lines[0][1]
            self.z_vals = lines[1][1:]
            self.combinations = lines[2][1:]
            self.filters = lines[3][1:]

        with fits.open(f'{self.selection_folder_path}/selection.fits') as hdul:
            self.all_galaxies = Table(hdul[1].data)

        self.eazy_photoz_path = f'{self.selection_folder_path}/eazy_out/photoz'
        
        with fits.open(f'{self.eazy_photoz_path}.data.fits') as hdul:
            self.zbest:np.ndarray = hdul[1].data
            self.zgrid:np.ndarray = hdul[2].data
            self.chi2:np.ndarray = hdul[3].data
            self.coeffs:np.ndarray = hdul[4].data
        with fits.open(f'{self.eazy_photoz_path}.zout.fits') as hdul:
            self.zout = Table(hdul[1].data)


    def _calc_single_combination(self, zbests, chi2s) -> tuple[float, float, np.ndarray]:
        """Combine the given zbests and chi2s to produce either mean z, or the min of the combined chi2.
        
        Parameters
        ----------
        zbests : array_like
            Best fit photo z values from EAZY
        chi2 : array_like
            The chi squared distributions for each of the galaxies.
                Should be shaped `(galaxy, chi2)`


        Returns
        -------
        z_mean : float
            Mean of the zbests
        z_chi2 : float
            Min of the combined chi squared distribution
        chi2_combo : ndarray
            The combined chi squared distribution of the galaxies
        """

        z_mean = np.mean(zbests)

        chi2_combo = np.sum(chi2s, axis=0)
        z_chi2 = self.zgrid[np.argmin(chi2_combo)]

        return z_mean, z_chi2, chi2_combo

    def _data_single_combination(self, indiv_range:tuple) -> None:
        """
        Get all data points for comparision of integrated to combination.
        Append to the various attributes of object
        
        Parameters
        ----------
        indiv_range : tuple
            The range of the individual galaxies (not including the integrated)
        """
        # get idx of the integrated galaxy
        int_idx = indiv_range[0] - 1
        z_int = self.zbest[int_idx]
        chi2_int = self.chi2[int_idx]

        indiv_slice = slice(*indiv_range)

        zbests = self.zbest[indiv_slice]
        chi2s = self.chi2[indiv_slice]
        z_mean, z_chi2, chi2_combo = self._calc_single_combination(zbests, chi2s)
        
        self.z_int.append(z_int)
        self.z_mean.append(z_mean)
        self.z_chi2.append(z_chi2)
        self.chi2_int.append(chi2_int)
        self.chi2_combo.append(chi2_combo)

    def _create_dataframe(self) -> None:
        self.combo_galaxies = pd.DataFrame({
            'z_red': self.z_red,
            'combo': self.combo,
            'z_int': self.z_int,
            'z_mean': self.z_mean,
            'z_chi2': self.z_chi2,
            'chi2_int': self.chi2_int,
            'chi2_combo': self.chi2_combo,
        })

    def combine(self) -> None:
        """
        Perform the combination calculations, with mean and chi2.
        then create Pandas Dataframe with all data
        """
        
        # reset, will then be filled
        self.z_red = []
        self.combo = []
        self.z_int = []
        self.z_mean = []
        self.z_chi2 = []
        self.chi2_int = []
        self.chi2_combo = []

        # go through each galaxy, see if a combo or not
        for i, galaxy in enumerate(self.all_galaxies):
            if galaxy['combo'] != 1:
                indiv_range = (i+1, i+1 + galaxy['combo'])

                # append zred and combo
                self.z_red.append(galaxy['zred'])
                self.combo.append(galaxy['combo'])
                self._data_single_combination(indiv_range)

        self._create_dataframe()

def main() -> None:
    pass

if __name__ == '__main__':
    main()
