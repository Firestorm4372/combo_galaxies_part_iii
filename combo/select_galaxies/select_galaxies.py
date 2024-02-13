import csv

import numpy as np
from astropy.io import fits
from astropy.table import Table

class Select():
    def __init__(self, folder_name:str, data_path:str='.data', filter_list:str='filter_list.csv') -> None:
        # directory names
        self.folder_name = folder_name
        self.data_path = data_path
        self.folder_path = f'{data_path}/{folder_name}'
        
        # get name and description
        self.name = folder_name.split('_', maxsplit=1)[1]
        with open(f'{self.folder_path}/description') as f:
            self.description = f.readline()

        # catalog path
        self.catalog_path = f'{self.folder_path}/catalog.fits'
        # open catalog
        with fits.open(self.catalog_path) as hdul:
            self.catalog = Table(hdul[1].data)

        # extract filters
        with open(filter_list) as f:
            read = csv.reader(f)
            self.filters:list[str] = [r for r in read][0]
        self.filter_errors = [f'E{filt[1:]}' for filt in self.filters]


    def _select_galaxies(self, z:float, number:int) -> Table:
        """Select `number` galaxies at redshfit `z`"""
        mask = (self.catalog['zred'] == z)
        sub = self.catalog[mask]
        rng = np.random.default_rng()
        return Table(rng.choice(sub, number, replace=False))
    
    def _filter_errors(self, filter_vals, filter_combo:float=None) -> tuple[np.ndarray, float] | list[float]:
        """
        Return the different errors in the single galaxy filter, and in the combinations filter.
        Errors as in OneNote lab book at end of 12/02.
        
        Parameters
        ----------
        filter_vals : array_like
            Values in the filter across the different single galaxies
        filter_combo : float, default None
            Value in the filter for the combo galaxy. If not provided simply taken as sum of `filter_vals`

        Returns
        -------
        error_vals : numpy.ndarray
            Values of the errors across the different single galaxies
        error_combo : float
            Error in the combination filter.
            If no `filter_combo` provided, not returned.
        """

        if filter_combo == None:
            filter_combo = np.sum(filter_vals)
        
        # frac_error of the singles squared is combo frac_error squared * square of sum / sum of squares
        single_frac_error = np.sqrt(self.frac_error**2 * filter_combo**2 / np.sum(np.square(filter_combo)))
        error_vals = single_frac_error * filter_vals

        if filter_combo == None:
            return error_vals
        else:
            error_combo = self.frac_error * filter_combo
            return error_vals, error_combo
        

    def create_galaxies_table(self, z_vals:list[float], combinations:list[int], frac_error:float=0.1) -> None:
        """
        Select at each `z_val`, some random galaxies at each `combination`.
        Then apply relevant errors and create relevant flux combinations.
        Put all in a `Table` and save to FITS also.
        
        Parameters
        ----------
        z_vals : list[float]
            List of z values to create the combos at
        combos : list[int]
            List of size of each combination (e.g. `[2,5]` will give combinations of 2 and of 5 galaxy filter fluxes)
        frac_error : float, default 0.1
            Size of fractional error in the combination galaxies.
            From this, errors in the galaxies that make up the combo can be found.
        """
        self.frac_error = frac_error

        for z in z_vals:
            for combo in combinations:
                # extra relevant number of the single galaxies
                single_galaxies = self._select_galaxies(z, combo)



def main() -> None:
    sel = Select('4_test')
    print(sel.filters)
    print(sel.filter_errors)

if __name__ == '__main__':
    main()