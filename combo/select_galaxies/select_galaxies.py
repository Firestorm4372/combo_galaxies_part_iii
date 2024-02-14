import os
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

        # add in combo column to catalog
        self.catalog.add_column(1, 2, 'combo')

        # extract filters
        with open(filter_list) as f:
            read = csv.reader(f)
            self.filters:list[str] = [r for r in read][0]
        self.filter_errors = [f'E{filt[1:]}' for filt in self.filters]

        self.last_id:int = self.catalog[-1]['id']

    def next_id(self) -> int:
        """Returns the next id to be used, and iterates previous by one
        (up to what has just been returned)."""
        self.last_id += 1
        return self.last_id

    def _select_galaxies(self, z:float, number:int) -> Table:
        """Select `number` galaxies at redshfit `z`"""
        mask = (self.catalog['zred'] == z)
        sub = self.catalog[mask]
        rng = np.random.default_rng()
        return Table(rng.choice(sub, number, replace=False))
    
    def _one_filter_errors(self, filter_vals, filter_combo:float=None) -> tuple[np.ndarray, float]:
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
        """

        if filter_combo == None:
            filter_combo = np.sum(filter_vals)
        
        # frac_error of the singles squared is combo frac_error squared * square of sum / sum of squares
        single_frac_error = self.frac_error * np.sqrt(filter_combo**2 / np.sum(np.square(filter_vals)))
        error_vals = np.multiply(single_frac_error, filter_vals)

        error_combo = self.frac_error * filter_combo
        return error_vals, error_combo
        
    def _combo_and_errors(self, single_galaxies:Table) -> Table:
        """
        Create new galaxy as a combo of the filters.
        Add in errors for all filters.
        
        Parameters
        ----------
        single_galaxies : Table
            Table of the single galaxies to add a combo to
        
        Returns
        -------
        galaxies : Table
            With combo galaxy and all errors
        """

        combo = len(single_galaxies)
        galaxies = single_galaxies.copy()
        # add in new galaxy row, with no data
        galaxies.insert_row(0, {'id': self.next_id(), 'zred': single_galaxies['zred'][0], 'combo': combo})

        for filt, err in zip(self.filters, self.filter_errors):
            filt_combo = np.sum(single_galaxies[filt])
            # add filter combo to relevant filter
            galaxies[filt][0] = filt_combo

            err_vals, err_combo = self._one_filter_errors(single_galaxies[filt], filt_combo)
            galaxies.add_column([err_combo, *err_vals], name=err)

        return galaxies


    def create_galaxies_table(self, z_vals:list[float], combinations:list[int], frac_error:float=0.1) -> Table:
        """
        Select at each `z_val`, some random galaxies for each `combination`.
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

        Returns
        -------
        all_galaxies : Table
            Astropy table of all the galaxies.
            Also stored in attribute `self.all_galaxies`.
        """

        self.z_vals = z_vals
        self.combinations = combinations
        self.frac_error = frac_error

        list_galaxies:list[Table] = []

        for z in z_vals:
            for combo in combinations:
                # extra relevant number of the single galaxies
                single_galaxies = self._select_galaxies(z, combo)
                galaxies = self._combo_and_errors(single_galaxies)
                list_galaxies.append(galaxies)

        # create one large table
        d = dict()
        for col in list_galaxies[0].colnames:
            d[col] = [val for sub_table in list_galaxies for val in sub_table[col]]
        
        self.all_galaxies = Table(d)
        return self.all_galaxies
    

    def _create_folder(self) -> None:
        # open file with current selections, create if not exist
        with open(f'{self.folder_path}/selections.txt', 'a+') as f:
            f.seek(0)
            lines = f.readlines()
            try:
                # one more than last idx
                idx = int(lines[-2].split(' ')[0]) + 1
            except:
                idx = 0 # no previous

            self.selection_folder_name = f'{idx}_{self.selection_name}'
            self.selection_folder_path = f'{self.folder_path}/{self.selection_folder_name}'
            # create folder
            os.makedirs(self.selection_folder_path, exist_ok=True)
            # append name and description to the sets.txt
            f.writelines(f'{idx} {self.selection_name}\n\t{self.selection_description}\n')      


    def save_galaxies_table(self, name:str='name', description:str='description') -> None:
        """Save the current `all_galaxies` table. With `name` and `description`."""
        self.selection_name = name
        self.selection_description = description

        self._create_folder()
        
        # add description, frac_error, z_vals, combinations, filters
        with open(f'{self.selection_folder_path}/properties.csv', 'w+') as f:
            csv_write = csv.writer(f)

            rows = [
                ['description', self.selection_description],
                ['frac_error', self.frac_error],
                ['z_vals', *self.z_vals],
                ['combinations', *self.combinations],
                ['filters', *self.filters]
            ]
            csv_write.writerows(rows)
        
        self.selection_table_path = f'{self.selection_folder_path}/selection.fits'
        self.all_galaxies.write(self.selection_table_path, format='fits', overwrite=True)

    def _convert_nano_to_micro_Jy(self) -> None:
        for col_name in [*self.filters, *self.filter_errors]:
            self.all_galaxies[col_name][:] *= 1e-3

    def _EAZY_file_rows(self) -> list[list]:
        column_names = ['id', 'zred', 'combo', *self.filters, *self.filter_errors]
        EAZY_rows = [column_names]

        self._convert_nano_to_micro_Jy()
        for row in self.all_galaxies:
            EAZY_rows.append([*row[column_names]])

        return EAZY_rows

    def save_EAZY_file(self) -> None:
        EAZY_rows = self._EAZY_file_rows()
        with open(f'{self.selection_folder_path}/EAZY_input.csv', 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerows(EAZY_rows)

    
    def create_and_save_galaxies(self, z_vals:list[float], combinations:list[int], frac_error:float=0.1,
                                 name:str='name', description:str='description') -> Table:
        """
        Select at each `z_val`, some random galaxies for each `combination`.
        Then apply relevant errors and create relevant flux combinations.
        Put all in a `Table` and save to FITS also.
        Also save EAZY file.
        
        Parameters
        ----------
        z_vals : list[float]
            List of z values to create the combos at
        combos : list[int]
            List of size of each combination (e.g. `[2,5]` will give combinations of 2 and of 5 galaxy filter fluxes)
        frac_error : float, default 0.1
            Size of fractional error in the combination galaxies.
            From this, errors in the galaxies that make up the combo can be found.
        name : str, Default 'name'
            Name to save under
        description : str, Default 'description'
            Description of selection

        Returns
        -------
        all_galaxies : Table
            Astropy table of all the galaxies.
            Also stored in attribute `self.all_galaxies`.
        """

        self.create_galaxies_table(z_vals, combinations, frac_error)
        self.save_galaxies_table(name, description)
        self.save_EAZY_file()
        return self.all_galaxies



def main() -> None:
    sel = Select('7_large')
    sel.create_galaxies_table([2,3,10], [2,3,5])
    print(sel.all_galaxies)

if __name__ == '__main__':
    main()