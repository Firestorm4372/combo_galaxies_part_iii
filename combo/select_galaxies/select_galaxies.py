import os
import csv

import numpy as np
from astropy.io import fits
from astropy.table import Table


class Combine():
    """
    Given set of single galaxies, will produce the combination filter values, and errors in all filters.
    Modifies `single_galaxies` in place. 

    Attributes
    ----------
    single_galaxies : Table
        Astropy table of the individual galaxies with no errors
    combo_id : int
        The id value to assign to the combination galaxy
    filters, error_filters : list[str]
        Names of the columns of the filters, and the names of the error columns
    combo_fractional_error : float
        Fractional error that is attempted to be achieved in the combination filter. 
        Excluding the effects of error floors etc.

    magnitude_filter : str, default F277W
        Filter to base normalisation and fractional error floor off

    is_flux_normalisation : bool, default False
        Normalise all galaxies to have equal `magnitude_filter`
    is_constant_error_floor : bool default False
        Use a constant error floor across all of the individual galaxies.
        Value to use will be `const_error_floor` if `True`.
        Otherwise error floor is used as `fractional_error_floor` of `magnitude_filter`
    is_combo_in_quadrature : bool, default True
        Calculate the combination errors as quadrature of the individuals.
        NOT IMPLEMENTED NON QUADRATURE (IE WITH AN ERROR FLOOR)

    frac_error_floor : float, default 0.01
        Fractional amount of `magnitude_filter` flux that will be used as the error floor.    
    const_error_floor : float, default 0
        Value to use if using a constant error floor
    """

    def __init__(self,
                 single_galaxies:Table, combo_id:int, filters:list[str], error_filters:list[str], combo_fractional_error:float,
                 magnitude_filter:str='F277W',
                 is_flux_normalisation:bool=False, is_constant_error:bool=False, is_combo_in_quadrature:bool=True,
                 frac_error_floor:float=0.01, const_error_floor:float=0
        ) -> None:

        self.galaxies = single_galaxies
        self.filters = filters
        self.error_filters = error_filters
        self.combo_fractional_error = combo_fractional_error

        self.magnitude_filter = magnitude_filter
        self.is_flux_normalisation = is_flux_normalisation
        self.is_constant_error = is_constant_error
        self.is_combo_in_quadrature = is_combo_in_quadrature

        self.frac_error_floor = frac_error_floor
        self.const_error_floor = const_error_floor

        # values that will be assigned to the combo galaxy
        self.combo = len(self.galaxies)
        self.combo_id = combo_id
        self.zred = self.galaxies['zred'][0]

        self.combo_flux_values = dict() # combo flux values, to fill and then add to the rows
        self.combo_flux_errors = dict() # combo flux errors, to fill and then add to the rows
        self.individual_fractional_errors = dict() # fractional errors that should be applied to each individual filter to acheive combo fractional error


    def _calc_combo_flux_values(self) -> None:
        for filt in self.filters:
            self.combo_flux_values |= {filt: np.sum(self.galaxies[filt])}

    def _calc_individual_fractional_errors(self) -> None:
        # get combo flux values if not exist
        if len(self.combo_flux_values) == 0:
            self._calc_combo_flux_values()

        for filt in self.filters:
            root_square_flux = np.sqrt(np.sum(np.square(self.galaxies[filt])))
            indiv_frac_err = self.combo_fractional_error * \
                self.combo_flux_values[filt] / root_square_flux
            
            self.individual_fractional_errors |= {filt: indiv_frac_err}


    def _flux_normalisation(self) -> None:
        raise Exception('Flux normailisation not implemented. ')

    def _get_individual_errors(self) -> None:
        # to store errors from each, list corresponding to filters of list of each galaxy
        indiv_filter_errors = [[] for _ in range(len(self.filters))]

        # make sure individual filter fractional errors calculated
        if len(self.individual_fractional_errors) == 0:
            self._calc_individual_fractional_errors()

        if self.is_flux_normalisation:
            self._flux_normalisation()

        for gal in self.galaxies:
            # get error floor as either the constant, or as a fraction of magnitude
            if self.is_constant_error:
                error_floor = self.const_error_floor
            else:
                error_floor = self.frac_error_floor * gal[self.magnitude_filter]

            for i, filt in enumerate(self.filters):
                error = max(error_floor, self.individual_fractional_errors[filt] * gal[filt])
                indiv_filter_errors[i].append(error)
        
        # add in the error columns
        self.galaxies.add_columns(cols=indiv_filter_errors, names=self.error_filters)

    
    def _calc_combo_flux_errors(self) -> None:
        if self.is_combo_in_quadrature:
            for err_filt in self.error_filters:
                combo_error = np.sqrt(np.sum(np.square(self.galaxies[err_filt])))
                self.combo_flux_errors |= {err_filt: combo_error}
        else:
            raise Exception('Non quadrature combo errors not implemented')
            
    def _add_row_combo_galaxy(self) -> None:
        if len(self.combo_flux_values) == 0:
            self._calc_combo_flux_values()
        if len(self.combo_flux_errors) == 0:
            self._calc_combo_flux_errors()

        combo_galaxy = {
            'id': self.combo_id,
            'zred': self.zred,
            'combo': self.combo,
            'combo_frac_err': self.combo_fractional_error
        } | self.combo_flux_values | self.combo_flux_errors

        self.galaxies.insert_row(0, combo_galaxy)

    def full_combine(self) -> Table:
        self._get_individual_errors()
        self._add_row_combo_galaxy()

        return self.galaxies


class RandomDraw():
    """
    Will add Gaussian variation to all SED values, based on the error in that filter.
    The combined SED values are then updated to be the sum of the single SEDs (errors are left unchanged).

    Attributes
    ----------
    galaxies : Table
        The complete galaxies table
    filters, error_filters : list[str]
        The filters list and error filters list

    Methods
    -------
    add_random : Table
        Applies the random method, and returns the table with the updated values.
    """

    def __init__(self, galaxies:Table, filters:list[str], error_filters:list[str]) -> None:
        self.galaxies = galaxies.copy()
        self.filters = filters
        self.error_filters = error_filters

        self.length = len(self.galaxies)

    def _filter_random(self, galaxy_idx:int, filter:str, error_filter:str) -> None:
        """Apply gaussian noise to an individual filter"""

        value = self.galaxies[filter][galaxy_idx]
        error = self.galaxies[error_filter][galaxy_idx]

        new_value = np.random.normal(value, error)
        self.galaxies[filter][galaxy_idx] = new_value

    def _individual_galaxies_random(self) -> None:
        """Iterate through all of the single galaxies, and then each filter, adding in the random error"""

        # go through the indiviual galaxies
        for i in range(1, self.length):
            # then each filter
            for filt, err_filt in zip(self.filters, self.error_filters):
                self._filter_random(i, filt, err_filt)

    def _new_combo_values(self) -> None:
        """Update the combo galaxy values as the sum of the individuals."""

        for filt in self.filters:
            self.galaxies[filt][0] = np.sum(self.galaxies[filt][1:])

    def add_random(self) -> Table:
        self._individual_galaxies_random()
        self._new_combo_values()

        return self.galaxies


class Select():
    """
    Manages creation of selections from a catalog specified. 
    """

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
        self.catalog.add_column(0, 2, 'combo')

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


    def create_galaxies_table(self, z_vals:list[float], combinations:list[int], combo_fractional_errors:list[float]=[0.1], random_draw:bool=False, **kwargs) -> Table:
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
        combo_fractional_errors : list[float], default [0.1]
            Size of fractional error in the combination galaxies.
            From this, errors in the galaxies that make up the combo can be found.
            Multiple values can be input as a list.
        random_draw : bool, default False
            Controls if random draw is applied.

        kwargs are passed to the `Combine` object after `combo_fractional_error`. 

        Returns
        -------
        all_galaxies : Table
            Astropy table of all the galaxies.
            Also stored in attribute `self.all_galaxies`.
        """

        self.z_vals = z_vals
        self.combinations = combinations
        self.combo_fractional_errors = combo_fractional_errors

        list_galaxies:list[Table] = []

        last_id = self.catalog['id'][-1]
        # get number that can be added to all ids without interfering
        # will add 0 times for first frac error, then 1 times etc
        # then for combo values do the next of that
        # ids in catalog start from 1, so if last id is eg 100, can use 101 as the next
        extra_digit_number = int(10**np.ceil(np.log10(last_id)))
        self.combo_extra_digit = extra_digit_number * len(combo_fractional_errors) # minus one so next_combo_id first returns the n0000 id

        def next_combo_id() -> int:
            self.combo_extra_digit += 1
            return self.combo_extra_digit

        for z in z_vals:
            for combo in combinations:
                # get relevant number of the single galaxies at given z
                single_galaxies = self._select_galaxies(z, combo)
                # combine at each fractional error
                for i, frac_err in enumerate(combo_fractional_errors):
                    # copy galaxies table for each fractional error
                    frac_err_single_galaxies = single_galaxies.copy()
                    frac_err_single_galaxies.add_column(frac_err, 3, 'combo_frac_err')
                    # add relevant extra digit to ids
                    for id in frac_err_single_galaxies['id']:
                        id += i * extra_digit_number

                    # combine and add errors
                    combiner = Combine(frac_err_single_galaxies, next_combo_id(), self.filters, self.filter_errors,
                                        frac_err, **kwargs)
                    galaxies = combiner.full_combine()

                    if random_draw:
                        draw = RandomDraw(galaxies, self.filters, self.filter_errors)
                        galaxies = draw.add_random()

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
                ['frac_errors', *self.combo_fractional_errors],
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
        column_names = ['id', 'zred', 'combo', 'combo_frac_err', *self.filters, *self.filter_errors]
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

    
    def create_and_save_galaxies(self, z_vals:list[float], combinations:list[int], combo_fractional_errors:list[float]=[0.1], random_draw:bool=False,
                                 name:str='name', description:str='description', **kwargs) -> Table:
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
        combo_fractional_errors : list[float], default [0.1]
            Size of fractional error in the combination galaxies.
            From this, errors in the galaxies that make up the combo can be found.
            Multiple values can be input as a list.
        random_draw : bool, default False
            Controls if random draw is applied.
        name : str, Default 'name'
            Name to save under
        description : str, Default 'description'
            Description of selection

        kwargs are passed to the `Combine` object after `combo_fractional_error`. 

        Returns
        -------
        all_galaxies : Table
            Astropy table of all the galaxies.
            Also stored in attribute `self.all_galaxies`.
        """

        self.create_galaxies_table(z_vals, combinations, combo_fractional_errors, random_draw, **kwargs)
        self.save_galaxies_table(name, description)
        self.save_EAZY_file()
        return self.all_galaxies



def main() -> None:
    sel = Select('1_more_variation')
    print(sel.create_galaxies_table([2,3], [2,3], [0.1,1], random_draw=True))

if __name__ == '__main__':
    main()