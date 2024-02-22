import os
import csv

from astropy.io import fits
from astropy.table import Table
import eazy
import eazy.hdf5

class RunEAZY():
    def __init__(self, folder_name:str, selection_name:str, data_path:str='.data') -> None:
        self.folder_name = folder_name
        self.selection_name = selection_name
        self.data_path = data_path

        # path to the overall catalog, and individual selection
        self.folder_path = f'{data_path}/{folder_name}'
        self.selection_folder_path = f'{self.folder_path}/{selection_name}'
        self.eazy_out_folder_path = f'{self.selection_folder_path}/eazy_out'

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
            
    
    def init_photoz(self, add_params:dict=dict(), param_file:str=None, translate_file='eazy_files/z_phot.translate') -> None:
        # don't use these 'default' params if params file given
        if param_file == None:
            params = {
                'CATALOG_FILE': f'{self.selection_folder_path}/EAZY_input.csv',
                'CATALOG_FORMAT': 'csv',
                'FILTERS_RES': 'eazy_files/FILTER.RES.latest',
                'TEMPLATES_FILE': 'templates/JADES/JADES_fsps_local.param',
                'APPLY_PRIOR': 'n',
                'Z_MIN': 1.,
                'Z_MAX': 15.,
                'Z_STEP': 0.01,
                'OUTPUT_DIRECTORY': self.eazy_out_folder_path,
                'MAIN_OUTPUT_FILE': f'{self.eazy_out_folder_path}/photoz'
            } | add_params
        else:
            params = add_params
        
        # create photoz object
        self.photoz = eazy.photoz.PhotoZ(param_file=param_file, params=params, translate_file=translate_file)

    def _standard_output(self) -> None:
        """Runs `photoz.standardoutput`, but puts outputs in attributes"""
        self.zout, self.hdu = self.photoz.standard_output()
    

    def EAZY_run(self, add_params:dict=dict(), param_file:str=None, translate_file='eazy_files/z_phot.translate',
                 standard_output:bool=True, hdf5_file:bool=True):
        
        self.init_photoz(add_params, param_file, translate_file)
        self.photoz.fit_catalog()

        if standard_output:
            os.makedirs(self.eazy_out_folder_path, exist_ok=True)
            self._standard_output()
        
        if hdf5_file:
            eazy.hdf5.write_hdf5(self.photoz, f'{self.eazy_out_folder_path}/photoz.h5')

    
    @classmethod
    def init_from_hdf5(cls, folder_name:str, selection_name:str, data_path:str='.data') -> None:
        run = cls(folder_name, selection_name, data_path)
        run.photoz = eazy.hdf5.initialize_from_hdf5(f'{run.eazy_out_folder_path}/photoz.h5')



def main() -> None:
    pass

if __name__ == '__main__':
    main()