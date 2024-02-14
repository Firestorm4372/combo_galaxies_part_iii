import os
import shutil
import subprocess
import argparse

from astropy.io import fits
from astropy.table import Table

from typing import Self

class Generate():
    """
    Attributes
    ----------
    name : str
        Short name for the set.
        Default of `name`
    description : str
        Brief description of what this set contains, single line.
        Default of `description`
    config : str
        Filepath to config file to be used (YAML).
        Default of `generate_fake_seds/config.yml`
    data_path : str
        Filepath to directory to store data in.
        Default of `.data`
    """

    def __init__(self, name:str='name', description:str='description', config:str='generate_fake_seds/config.yml', data_path:str='.data') -> None:
        self.config = config
        self.name = name
        self.description = description
        self.data_path = data_path

        # make data folder if not exist
        os.makedirs(self.data_path, exist_ok=True)


    def _create_folder(self) -> None:
        # open file with current sets to get last entry
        # create sets.txt if not exist
        with open(f'{self.data_path}/sets.txt', 'a+') as f:
            f.seek(0)
            lines = f.readlines()
            try:
                # one more than last idx
                idx = int(lines[-2].split(' ')[0]) + 1
            except:
                idx = 0 # no previous

            self.folder_name = f'{idx}_{self.name}'
            self.folder_path = f'{self.data_path}/{self.folder_name}'
            # create folder
            os.makedirs(self.folder_path, exist_ok=True)
            # append name and description to the sets.txt
            f.writelines(f'{idx} {self.name}\n\t{self.description}\n')
        
        # add description to the folder
        with open(f'{self.folder_path}/description', 'w+') as f:
            f.writelines(self.description)

    def _run_generation(self)-> None:
        subprocess.run(['python', 'generate_fake_seds/generate_injection_catalog.py', '-v', '--config_file', self.config, '--output_dir', f'{self.folder_path}/'])

    def fake_galaxies(self) -> None:
        """Generate the fake galaxies using the code provided in `generate_fake_seds`
        
        Will then save to relevant folder, along with config.yml file
        """

        # create folder to store generation
        self._create_folder()
        # add config file to this folder, reset attribute to point to this
        new_config = f'{self.folder_path}/config.yml'
        shutil.copy2(self.config, new_config)
        self.config = new_config

        # generate the galaxies and save fits to this folder
        self._run_generation()
        self.fake_catalog_raw_path = f'{self.folder_path}/fake_catalog_raw.fits'

    
    def parse_catalog(self) -> None:
        with fits.open(self.fake_catalog_raw_path) as hdul:
            filter_data = Table(hdul[3].data)
            physical_data = Table(hdul[4].data)
        
        # add physical data columns for filter_data, but not the id again, and zred near start
        filter_data.add_column(physical_data['zred'], index=1)
        filter_data.add_columns(physical_data.columns[2:])

        self.catalog_path = f'{self.folder_path}/catalog.fits'
        filter_data.write(self.catalog_path, format='fits', overwrite=True)


    @classmethod
    def init_from_read(cls, folder_name:str, data_path:str='.data') -> Self:
        name = folder_name.split('_', maxsplit=1)[1]
        folder_path = f'{data_path}/{folder_name}'
        config = f'{folder_path}/config.yml'
        with open(f'{folder_path}/description') as f:
            description = f.readline()
        
        gen = cls(name, description, config, data_path)

        # add in folder_name and folder_path
        gen.folder_name = folder_name
        gen.folder_path = folder_path
        
        if os.path.isfile(f'{folder_path}/fake_catalog_raw.fits'):
            gen.fake_catalog_raw_path = f'{folder_path}/fake_catalog_raw.fits'
        
        if os.path.isfile(f'{folder_path}/catalog.fits'):
            gen.catalog_path = f'{folder_path}/catalog.fits'

        return gen


parser = argparse.ArgumentParser(
    description='Produce fake seds catalog and then parse it for use in selection etc'
)
parser.add_argument('name', nargs='?', default='name', help="Name of fake seds catalog to be produced. Default 'name'.")
parser.add_argument('description', nargs='?', default='description', help="Description of fake seds catalog. Default 'description'.")
parser.add_argument('-c', '--config', default='generate_fake_seds/config.yml', help="Config file to use for generation. Default 'generate_fake_seds/config.yml'.")
parser.add_argument('-d', '--data_path', default='.data', help="Path to store data in. Default '.data.")


def main() -> None:
    args = parser.parse_args()
    gen = Generate(args.name, args.description, args.config, args.data_path)
    print('Creating fake galaxies...')
    gen.fake_galaxies()
    print('Created fake galaxies.')
    print('Parsing catalog...')
    gen.parse_catalog()
    print(f'Catalog parsed at {gen.folder_path}/catalog.fits')

if __name__ == '__main__':
    main()