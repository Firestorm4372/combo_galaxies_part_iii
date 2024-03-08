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
        
        # get the dataframe from file
        self.df_combo = pd.read_csv(self.combo_galaxies_file_path)
        # massage into better
        self.df = self._massage(self.df_combo)

        abserr = np.abs(self.df['err'])
        self.df.insert(len(self.df.columns), 'abs_err', abserr)

        # extract different values
        self.combinations = np.unique(self.df['combo'])
        self.frac_errs = np.unique(self.df['frac_err'])
        self.methods = ['int', 'mean', 'chi2']

        # set figure save path
        if figure_save_path == None:
            self.figure_save_path = f'{self.folder_path}/figures'
            os.makedirs(self.figure_save_path, exist_ok=True)
        else:
            self.figure_save_path = figure_save_path
    
    @staticmethod
    def _massage(df_combo:pd.DataFrame) -> pd.DataFrame:
        """
        Massage or melt the dataframe for better use with Seaborn.
        Make method a column, with `z` and `err` values for each.
        """

        df1 = df_combo.melt(id_vars=['idx', 'z_red', 'combo', 'frac_err'], value_vars=['z_int', 'z_mean', 'z_chi2'], var_name='method', value_name='z')
        df2 = df_combo.melt(value_vars=['err_int', 'err_mean', 'err_chi2'], var_name='err_method', value_name='err')

        df3 = pd.concat([df1, df2], axis=1)

        df3.drop(columns='err_method', inplace=True)
        df3['method'] = [m.split('_')[-1] for m in df3['method']]

        df3.sort_values('idx', inplace=True)
        df3.reset_index(drop=True, inplace=True)
        
        return df3
    

    def plot_abs_err_mean(self, errorbar=('pi', 75), show_title:bool=False) -> sns.FacetGrid:
        """
        Plot of the `abs_err`.
        Will markers as the mean values, with an errorband as specified in argument.

        Parameters
        ----------
        errorbar, default `('pi', 75)`
            Argument passed to `errorbar` of the `relplot`.
            Default gives percentile interval of 75%.
        show_title : bool, default False
            Controls if a suptitle is shown.
        """

        rp = sns.relplot(
            kind='line',
            x='combo', y='abs_err', col='method',
            hue='frac_err', style='frac_err', markers=True, dashes=False,
            errorbar=errorbar,
            palette='flare',
            data=self.df
        )

        rp.set(ylim=0, xlim=(1, self.combinations[-1]))
        rp.legend.set_title(r'Frac Err')
        rp.set(xlabel='Number of Combined Galaxies', ylabel='Absolute Error')
        for ax, title in zip(rp.axes.flatten(), ['Pre-Integrate', 'Mean of Best', r'Min of $\sum \chi^2$']):
            ax.set_title(title)

        if show_title:
            rp.figure.suptitle('Plot of Absolute Errors of the different methods.')

        return rp


    def _create_df_quant(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Create `df_quant` and return.
        """
        data = []

        for i, m in enumerate(self.methods):
            for f in self.frac_errs:
                for c in self.combinations:
                    sub = df.query('(combo == @c) and (frac_err == @f) and (method == @m)')
                    errors = np.asarray(sub['err'])

                    mean = np.mean(errors)
                    abs_mean = np.abs(mean)
                    rms = np.sqrt(np.mean(errors**2))
                    std = np.std(errors)

                    data.append({
                        'method': m, 'frac_err': f, 'combo': c,
                        'mean': mean, 'abs_mean': abs_mean, 'rms': rms, 'std': std
                    })

        df_quant = pd.DataFrame.from_dict(data)
        df_quant = df_quant.melt(id_vars=['method', 'frac_err', 'combo'], value_vars=['mean', 'abs_mean', 'rms', 'std'], var_name='quantity')
        return df_quant

    def _check_df_quant(self) -> None:
        """Check for `df_quant` attribute, and populate if not present."""
        if not hasattr(self, 'df_quant'):
            self.df_quant = self._create_df_quant(self.df)
    
    def plot_all_quantities(self, quantities:list=None, show_title:bool=False) -> sns.FacetGrid:
        
        self._check_df_quant()

        if quantities == None:
            df = self.df_quant
        else:
            # only take the given subset
            df = self.df_quant.query('quantity in @quantities')

        rp = sns.relplot(
            kind='line',
            x='combo', y='value',
            col='method', row='quantity',
            hue='frac_err', style='frac_err', markers=True, dashes=False,
            palette='flare',
            data=df
        )

        return rp
    

    def plot_abs_err_cumulative(self, frac_err:float, combos:list[int]=[1,2], col_wrap:int=2, upper_xlim:float=1) -> sns.FacetGrid:
        """
        Produces 'Empirical Cumulative Distribution Function' (ecdf) plot.
        This shows the proportion with absolute errors (delta z / 1+z) below each value on the x-axis.

        Parameters
        ----------
        frac_err : float
            The fractional error value to view for.
        combos : list[int], default [1,2]
            Combo values to plot for.
        col_wrap : int, default 2
            Number of plots to show per row.
        upper_xlim : float, default 1
            Right x limit for the plot.
            If `None` will let plotter decide.
        """

        df = self.df.query(f'(frac_err==@frac_err) and (combo in @combos)')

        ecdf = sns.displot(
            kind='ecdf',
            x='abs_err',
            hue='method',
            col='combo',
            col_wrap=col_wrap,
            data=df
        )

        if upper_xlim == None:
            ecdf.set(xlim=0)
        else:
            ecdf.set(xlim=(0,upper_xlim))

        return ecdf

