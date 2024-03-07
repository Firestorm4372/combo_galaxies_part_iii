import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import eazy.hdf5


class Selection():
    """
    Internal class for opening up the galaxy selection (before was passed to EAZY)
    """

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
            self.frac_errors = [float(f) for f in lines[1][1:]]
            self.z_vals = [float(z) for z in lines[2][1:]]
            self.combinations = [int(c) for c in lines[3][1:]]
            self.filters = lines[4][1:]

        with fits.open(f'{self.selection_folder_path}/selection.fits') as hdul:
            self.all_galaxies = Table(hdul[1].data)

        self.eazy_photoz_path = f'{self.selection_folder_path}/eazy_out/photoz'


class SedPlot(Selection):
    """
    Faster init if just plotting SEDs. With method to plot a combo galaxy then its components.

    Can pass a `PhotoZ` object to just use that instead.
    """

    def __init__(self, folder_name: str, selection_name: str, data_path: str = '.data',
                 photoz:eazy.photoz.PhotoZ=None) -> None:
        super().__init__(folder_name, selection_name, data_path)

        if photoz == None:
            self.photoz = eazy.hdf5.Viewer(f'{self.eazy_photoz_path}.h5')
        else:
            self.photoz = photoz
            

    def combo_SEDs(self, idx:int, idx_is_id:bool=False, combined_only:bool=False, **kwargs) -> tuple[list[plt.Figure], int]:
        """
        Display figures of all SEDs in a combo, or just the combined if `combined_only = True`.

        Parameters
        ----------
        idx : int
            Number of the combination galaxy to display. Count from 0. 
        idx_is_id : bool, default False
            If True, searches for combo galaxy with id of `idx`.
        combined_only : bool, default False
            If True, will only display the combo SED.
        
        Any other kwargs are passed to `photoz.show_fit`

        Returns
        -------
        figs : list[Figure]
            The SED plots produced by `photoz.show_fit` for first the combo and then individual galaxies
        combo : int
            Combination number
        """
        
        if not idx_is_id:
            # count from first id of the combo galaxies
            combo_id = idx + self.all_galaxies['id'][0]
        else:
            combo_id = idx

        # get table idx of the combo galaxy
        table_idx = np.asarray(self.all_galaxies['id'] == combo_id).nonzero()[0][0] # this works

        combo = self.all_galaxies[table_idx]['combo']

        if not combined_only:
            # then take sub selection, and get ids from
            ids = self.all_galaxies['id'][table_idx:table_idx+combo+1]
        else:
            ids = [combo_id]

        # return figs objects only (first return of show_fit)
        return [self.photoz.show_fit(id, **kwargs)[0] for id in ids], combo
    

class ProcessData(Selection):
    """
    Internal class for performing the galaxy combination.
    """

    def __init__(self, folder_name: str, selection_name: str, data_path: str = '.data') -> None:
        super().__init__(folder_name, selection_name, data_path)

        # init from hdf5
        self.photoz = eazy.hdf5.initialize_from_hdf5(f'{self.eazy_photoz_path}.h5')

        # get relevant data
        self.zbest = self.photoz.zbest
        self.zgrid = self.photoz.zgrid
        self.chi2 = self.photoz.chi2_fit


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
        """Creates `combo_galaxies` dataframe"""

        self.combo_galaxies = pd.DataFrame({
            'z_red': self.z_red,
            'combo': self.combo,
            'combo_frac_err': self.combo_frac_err,
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
        self.combo_frac_err = []
        self.z_int = []
        self.z_mean = []
        self.z_chi2 = []
        self.chi2_int = []
        self.chi2_combo = []

        # go through each galaxy, see if a combo or not
        for i, galaxy in enumerate(self.all_galaxies):
            if galaxy['combo'] != 0:
                indiv_range = (i+1, i+1 + galaxy['combo'])

                # append zred and combo
                self.z_red.append(galaxy['zred'])
                self.combo.append(galaxy['combo'])
                self.combo_frac_err.append(galaxy['combo_frac_err'])
                self._data_single_combination(indiv_range)

        self._create_dataframe()


class Analyse(ProcessData):
    """
    Basic analysis tools on output from EAZY
    
    Parameters
    ----------
    folder_name : str
        Folder of the catalog
    selection_name : str
        Folder in which the particular selection is found
    data_path : str, default `.data`
        Path to the data folder
    catastrophic : float, default 0.1
            Fractional error in $Delta z / (1+z)$ past which fit is considered bad
    """

    def __init__(self, folder_name:str, selection_name:str, data_path:str='.data', catastrophic:float=0.1) -> None:
        super().__init__(folder_name, selection_name, data_path)
        self.catastrophic = catastrophic

    
    def _check_combined(self) -> None:
        """Makes sure the galaxies have been combined, and will combine if not.
        
        Used before plotting so `combine` does not have to be called.  
        """
        if (not hasattr(self, 'combo_galaxies')) or (len(self.combo_galaxies) == 0):
            self.combine()


    def rms_error_combo(self, fractional_errors:list[float]=None, z_split:list[float]=[], plot_no_catastrophic:bool=False, zero_y_lim:bool=False, show_title:bool=False) -> plt.Figure:
        """
        Plot of root mean square errors as compared to combo values.
        Can bin sections of the redshift range.

        Parameters
        ----------
        fractional_errors : list[float], default None
            Values of fractional error to plot.
            Default `None` plots all.
        z_split : list[float], default empty list
            Bins to separate the redshift range into.
            Default empty list has no binning (single line).

            Values in the list are intermediate bin values, between the min and max of the redshift range.
            Hence `[5]` will bin from [min, 5] and (5,max].
            And `[5,8]` will bin [min, 5] (5, 8] (8, max].
        plot_no_catastrophic : bool, default False
            If True, will plot means without catastrophic errors in each case also.
        zero_y_lim : bool, default False
            If True, will set lower y limit to be zero.
        show_title : bool, default False
            Controls if title is drawn on figure

        Returns
        -------
        fig : Figure
            Produced figure of mean in absolute of errors
        """

        self._check_combined()

        if fractional_errors == None:
            fractional_errors = self.frac_errors

        self.sub_plot_titles = {
            'z_int': 'Sum Before EAZY',
            'z_mean': r'Mean of $\{z_\mathrm{best}\}$',
            'z_chi2': r'Min of $\sum \chi^2$'
        }

        # z_split masks
        z_bin = [np.min(self.z_vals), *z_split, np.max(self.z_vals)]
        bin_masks = []
        # exclusive on the bottom, so just take a bit off
        z_bin_mod = [z_bin[0]-1, *z_bin[1:]]
        for lower, upper in zip(z_bin_mod[:-1], z_bin[1:]):
            mask = np.logical_and(np.greater(self.z_vals, lower), np.less_equal(self.z_vals, upper))
            bin_masks.append(mask)

        # create figure
        fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
        axs:list[plt.Axes]

        # plot for each method, fractional error, and z_bin
        for i, z_phot in enumerate(['z_int', 'z_mean', 'z_chi2']):
            for frac_err in fractional_errors:
                frac_err_mask = (self.combo_galaxies['combo_frac_err'] == frac_err)

                for k, bin_mask in enumerate(bin_masks):
                    means_all = []
                    means_no_cat = []

                    for combo in self.combinations:
                        combo_mask = (self.combo_galaxies['combo'] == combo)
                        err_and_combo_mask = np.logical_and(frac_err_mask, combo_mask)
                        sub = self.combo_galaxies[err_and_combo_mask][bin_mask]

                        all_frac_errors = (sub[z_phot] - sub['z_red']) / (1 + sub['z_red'])
                        no_catastrophic_mask = (all_frac_errors <= self.catastrophic)
                        
                        means_all.append(np.sqrt(np.mean(all_frac_errors**2)))
                        means_no_cat.append(np.sqrt(np.mean(all_frac_errors[no_catastrophic_mask]**2)))
                    
                    # simplify label if only fractional error
                    if (len(z_split)==0) and (not plot_no_catastrophic):
                        label = r'$\Delta = $' + f'{frac_err}'
                    else:
                        label = f'{frac_err} All {z_bin[k]} {z_bin[k+1]}'

                    axs[i].plot(self.combinations, means_all, label=label)
                    if plot_no_catastrophic:
                        axs[i].plot(self.combinations, means_no_cat, label=f'{frac_err} No cat {z_bin[k]} {z_bin[k+1]}')
            
            axs[i].set_xlim(0)
            axs[i].set_title(self.sub_plot_titles[z_phot])
            axs[i].tick_params(direction='in', right=True)
        
        if zero_y_lim:
            axs[0].set_ylim(0)

        axs[0].set_xticks(axs[0].get_xticks()[1:])
        axs[0].set_ylabel(r'RMS of $\Delta z / (1 + z_\mathrm{red})$')
        axs[1].set_xlabel('Number of Combined Galaxies')

        fig.subplots_adjust(wspace=0.05, right=0.88)

        # show legend only if more than one line
        if np.all([(len(ax.lines) > 1) for ax in axs]):
            fig.legend(*axs[0].get_legend_handles_labels(), loc='center right', bbox_to_anchor=(1,0.5), labelcolor='linecolor', handlelength=0, handletextpad=0, frameon=True)

        if show_title:
            fig.suptitle('RMS of Redshift Errors')
        else:
            fig.subplots_adjust(top=0.93)

        return fig
    
    def stdev_error_combo(self, fractional_errors:list[float]=None, z_split:list[float]=[], plot_no_catastrophic:bool=False, zero_y_lim:bool=False, show_title:bool=False) -> plt.Figure:
        """
        Plot of standard deviations of errors as compared to combo values.
        Can bin sections of the redshift range.

        Parameters
        ----------
        fractional_errors : list[float], default None
            Values of fractional error to plot.
            Default `None` plots all.
        z_split : list[float], default empty list
            Bins to separate the redshift range into.
            Default empty list has no binning (single line).

            Values in the list are intermediate bin values, between the min and max of the redshift range.
            Hence `[5]` will bin from [min, 5] and (5,max].
            And `[5,8]` will bin [min, 5] (5, 8] (8, max].
        plot_no_catastrophic : bool, default False
            If True, will plot means without catastrophic errors in each case also.
        zero_y_lim : bool, default False
            If True, will set lower y limit to be zero.
        show_title : bool, default False
            Controls if title is drawn on figure

        Returns
        -------
        fig : Figure
            Produced figure of mean in absolute of errors
        """

        self._check_combined()

        if fractional_errors == None:
            fractional_errors = self.frac_errors

        self.sub_plot_titles = {
            'z_int': 'Sum Before EAZY',
            'z_mean': r'Mean of $\{z_\mathrm{best}\}$',
            'z_chi2': r'Min of $\sum \chi^2$'
        }

        # z_split masks
        z_bin = [np.min(self.z_vals), *z_split, np.max(self.z_vals)]
        bin_masks = []
        # exclusive on the bottom, so just take a bit off
        z_bin_mod = [z_bin[0]-1, *z_bin[1:]]
        for lower, upper in zip(z_bin_mod[:-1], z_bin[1:]):
            mask = np.logical_and(np.greater(self.z_vals, lower), np.less_equal(self.z_vals, upper))
            bin_masks.append(mask)

        # create figure
        fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
        axs:list[plt.Axes]

        # plot for each method, fractional error, and z_bin
        for i, z_phot in enumerate(['z_int', 'z_mean', 'z_chi2']):
            for frac_err in fractional_errors:
                frac_err_mask = (self.combo_galaxies['combo_frac_err'] == frac_err)

                for k, bin_mask in enumerate(bin_masks):
                    means_all = []
                    means_no_cat = []

                    for combo in self.combinations:
                        combo_mask = (self.combo_galaxies['combo'] == combo)
                        err_and_combo_mask = np.logical_and(frac_err_mask, combo_mask)
                        sub = self.combo_galaxies[err_and_combo_mask][bin_mask]

                        all_frac_errors = (sub[z_phot] - sub['z_red']) / (1 + sub['z_red'])
                        no_catastrophic_mask = (all_frac_errors <= self.catastrophic)
                        
                        means_all.append(np.std(all_frac_errors))
                        means_no_cat.append(np.std(all_frac_errors[no_catastrophic_mask]))
                    
                    # simplify label if only fractional error
                    if (len(z_split)==0) and (not plot_no_catastrophic):
                        label = r'$\Delta = $' + f'{frac_err}'
                    else:
                        label = f'{frac_err} All {z_bin[k]} {z_bin[k+1]}'

                    axs[i].plot(self.combinations, means_all, label=label)
                    if plot_no_catastrophic:
                        axs[i].plot(self.combinations, means_no_cat, label=f'{frac_err} No cat {z_bin[k]} {z_bin[k+1]}')
            
            axs[i].set_xlim(0)
            axs[i].set_title(self.sub_plot_titles[z_phot])
            axs[i].tick_params(direction='in', right=True)
        
        if zero_y_lim:
            axs[0].set_ylim(0)

        axs[0].set_xticks(axs[0].get_xticks()[1:])
        axs[0].set_ylabel(r'$\sigma$ of $\Delta z / (1 + z_\mathrm{red})$')
        axs[1].set_xlabel('Number of Combined Galaxies')

        fig.subplots_adjust(wspace=0.05, right=0.88)

        # show legend only if more than one line
        if np.all([(len(ax.lines) > 1) for ax in axs]):
            fig.legend(*axs[0].get_legend_handles_labels(), loc='center right', bbox_to_anchor=(1,0.5), labelcolor='linecolor', handlelength=0, handletextpad=0, frameon=True)

        if show_title:
            fig.suptitle('Std dev of Redshift Errors')
        else:
            fig.subplots_adjust(top=0.93)

        return fig
    
    def hist_errors_combo(self, combinations:list[int]=None, fractional_errors:list[int]=None, bins:int|list|str=10, show_zero_line:bool=True) -> list[plt.Figure]:
        """
        Histogram of errors in redshift using the different methods.
        Currently produces horizontal figure.
        
        Parameters
        ----------
        combinations : list[int], default None
            If argument supplied, will only show and return those figures.
            If `None` all combinations are shown.
        fractional_errors : list[int], default None
            If argument supplied, will only show and return those figures.
            If `None` all fractional errors are shown.
        bins : int | list | str, default 10
            Argument passed to `plt.hist` as bins (see `hist` docs)
        show_zero_line : bool, default True
            Controls whether horizontal zero line is plotted

        Returns
        -------
        figs : list[Figure]
            Returns in list even if only one figure produced 
        """

        self._check_combined()

        # combinations list is either all or the selection
        if combinations == None:
            combinations = self.combinations
        
        if fractional_errors == None:
            fractional_errors = self.frac_errors

        figs = []

        for frac_err in fractional_errors:
            frac_err_mask = (self.combo_galaxies['combo_frac_err'] == frac_err)
            for combo in combinations:
                combo_mask = (self.combo_galaxies['combo'] == combo)
                mask = np.logical_and(frac_err_mask, combo_mask)
                sub = self.combo_galaxies[mask]

                # create figure
                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)

                for i, z_phot in enumerate(['z_int', 'z_mean', 'z_chi2']):
                    frac_error = (sub[z_phot] - sub['z_red']) / (1 + sub['z_red'])

                    # make histogram
                    axs[i].hist(frac_error, bins, orientation='horizontal')

                    if show_zero_line:
                        axs[i].axhline(0, color='black')

                    if i != 0:
                        axs[i].tick_params(left=False)

                    axs[i].set_title(z_phot)
                
                fig.supylabel(r'$\Delta z / (1 + z_\mathrm{red})$')
                fig.supxlabel('Density')

                fig.subplots_adjust(wspace=0)
                fig.suptitle(f'frac_err = {frac_err}, combo = {combo}')
                
                figs.append(fig)

        return figs

    def zred_zphot(self, combinations:list[int]=None, fractional_errors:list[int]=None, errors_lim:list[float]=None) -> list[plt.Figure]:
        """
        Produce figures showing plots of z_red compared to various z_phot methods.
        These are integrate before EAZY, mean of EAZY z_best, and min of combined chi squared.

        Parameters
        ----------
        combinations : list[int], default None
            If argument supplied, will only show and return those figures.
            If `None` all combinations are shown.
        fractional_errors : list[int], default None
            If argument supplied, will only show and return those figures.
            If `None` all fractional_errors are shown.
        errors_lim : list[float], default None
            y limits for the errors plots.
            If `None` will be plus minus `2 * self.catastrophic`

        Returns
        -------
        figs : list[Figure]
            Figure for each combination value
        """

        self._check_combined()

        if combinations == None:
            combinations = self.combinations

        if fractional_errors == None:
            fractional_errors = self.frac_errors

        if errors_lim == None:
            errors_lim = [-2*self.catastrophic, 2*self.catastrophic]

        figs = []
        
        for frac_err in fractional_errors:
            frac_err_mask = (self.combo_galaxies['combo_frac_err'] == frac_err)
            for combo in combinations:
                combo_mask = (self.combo_galaxies['combo'] == combo)
                mask = np.logical_and(frac_err_mask, combo_mask)
                sub = self.combo_galaxies[mask]

                fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', height_ratios=[4,1])

                # then for each of the different z_phot methods
                for i, z_phot in enumerate(['z_int', 'z_mean', 'z_chi2']):
                    frac_error = (sub[z_phot] - sub['z_red']) / (1 + sub['z_red'])

                    # mask for catastrophic outliers
                    mask_bad = (np.abs(frac_error) > self.catastrophic)
                    mask_good = (np.abs(frac_error) <= self.catastrophic)

                    # fractional errors plot
                    axs[1,i].axhline(0, color='black', linewidth=0.6)
                    axs[1,i].plot(sub['z_red'][mask_good], frac_error[mask_good], 'c.')
                    axs[1,i].plot(sub['z_red'][mask_bad], frac_error[mask_bad], 'r.')
                    axs[1,i].set_ylim([-0.2, 0.2])

                    # linear plot
                    axs[0,i].plot(sub['z_red'], sub['z_red'], 'k')
                    axs[0,i].plot(sub['z_red'][mask_good], sub[z_phot][mask_good], 'c.')
                    axs[0,i].plot(sub['z_red'][mask_bad], sub[z_phot][mask_bad], 'r.')
                    axs[0,i].set_xticks(axs[0,i].get_yticks())
                    axs[0,i].set_xlim(axs[0,i].get_ylim())
                    axs[0,i].set_title(z_phot)

                    axs[1,i].set_xlabel(r'$z_{\mathrm{true}}$')

                    axs[0,i].tick_params(bottom=False)
                    if i != 0:
                        axs[0,i].tick_params(left=False)
                        axs[1,i].tick_params(left=False)

                axs[0,0].set_ylabel(r'$z_{\mathrm{phot}}$')
                axs[1,0].set_ylabel(r'$\Delta z / (1 + z_{\mathrm{true}})$')

                fig.subplots_adjust(wspace=0, hspace=0)
                fig.set_figheight(0.8 * fig.get_figheight())
                fig.set_figwidth(12/5 * fig.get_figheight())
                fig.suptitle(f'frac_err = {frac_err}, combo = {combo}')

                figs.append(fig)
            
        return figs
    
    
    def combo_SEDs(self, idx:int, idx_is_id:bool=False, combined_only:bool=False, **kwargs) -> tuple[list[plt.Figure], int]:
        """
        Wrapper around `SedPlot.combo_SEDs` method.

        Display figures of all SEDs in a combo, or just the combined if `combined_only = True`.

        Parameters
        ----------
        idx : int
            Number of the combination galaxy to display. Count from 0. 
        idx_is_id : bool, default False
            If True, searches for combo galaxy with id of `idx`.
        combined_only : bool, default False
            If True, will only display the combo SED.
        
        Any other kwargs are passed to `photoz.show_fit`

        Returns
        -------
        figs : list[Figure]
            The SED plots produced by `photoz.show_fit` for first the combo and then individual galaxies
        combo : int
            Combination number
        """

        if not hasattr(self, 'sedplot'):
            self.sedplot = SedPlot(self.folder_name, self.selection_name, self.data_path, self.photoz)

        return self.sedplot.combo_SEDs(idx, idx_is_id, combined_only, **kwargs)
    
    