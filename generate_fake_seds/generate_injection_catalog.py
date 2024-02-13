#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_injection_catalog.py

"""

import sys
from argparse import ArgumentParser, Namespace
from itertools import product
import yaml

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import OrderedDict
from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.wcs import WCS
from astropy import cosmology
cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)

#import fsps
from sedpy import observate


from prospect.sources import CSPSpecBasis


def build_model(add_duste=False, complex_dust=False,
                add_neb=True, free_neb_met=False,
                free_fesc=False, free_igm=True,
                **kwargs):

    from prospect.models.templates import TemplateLibrary
    from prospect.models.transforms import dustratio_to_dust1
    from prospect.models import priors, sedmodel

    # --- Basic non-parametric SFH parameter set ---
    model_params = TemplateLibrary["parametric_sfh"]
    model_params["imf_type"]["init"] = 1
    model_params["mass"] = dict(N=1, isfree=False, init=1.0)

    # --- We *are* changing redshift ---
    model_params["zred"]["isfree"] = True
    model_params["zred"]["prior"] = priors.Uniform(mini=1, maxi=20)

    # --- Complexify Dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Slope of the attenuation curve, as delta from Calzetti
    model_params["dust_index"]  = dict(N=1, isfree=False, init=0.0,
                                       prior=priors.ClippedNormal(mini=-1, maxi=0.4, mean=0, sigma=0.5))
    # Young star dust, as a ratio to old star dust
    model_params["dust_ratio"]  = dict(N=1, isfree=False, init=0,
                                       prior=priors.ClippedNormal(mini=0, maxi=1.5, mean=1.0, sigma=0.3))
    model_params["dust1"]       = dict(N=1, isfree=False, init=0.0, depends_on=dustratio_to_dust1)
    model_params["dust1_index"] = dict(N=1, isfree=False, init=-1.0)
    model_params["dust_tesc"]   = dict(N=1, isfree=False, init=7.0)
    if complex_dust:
        model_params["dust_index"]["isfree"] = True
        model_params["dust_ratio"]["isfree"] = True

    # --- IGM attenuation and nebular emission ---
    model_params.update(TemplateLibrary["igm"])
    if free_igm:
        # Allow IGM transmission scaling to vary
        model_params["igm_factor"]['isfree'] = True
        model_params["igm_factor"]["prior"] = priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_neb_met:
            # Fit for independent gas metallicity
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
        # get rid of Lya
        model_params["nebemlineinspec"]["init"] = False
        model_params["elines_to_ignore"] = dict(init="Ly alpha 1216", isfree=False)
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if free_fesc:
        model_params["frac_obrun"] = dict(N=1, isfree=True, init=0,
                                          prior=priors.ClippedNormal(mini=0, maxi=1.0, mean=0.1, sigma=0.3))

    return sedmodel.SpecModel(model_params)


def get_sed(model, csp, source_dict, config):
    filters = observate.load_filters(config.filters)
    idx_f444w = np.where(np.array(config.filters) == 'jwst_f444w')[0][0]
    fuv = observate.load_filters(["galex_FUV"])

    obs = dict(filters=filters)
    wave = csp.wavelengths

    model.params["mass"] = np.array([1])

    beta, flux, muv = [], [], []

    for k in model.free_params:
        if k not in source_dict:
            print(f"free parameter {k} does not have a source distribution")

    for ii in range(config.number_source):

        theta = model.theta.copy()
        for k in model.free_params:
            if k in source_dict:
                theta[model.theta_index[k]] = source_dict[k][ii]
        theta[model.theta_index["logzsol"]] = -1.0
        theta[model.theta_index["gas_logu"]] = -2.0
        spec, phot, mstar = model.predict(theta, obs, sps=csp)
        bs = get_beta(wave, spec)
        renorm = source_dict['norm'][ii] / phot[idx_f444w]
        flux_norm = phot * renorm
        muv_mag = -2.5*np.log10(model.absolute_rest_maggies(fuv)[0]) - 2.5*np.log10(renorm)

        flux.append(flux_norm)
        beta.append(bs)
        muv.append(muv_mag)

    return flux, beta, muv

# define constants


def read_config(config_file):
    '''
    Read configuration file.
    '''
    with open(config_file, "r") as c:
        conf = yaml.load(c, Loader=yaml.FullLoader)
    config = Namespace(**conf)
    return config


def make_truncnorm(min=0, max=1, mu=0, sigma=1, **extras):
    a = (min - mu) / sigma
    b = (max - mu) / sigma
    return stats.truncnorm(a, b, loc=mu, scale=sigma)


def get_phot_njy(wave, fnu, zred, filt, mass=1.0):
    # define constants
    lsun = 3.846e33  # erg/s
    pc = 3.085677581467192e18  # in cm
    lightspeed = 2.998e18  # AA/s
    # value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
    to_cgs = lsun / (4.0 * np.pi * (pc*10)**2)
    # get distance
    lumdist = cosmo.luminosity_distance(zred).to('Mpc').value
    dfactor = (lumdist * 1e5)**2
    unit_conversion = to_cgs * (1 + zred)
    norm = mass * unit_conversion / dfactor
    obs_wave = wave * (1 + zred)
    flam = fnu * norm * lightspeed / obs_wave**2
    maggies = observate.getSED(wave * (1 + zred), flam, filterlist=filt, linear_flux=True)
    njy = maggies * 3631e9
    return njy


def lin_func(x, a, b):
    return a*x + b


def get_beta(wave, fnu):
    rest_uv = np.concatenate([np.arange(1268, 1284),  np.arange(1309, 1316), np.arange(1342, 1371), np.arange(1407, 1515), np.arange(1562, 1583), np.arange(1677, 1740), np.arange(1760, 1833), np.arange(1866, 1890), np.arange(1930, 1950), np.arange(2400, 2580)], dtype=float)
    flux_dusty = np.interp(rest_uv, wave, fnu)
    idx_good = np.isfinite(flux_dusty) & ~np.isnan(np.log10(flux_dusty))
    popt, pcov = curve_fit(lin_func, np.log10(rest_uv[idx_good]), np.log10(flux_dusty[idx_good]), p0=[0.0, 0.0])
    return(popt[0]-2.0)


def draw_sources(config):
    '''
    Draw galaxy parameter
    '''
    source_dict = {}

    # get filters, including F444W
    #filters = observate.load_filters(config.filters)
    #idx_f444w = np.where(np.array(config.filters) == 'jwst_f444w')[0][0]

    # draw flux normalization of F444W in nJy
    log_flux_f444w_list = np.random.uniform(low=config.log_flux_f444w["min"], high=config.log_flux_f444w["max"], size=config.number_source)
    source_dict['norm'] = np.power(10, log_flux_f444w_list)

    # draw redshift
    redshift_list = np.random.uniform(low=config.redshift["min"], high=config.redshift["max"], size=config.number_source)
    redshift_list = np.round(redshift_list, 1)
    source_dict['zred'] = redshift_list

    # draw dust2
    dust_fct = make_truncnorm(**config.dust2)
    source_dict['dust2'] = dust_fct.rvs(config.number_source)

    # draw SFH lookback time (i.e. age)
    log_sf_start_list = np.random.uniform(low=config.log_sf_start["min"], high=config.log_sf_start["max"], size=config.number_source)
    source_dict['tage'] = np.power(10, log_sf_start_list)

    # draw SFH tau
    log_tau_list = np.random.uniform(low=config.log_tau["min"], high=config.log_tau["max"], size=config.number_source)
    source_dict['tau'] = np.power(10, log_tau_list)

    # draw IGM factor
    figm_fct = make_truncnorm(**config.igm_factor)
    source_dict['igm_factor'] = figm_fct.rvs(config.number_source)

    # draw stellar metallicity
    logzsol_fct = make_truncnorm(**config.logzsol)
    source_dict['logzsol'] = logzsol_fct.rvs(config.number_source)

    print(source_dict.keys())
    print("tage" in source_dict)

    # draw fluxes
    csp = CSPSpecBasis(zcontinuous=1)
    model = build_model()
    print(f"model free parameters are {model.free_params}")
    flux, beta, muv = get_sed(model, csp, source_dict, config)

    source_dict['flux'] = np.array(flux)
    source_dict['muv'] = np.array(muv)
    source_dict['beta'] = np.array(beta)

    # draw size in pixels, convert to arcsec
    log_size_fct = make_truncnorm(**config.log_size)
    size_list = np.power(10, log_size_fct.rvs(config.number_source))
    source_dict['rhalf'] = size_list

    # draw Sersic index
    sersic_fct = make_truncnorm(**config.sersic)
    sersic_list = sersic_fct.rvs(config.number_source)
    source_dict['sersic'] = sersic_list

    # draw axis ratio
    q_fct = make_truncnorm(**config.q)
    q_list = q_fct.rvs(config.number_source)
    source_dict['q'] = q_list

    # draw PA in degree
    pa_list = np.random.uniform(low=0.0, high=180.0, size=config.number_source)
    source_dict['pa'] = pa_list

    return(source_dict)


def injection_data_model(bands=[], wcs=None, n_obj=1, col_dtype=np.float64):
    """
    Parameters
    ----------

    bands : sequence of str
        List of filter names

    wcs : astropy.wcs.WCS() instance
        The wcs giving the mapping between celestial coordinates and image
        pixel.  If supplied, it will be added ot the header of the 'POSITION'
        extension.

    n_obj : int
        Number of objects.  Each table will have this length

    Returns
    -------
    injection_data_model : astropy.fits.HDUList
        A list of HDUs, each of which is a binary FITS table, including
        * POSITION - ra, dec
        * SHAPE - profile and orientation
        * FLUX - total fluxes
        * PHYSICAL - physical parameters like redshift, stellar mass
    """

    meta = [("id", np.int64)]
    coldefs = OrderedDict()
    coldefs["POSITION"] = ["ra", "dec", "x_tile", "y_tile"]
    coldefs["SHAPE"] = ["sersic", "rhalf", "q", "pa"]
    coldefs["FLUX"] = bands
    coldefs["PHYSICAL"] = ["zred", "dust2", "tage", "tau", "igm_factor", "logzsol", "norm", "muv", "beta"]


    hdul = fits.HDUList([fits.PrimaryHDU()])
    for extn, cols in coldefs.items():
        cd = meta + [(c, col_dtype) for c in cols]
        arr = np.zeros(n_obj, dtype=np.dtype(cd))
        hdul.append(fits.BinTableHDU(arr, name=extn))

    ids = np.arange(1, n_obj+1)
    for hdu in hdul[1:]:
        hdu.data["id"] = ids

    hdul["FLUX"].header["FILTERS"] = ",".join(bands)
    hdul["FLUX"].header["BUNIT"] = "nJy"
    hdul["SHAPE"].header["PA_UNIT"] = "degrees"
    hdul["SHAPE"].header["RH_UNIT"] = "arcsec"

    if wcs is not None:
        hdul["POSITION"].header.update(wcs.to_header())
        hdul["POSITION"].verify('fix')

    return hdul


def rename_filters(filters):
    filter_names = []
    for ii_f in filters:
        if 'moda' in ii_f:
            filter_names.append(ii_f.split("_")[-1].upper() + "A")
        elif 'modb' in ii_f:
            filter_names.append(ii_f.split("_")[-1].upper() + "B")
        else:
            filter_names.append(ii_f.split("_")[-1].upper())
    return(filter_names)


def get_data_model_populated(config, source_dict):

    filter_names = rename_filters(config.filters)
    print(filter_names)

    dm = injection_data_model(bands=filter_names, n_obj=config.number_source)

    # update shape parameters
    dm["SHAPE"].data["sersic"] = source_dict['sersic']
    dm["SHAPE"].data["rhalf"] = source_dict['rhalf']
    dm["SHAPE"].data["q"] = source_dict['q']
    dm["SHAPE"].data["pa"] = source_dict['pa']

    # update physical model
    dm["PHYSICAL"].data["zred"] = source_dict['zred']
    dm["PHYSICAL"].data["dust2"] = source_dict['dust2']
    dm["PHYSICAL"].data["tage"] = source_dict['tage']
    dm["PHYSICAL"].data["tau"] = source_dict['tau']
    dm["PHYSICAL"].data["igm_factor"] = source_dict['igm_factor']
    dm["PHYSICAL"].data["logzsol"] = source_dict['logzsol']
    dm["PHYSICAL"].data["norm"] = source_dict['norm']
    dm["PHYSICAL"].data["muv"] = source_dict['muv']
    dm["PHYSICAL"].data["beta"] = source_dict['beta']

    # update fluxes
    for ii, ii_f in enumerate(filter_names):
        dm["FLUX"].data[ii_f] = source_dict['flux'][:, ii]

    return(dm)


if __name__ == "__main__":

    # setup argument parser
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default="config.yml")
    parser.add_argument("--output_dir", type=str,
                        default="")
    parser.add_argument("-v", "--verbose", action='store_true')

    # read args
    args = parser.parse_args()

    # read config file
    config = read_config(args.config_file)
    if args.verbose: print(f'Read config: {args.config_file}')

    # draw galaxies
    if args.verbose: print('Drawing galaxies...')
    source_dict = draw_sources(config)
    if args.verbose: print('Galaxies drawn.')

    # generate catalog
    if args.verbose: print('Generating catalog...')
    dm = get_data_model_populated(config, source_dict)
    if args.verbose: print('Catalog generated.')

    # write image
    if args.verbose: print('Writing image...')
    filename = args.output_dir + config.name + ".fits"
    dm.writeto(filename, overwrite=True)
    print(f"Successfully wrote image to {filename}")



