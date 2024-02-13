import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sedpy import observate

filter_list = ["jwst_f070w", "jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f162m", "jwst_f182m", "jwst_f200w", "jwst_f210m", "jwst_f250m", "jwst_f277w", "jwst_moda_f277w", "jwst_modb_f277w", "jwst_f300m", "jwst_f335m", "jwst_moda_f335m", "jwst_modb_f335m", "jwst_f356w", "jwst_moda_f356w", "jwst_modb_f356w", "jwst_f410m", "jwst_moda_f410m", "jwst_modb_f410m", "jwst_f444w", "jwst_moda_f444w", "jwst_modb_f444w", "acs_wfc_f435w", "acs_wfc_f606w", "acs_wfc_f775w", "acs_wfc_f814w", "acs_wfc_f850lp", "wfc3_ir_f105w", "wfc3_ir_f125w", "wfc3_ir_f140w", "wfc3_ir_f160w"]

t = fits.open("/Users/sandrotacchella/ASTRO/jades/source_injection/injection_catalog_v2.fits")

filters = observate.load_filters(filter_list)

wl = np.array([filters[ii].wave_effective for ii in range(len(filters))])


ii_gal = 4000
plt.plot(wl, t['FLUX'].data[ii_gal][1:], 'o')
plt.axvline((1+t['PHYSICAL'].data['zred'][ii_gal])*1215)
# plt.yscale('log')
plt.show()

print(t['PHYSICAL'].data['zred'][ii_gal])
print(t['PHYSICAL'].data['igm_factor'][ii_gal])



t['SHAPE']


plt.scatter(t['PHYSICAL'].data['muv'], t['PHYSICAL'].data['beta'], c=t['PHYSICAL'].data['zred'], s=1)
plt.show()


