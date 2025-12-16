#%%

#The program requires the "vip_hci" library
#pip install vip_hci

import numpy as np
from vip_hci.fits import open_fits
from vip_hci.fm import normalize_psf, cube_inject_companions, cube_planet_free
from vip_hci.var import frame_center
from vip_hci.config import VLT_NACO
from vip_hci.psfsub import pca_annular, pca
from vip_hci.metrics import snr
import matplotlib.pyplot as plt

pref = "datasets/naco_betapic_"                                         #
cube = open_fits("{}cube_cen".format(pref))                             #open the raw cube file
psf = open_fits("{}psf".format(pref))                                   #open the psf file
angs = open_fits("{}derot_angles".format(pref))                         #open the parallactic angles file

psfn, flux, fwhm = normalize_psf(psf, full_output=True)                 #normalize the psf for easier experiments
cy, cx = frame_center(cube)                                             #obtain the coordinates of the frame center
plsc = VLT_NACO['plsc']                                                 #obtain the pixel scale, it depends on the instrument used

#%%

ncomps = range(1,50,2)
    
#applies pca for each ncomp value and stores the snr at the location of the planet
#in the respective array
snrs_full = []
snrs_annu = []
for ncomp in ncomps:
    test = pca(cube,angs,ncomp=ncomp,verbose=False)
    snrs_full.append(snr(test, source_xy=(58.5,35.5), fwhm=fwhm, verbose=False))
    test = pca_annular(cube,angs,ncomp=ncomp,verbose=False)
    snrs_annu.append(snr(test, source_xy=(58.5,35.5), fwhm=fwhm, verbose=False))

#plots results
fig, ax = plt.subplots(figsize=(10,5))
ax.set_title('PCA type vs. SNR')
ax.plot(ncomps,snrs_full, 'o-', alpha=0.5, label='full-frame')
ax.plot(ncomps,snrs_annu, 'o-', alpha=0.5, label='annular')
ax.set_xlabel('Principal components'); ax.set_ylabel('S/N ratio')
ax.legend()
plt.savefig('images/snrs.pdf', bbox_inches='tight', dpi=500)

#%%

#these are the "real" coordinates and flux of the planet
#used to delete it from the cube
r_b =  0.452/plsc # Absil et al. (2013)
theta_b = 211.2+90 # Absil et al. (2013)
f_b = 648.2

#create a new cube without real planets
cube_inj = cube_planet_free([(r_b, theta_b, f_b)], cube, angs, psfn=psfn)
cube_inj = cube_inject_companions(cube_inj, psfn, angs, flevel=400,
                                                        rad_dists=15,
                                                        theta=135) #near-bright
cube_inj = cube_inject_companions(cube_inj, psfn, angs, flevel=200,
                                                        rad_dists=45,
                                                        theta=225) #far-dim
cube_inj = cube_inject_companions(cube_inj, psfn, angs, flevel=400,
                                                        rad_dists=45,
                                                        theta=305) #far-bright

#define the cartesian coordinates of the sources for snr extraction
far_bright = (cx + 45 * np.cos(np.deg2rad(305)), cy + 45 * np.sin(np.deg2rad(305)))
far_dim = (cx + 45 * np.cos(np.deg2rad(225)), cy + 45 * np.sin(np.deg2rad(225)))
near_bright = (cx + 15 * np.cos(np.deg2rad(135)), cy + 15 * np.sin(np.deg2rad(135)))

#create an array for each of the possible cases
ncomps = [10,20,30,40,50,60]
ff_snrs_fb = []; ff_snrs_fd = []; ff_snrs_nb = []
an_snrs_fb = []; an_snrs_fd = []; an_snrs_nb = []

#applies both pca types with each ncomp value, adds the snr value to the respective
#arrays and plots the resulting image for visual appreciation of the planets
for ncomp in ncomps:
    res_full = pca(cube_inj, angs, ncomp=ncomp, verbose=False)
    res_annu = pca_annular(cube_inj, angs, ncomp=ncomp, radius_int=10 ,verbose=False)
    
    ff_snrs_fb.append(snr(res_full, source_xy=far_bright, fwhm=fwhm, verbose=False))
    ff_snrs_fd.append(snr(res_full, source_xy=far_dim, fwhm=fwhm, verbose=False))
    ff_snrs_nb.append(snr(res_full, source_xy=near_bright, fwhm=fwhm, verbose=False))
    
    an_snrs_fb.append(snr(res_annu, source_xy=far_bright, fwhm=fwhm, verbose=False))
    an_snrs_fd.append(snr(res_annu, source_xy=far_dim, fwhm=fwhm, verbose=False))
    an_snrs_nb.append(snr(res_annu, source_xy=near_bright, fwhm=fwhm, verbose=False))
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10), layout="constrained")
    im1 = ax1.imshow(res_full, cmap='viridis')
    im2 = ax2.imshow(res_annu, cmap='viridis')
    ax1.invert_yaxis(), ax2.invert_yaxis()
    fig.colorbar(im1, ax=ax1), fig.colorbar(im2,ax=ax2)
    ax1.set_title('PCA full-frame',fontsize=15); ax2.set_title('PCA annular', fontsize=15)
    fig.suptitle('ncomp: '+str(ncomp), fontsize=20)
    plt.savefig('images/fullvsannu_'+str(ncomp)+'.pdf', bbox_inches='tight', dpi=500)

#takes all of the snr values and plots them in three axes, one for each source
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(5,8), layout='constrained')
ax1.set_title('far-bright')  ; ax2.set_title('far-dim'); ax3.set_title('near-bright')
ax1.plot(ncomps, ff_snrs_fb, 'o-', label='full-frame') ; ax1.plot(ncomps, an_snrs_fb, 'o-', label='annular')
ax2.plot(ncomps, ff_snrs_fd, 'o-', label='full-frame') ; ax2.plot(ncomps, an_snrs_fd, 'o-', label='annular')
ax3.plot(ncomps, ff_snrs_nb, 'o-', label='full-frame') ; ax3.plot(ncomps, an_snrs_nb, 'o-', label='annular')
ax1.legend(); ax2.legend(); ax3.legend()
ax1.set_xlabel('ncomp'), ax2.set_xlabel('ncomp'), ax3.set_xlabel('ncomp')
ax1.set_ylabel('SNR'), ax2.set_ylabel('SNR'), ax3.set_ylabel('SNR')
plt.savefig('images/experiments.pdf', bbox_inches='tight', dpi =500)
