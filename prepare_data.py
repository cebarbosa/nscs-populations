import os
import copy

import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from specutils import Spectrum1D
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from spectres import spectres
from paintbox.utils import broad2res, disp2vel

import context

def get_muse_fwhm():
    """ Returns the FWHM of the MUSE spectrograph as a function of the
    wavelength. """
    wave, R = np.loadtxt(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "muse_wave_R.dat")).T
    wave = wave * u.nm
    fwhm = wave.to("angstrom") / R
    # First interpolation to obtain extrapolated values
    f1 = interp1d(wave.to("angstrom"), fwhm, kind="linear", bounds_error=False,
                 fill_value="extrapolate")
    # Second interpolation using spline
    wave = np.hstack((4000, wave.to("angstrom").value, 10000))
    f = interp1d(wave, f1(wave), kind="cubic", bounds_error=False)
    return f

def plot_muse_fwhm():
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, f(wave), "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Spectral resolution $\alpha$ FWHM (Angstrom)")
    plt.show()

def plot_vel_resolution():
    c = const.c
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, c.to("km/s") * f(wave) / wave, "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Resolution FWHM (km/s)")
    plt.show()

    def review_masks(target_sigma=300):
        wdir = os.path.join(context.home_dir, f"paintbox/dr1_sig{target_sigma}")
        filenames = [_ for _ in os.listdir(wdir) if
                     _.endswith(f"sig{target_sigma}.fits")]
        plt.figure(figsize=(20, 5))
        plt.ion()
        plt.show()
        for filename in filenames:
            galaxy = filename.split("_")[0]
            table = Table.read(os.path.join(wdir, filename))
            norm = fits.getval(os.path.join(wdir, filename), "NORM", ext=1)
            wave = table["wave"].data
            flux = table["flux"].data
            mask = table["mask"].data
            fluxerr = table["fluxerr"].data
            while True:
                plt.clf()
                flux_plot = flux
                flux_plot[mask == 1] = np.nan
                plt.plot(wave, flux_plot)
                plt.title(galaxy)
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                process = input("Update mask? (N/y): ")
                if process.lower() in ["", "n", "no"]:
                    break
                plt.waitforbuttonpress()
                pts = np.asarray(plt.ginput(2, timeout=-1))
                wmin = pts[:, 0].min()
                wmax = pts[:, 0].max()
                idx = np.where((wave >= wmin) & (wave <= wmax))
                mask[idx] = 1
            table = Table([wave, flux, fluxerr, mask],
                          names=["wave", "flux", "fluxerr", "mask"])
            hdu = fits.BinTableHDU(table)
            hdu.header["NORM"] = (norm, "Flux normalization")
            hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdulist.writeto(os.path.join(wdir, filename), overwrite=True)

def prepare_data(sigma=100, r=3, r1=5, r2=8, plot=False, redo=False):
    """ Match the resolution of observations with those of the CvD models."""
    names = ["NSC", "Annulus"]
    locations = [f"r_{r}", f"r1_{r1}_r2_{r2}"]
    wdir = os.path.join(context.home_dir, "data/nscs_sample")
    pb_dir = os.path.join(context.home_dir, "paintbox")
    outdir = os.path.join(pb_dir, f"sigma{sigma}")
    for path in [pb_dir, outdir]:
        if not os.path.exists(path):
            os.mkdir(path)

    galaxies = os.listdir(wdir)
    muse_fwhm = get_muse_fwhm()
    outfiles = []
    wave = disp2vel([4600, 9400], 100)
    muse_FWHM = muse_fwhm(wave)
    if plot:  # Comparing input and output resolutions
        outres = sigma / const.c.to("km/s").value * wave * 2.634
        FWHM_vel = lambda f: const.c.to("km/s") * f / wave
        plt.plot(wave, FWHM_vel(muse_FWHM))
        plt.plot(wave, FWHM_vel(outres))
        plt.show()
    for galaxy in galaxies:
        indir = os.path.join(wdir, galaxy)
        outname = f"{galaxy}_r_{r}_r1_{r1}_r2_{r2}_sig_{sigma}.fits"
        galdir = os.path.join(outdir, galaxy)
        if not os.path.exists(galdir):
            os.mkdir(galdir)
        output = os.path.join(galdir, outname)
        outfiles.append(output)
        if os.path.exists(output) and not redo:
            continue
        hdulist = [fits.PrimaryHDU()]
        for name, loc in zip(names, locations):
            flux_file = f"{name}_spec_from_cube_sum_{loc}.fits"
            var_file = f"{name}_variance_from_cube_sum_{loc}.fits"
            flux = fits.getdata(os.path.join(indir, flux_file))
            fluxvar= fits.getdata(os.path.join(indir, var_file))
            fluxerr = np.sqrt(fluxvar)
            spec1d =  Spectrum1D.read(os.path.join(indir, flux_file))
            wave = spec1d.spectral_axis.to(u.Angstrom).value
            inres = muse_fwhm(wave)
            outres = sigma / const.c.to("km/s").value * wave * 2.634
            flux_broad, fluxerr_broad = broad2res(wave, flux, inres, outres,
                                                  fluxerr=fluxerr)
            owave = disp2vel([wave[0], wave[-1]], int(sigma/3))
            flux_rebin, fluxerr_rebin = spectres(owave, wave, flux_broad,
                                       spec_errs=fluxerr_broad, fill=0,
                                       verbose=False)
            fluxnorm = np.median(flux_rebin)
            oflux = flux_rebin / fluxnorm
            ofluxerr = fluxerr_rebin / fluxnorm
            bscale = fluxnorm * spec1d.unit
            bscale = bscale.to(1e-17 * u.erg / (u.Angstrom *  u.cm * u.cm *
                                                u.s))
            omask = np.where(np.isfinite(oflux * ofluxerr), 0, 1)
            omask = review_mask(owave, oflux, omask)
            t = Table([owave, oflux, ofluxerr, omask],
                      names=["wave", "flux", "fluxerr", "mask"])
            hdu = fits.BinTableHDU(t)
            hdu.header["WAVEUNITS"] = (u.Angstrom.to_string(), \
                                       "Units for wavelength")
            hdu.header["FLUXUNITS"] = (bscale.unit.to_string(),
                                       "Units for flux and fluxerr")
            hdu.header["BSCALE"] = (bscale.value,
                                       "Normalization for flux and fluxerr.")
            hdu.header["EXTNAME"] = (name, "Name of the spectrum")
            hdulist.append(hdu)
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(output, overwrite=True)

def review_mask(wave, flux, mask):
    """ Remove regions from fit."""
    plt.figure(figsize=(20, 5))
    plt.ion()
    plt.show()
    while True:
        plt.clf()
        flux_plot = flux
        flux_plot[mask==1] = np.nan
        plt.plot(wave, flux_plot)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        process = input("Update mask? (N/y): ")
        if process.lower() in ["", "n", "no"]:
            break
        plt.waitforbuttonpress()
        pts = np.asarray(plt.ginput(2, timeout=-1))
        wmin = pts[:, 0].min()
        wmax = pts[:, 0].max()
        idx = np.where((wave >= wmin) & (wave <= wmax))
        mask[idx] = 1
    plt.close()
    return mask

if __name__ == "__main__":
    # plot_muse_fwhm()
    # plot_vel_resolution()
    prepare_data()
