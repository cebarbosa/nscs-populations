""" Run paintbox for INSPIRE test cases. """
import os
import glob
import shutil
import platform
import copy

import astropy.units as u
from astropy.io import fits
from astropy.table import Table, vstack, hstack
import numpy as np
from scipy import stats
from specutils import Spectrum1D
import paintbox as pb
from paintbox.utils import CvD18, disp2vel
import emcee
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing as mp

import context

def model_nscs_plus_annulus(wave, dlam=100, sigma=100, wmin=4600, wmax=9400):
    """ Produces a model for the annulus and another for the NSCS. """
    velscale = int(sigma * 2.634 / 3)
    twave = disp2vel(np.array([wmin, wmax], dtype="object"), velscale)
    store = os.path.join(context.home_dir, "templates",
                         f"CvD18_w{wmin}-{wmax}_sig{sigma}.fits")
    ssp = CvD18(twave, sigma=sigma, store=store, libpath=context.cvd_dir)
    limits = ssp.limits
    porder = int((wave[-1] - wave[0]) / dlam)
    # Building model for annulus
    ssp_ann = copy.deepcopy(ssp)
    ssp_ann.parnames = [f"{p}_ann" for p in ssp_ann.parnames]
    stars_ann = pb.Resample(wave, pb.LOSVDConv(ssp_ann, losvdpars=[
        "Vsyst_ann", "sigma_ann"]))
    poly_ann = pb.Polynomial(wave, porder)
    poly_ann.parnames = [f"{p}_ann" for p in poly_ann.parnames]
    pop_ann = stars_ann * poly_ann
    # Model for NSC and composite model for center
    ssp_nsc = copy.deepcopy(ssp)
    ssp_nsc.parnames = [f"{p}_nsc" for p in ssp_nsc.parnames]
    w_ann = pb.Polynomial(twave, 0)
    w_ann.parnames = ["weight_ann"]
    w_nsc = pb.Polynomial(twave, 0)
    w_nsc.parnames = ["weight_nsc"]
    ssp_cen = w_ann * ssp_ann + w_nsc * ssp_nsc
    stars_cen = pb.Resample(wave,
                pb.LOSVDConv(ssp_cen, losvdpars=["Vsyst_cen", "sigma_cen"] ))
    poly_cen = pb.Polynomial(wave, porder)
    poly_cen.parnames = [f"{p}_cen" for p in poly_cen.parnames]
    pop_cen = stars_cen * poly_cen
    return pop_ann, pop_cen, limits

def set_priors(parnames, limits, vsyst=0):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname_full in parnames:
        loc = parname_full.split("_")[-1]
        parname = "_".join(parname_full.split("_")[:-1])
        name = parname.split("_")[0]
        if name in limits:
            vmin, vmax = limits[name]
            delta = vmax - vmin
            priors[parname_full] = stats.uniform(loc=vmin, scale=delta)
        elif parname == "Vsyst":
            priors[parname_full] = stats.norm(loc=vsyst, scale=500)
        elif parname_full == "eta":
            priors["eta"] = stats.uniform(loc=0.01, scale=9.99)
        elif parname_full == "nu":
            priors["nu"] = stats.uniform(loc=2, scale=20)
        elif parname == "sigma":
            priors[parname_full] = stats.uniform(loc=50, scale=300)
        elif name == "weight":
            priors[parname_full] = stats.uniform(loc=0, scale=1)
        elif name == "p":
            porder = int(parname.split("_")[1])
            nssps = 1 if loc == "ann" else 2
            if porder == 0:
                mu, sd = np.sqrt(2 * nssps), 1
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname_full] = stats.truncnorm(a, b, mu, sd)
            else:
                priors[parname_full] = stats.norm(0, 0.05)
        else:
            raise ValueError(f"Parameter without prior: {parname_full}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_likelihood(logp, outdb):
    nsteps = int(outdb.split("_")[2].split(".")[0])
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(0.9 * nsteps), thin=500,
                                 flat=True)
    n = len(tracedata)
    llf = np.zeros(n)
    for i in tqdm(range(n)):
        llf[i] = logp(tracedata[i])
    plt.plot(llf)
    plt.show()

def plot_fitting(wave, flux, fluxerr, sed, trace, db, redo=True, sky=None,
                 print_pars=None, mask=None, lw=1, galaxy=None):
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    galaxy = "Observed" if galaxy is None else galaxy
    mask = np.full_like(wave, True) if mask is None else mask
    pmask = mask.astype(np.bool)
    fig_width = 3.54  # inches - A&A template for 1 column
    print_pars = sed.parnames if print_pars is None else print_pars
    ssp_model = "CvD"
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]",
              "Na": "[Na/Fe]" if ssp_model == "emiles" else "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]",
              "as/Fe": "[as/Fe]", "Vsyst": "$V$", "sigma": "$\sigma$",
              "Cr": "[Cr/H]", "Ba": "[Ba/H]", "Ni": "[Ni/H]", "Co": "[Co/H]",
              "Eu": "[Eu/H]", "Sr": "[Sr/H]", "V": "[V/H]", "Cu": "[Cu/H]",
              "Mn": "[Mn/H]"}
    # Getting numpy array with trace
    tdata = np.array([trace[p].data for p in sed.parnames]).T
    # Arrays for the clean plot
    w = np.ma.masked_array(wave, mask=pmask)
    f = np.ma.masked_array(flux, mask=pmask)
    ferr = np.ma.masked_array(fluxerr, mask=pmask)
    # Defining percentiles/colors for model plots
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    # Calculating models
    models = np.zeros((len(trace), len(wave)))
    for i in tqdm(range(len(trace)), desc="Loading spectra for plots"):
        models[i] = sed(tdata[i])
    m50 = np.median(models, axis=0)
    mperc = np.zeros((len(percs), len(wave)))
    for i, per in enumerate(percs):
        mperc[i] = np.percentile(models, per, axis=0)
    # Calculating sky model if necessary
    skyspec = np.zeros((len(trace), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = trace[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
    sky50 = np.median(skyspec, axis=0)
    s50 = np.ma.masked_array(sky50, mask=pmask)
    # Starting plot
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * fig_width, 3))
    # Main plot
    ax = fig.add_subplot(axs[0])
    ax.plot(w, f, "-", c="0.8", lw=lw)
    ax.fill_between(w, f + ferr, f - ferr, color="C0", alpha=0.7)
    ax.plot(w, f - s50, "-", label=galaxy, lw=lw)
    ax.plot(w, m50 - s50, "-", lw=lw, label="Model")
    for c, per in zip(colors, percs):
        y1 = np.ma.masked_array(np.percentile(models, per, axis=0) - sky50,
                                mask=pmask)
        y2 = np.ma.masked_array(np.percentile(models, per + 10, axis=0) - sky50,
                                mask=pmask)
        ax.fill_between(w, y1, y2, color=c)
    ax.set_ylabel("Normalized flux")
    ax.xaxis.set_ticklabels([])
    plt.legend()
    # Residual plot
    ax = fig.add_subplot(axs[1])
    for c, per in zip(colors, percs):
        y1 = 100 * (flux - np.percentile(models, per, axis=0)) / flux
        y2 = 100 * (flux - np.percentile(models, per + 10, axis=0)) / flux
        y1 = np.ma.masked_array(y1, mask=pmask)
        y2 = np.ma.masked_array(y2, mask=pmask)
        ax.fill_between(w, y1, y2, color=c)
    rmse = np.std((f - m50)/flux)
    ax.plot(w, 100 * (f - m50) / f, "-", lw=lw, c="C1",
            label="RMSE={:.1f}\%".format(100 * rmse))
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel("Residue (\%)")
    ax.set_ylim(-3 * 100 * rmse, 3 * 100 * rmse)
    plt.legend()
    plt.subplots_adjust(left=0.065, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    plt.savefig("{}.png".format(outfig), dpi=250)
    plt.show()
    return


def run_nscs(sigma=100, r=3, r1=5, r2=8, nsteps=6000, redo=False):
    """ Run paintbox on test galaxies. """
    global logp
    global priors
    wdir = os.path.join(context.home_dir, f"paintbox/sigma{sigma}")
    galaxies = os.listdir(wdir)
    vels = {"FCC223": 900}
    for galaxy in galaxies:
        gal_dir = os.path.join(wdir, galaxy)
        filename = f"{galaxy}_r_{r}_r1_{r1}_r2_{r2}_sig_{sigma}.fits"
        tablename = os.path.join(gal_dir, filename)
        table_cen =  Table.read(tablename, hdu=1)
        table_ann = Table.read(tablename, hdu=2)
        wave = table_cen["wave"].data
        sed_ann, sed_cen, limits = model_nscs_plus_annulus(wave, sigma=sigma)
        logp_ann = pb.Normal2LogLike(table_ann["flux"].data, sed_ann,
                                     obserr=table_ann["flux"].data,
                                     mask=table_ann["mask"].data)
        logp_cen = pb.Normal2LogLike(table_cen["flux"].data, sed_cen,
                                     obserr=table_cen["flux"].data,
                                     mask=table_cen["mask"].data)
        logp = logp_cen + logp_ann
        priors = set_priors(logp.parnames, limits, vsyst=vels[galaxy])
        # Run in any directory outside Dropbox to avoid conflicts
        dbname = f"{galaxy}_nsteps{nsteps}.h5"
        tmp_db = os.path.join(os.getcwd(), dbname)
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
        outdb = os.path.join(wdir, dbname)
        if not os.path.exists(outdb) or redo:
            run_sampler(tmp_db, nsteps=nsteps)
            shutil.move(tmp_db, outdb)
        # Load database and make a table with summary statistics
        reader = emcee.backends.HDFBackend(outdb)
        tracedata = reader.get_chain(discard=int(nsteps * 0.94), flat=True,
                                     thin=100)
        trace = Table(tracedata, names=logp.parnames)
        outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
        make_table(trace, outtab)
        # if platform.node() == "kadu-Inspiron-5557":
        #     plot_fitting(wave, flux, fluxerr, sed, trace,  outdb,
        #                  mask=mask, galaxy=galaxy)

if __name__ == "__main__":
    run_nscs()

