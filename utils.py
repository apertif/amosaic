import glob
import os
import sys
from configparser import ConfigParser

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits as pyfits
from radio_beam import Beam, Beams, commonbeam


def load_config(config_object, file_=None):
    """
    Function to load the config file
    """
    config = ConfigParser()  # Initialise the config parser
    config.readfp(open(file_))
    for s in config.sections():
        for o in config.items(s):
            setattr(config_object, o[0], eval(o[1]))
    return config  # Save the loaded config file as defaults for later usage


def set_mosdirs(self):
    """
    Creates the directory names for the subdirectories to make scripting easier
    """
    self.qacontdir = os.path.join(self.qadir, 'continuum')
    self.qapoldir = os.path.join(self.qadir, 'polarisation')

    self.contworkdir = os.path.join(self.basedir, self.obsid, self.mossubdir, self.moscontdir)
    self.contimagedir = os.path.join(self.contworkdir, 'images')
    self.contbeamdir = os.path.join(self.contworkdir, 'beams')
    self.contmosaicdir = os.path.join(self.contworkdir, 'mosaic')

    self.polworkdir = os.path.join(self.basedir, self.obsid, self.mossubdir, self.mospoldir)
    self.polimagedir = os.path.join(self.polworkdir, 'images')
    self.polbeamdir = os.path.join(self.polworkdir, 'beams')
    self.polmosaicdir = os.path.join(self.polworkdir, 'mosaic')


def gen_contdirs(self):
    """
    Function to generate the necessary continuum directories
    """
    if os.path.isdir(self.contworkdir):
        pass
    else:
        os.makedirs(self.contworkdir)

    if os.path.isdir(self.contimagedir):
        pass
    else:
        os.makedirs(self.contimagedir)

    if os.path.isdir(self.contbeamdir):
        pass
    else:
        os.makedirs(self.contbeamdir)

    if os.path.isdir(self.contmosaicdir):
        pass
    else:
        os.makedirs(self.contmosaicdir)


def copy_contimages(self):
    """
    Function to copy the continuum images to the working directory
    """
    if self.cont_mode == 'all':
        # copy all the images from the continuum directory
        for image in range(40):
            os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image_mf_*.fits ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
    elif self.cont_mode == 'qa':
        # Load the qa-continuum file and only copy the images with good quality
        c_arr = np.full(40, True)
        if os.path.isfile(os.path.join(self.qacontdir, self.obsid, 'dynamicRange.dat')):
            data = ascii.read(os.path.join(self.qacontdir, self.obsid, 'dynamicRange.dat'))
            c_arr[np.where(data['col2'] == 'X')] = False
            for image in range(40):
                if c_arr[image]:
                    os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image_mf_*.fits ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
                else:
                    print('Image for beam ' + str(image).zfill(2) + ' not available or validated as bad!')
        else:
            print('No continuum quality assurance available for observation id ' + str(self.obsid) + '. Copying all available images.')
            for image in range(40):
                os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image_mf_*.fits ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
    elif (type(self.cont_mode) == list):
        # Copy only the beams given as a list
        for image in self.cont_mode:
            os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image_mf_*.fits ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')


def copy_contbeams(self):
    """
    Find the right beam models in time and frequency for the appropriate beams and copy them over to the working directory
    """
    # Get the right directory with the minimum difference in time with regard to the observation
    beamtimes = sorted(glob.glob(self.beamsrcdir + '*'))
    beamtimes_arr = [float(bt.split('/')[-1][:6]) for bt in beamtimes]
    bt_array = np.unique(beamtimes_arr)
    obstime = float(self.obsid[:6])
    deltat = np.abs(bt_array - obstime)
    loc_min = np.argmin(deltat)
    rightbeamdir = beamtimes[loc_min]

    # Get the frequencies of the beam models
    channs = sorted(glob.glob(os.path.join(rightbeamdir, 'beam_models/chann_[0-9]')))
    freqs = np.full(len(channs), np.nan)
    for b, beam in enumerate(channs):
        hdul = pyfits.open(os.path.join(beam, rightbeamdir.split('/')[-1] + '_00_I_model.fits'))
        freqs[b] = hdul[0].header['CRVAL3']
        hdul.close()

    # Copy the beam models with the right frequency over to the working directory
    for beam in range(40):
        if os.path.isfile(self.contimagedir + '/I' + str(beam).zfill(2) + '.fits'):
            hdulist = pyfits.open(self.contimagedir + '/I' + str(beam).zfill(2) + '.fits')
            freq = hdulist[0].header['CRVAL3']
            nchann = np.argmin(np.abs(freqs - freq)) + 1
            os.system('cp ' + os.path.join(rightbeamdir, 'beam_models/chann_' + str(nchann) + '/') + rightbeamdir.split('/')[-1] + '_' + str(beam).zfill(2) + '_I_model.fits ' + self.contbeamdir + '/B' + str(beam).zfill(2) + '.fits')


def get_contfiles(self):
    """
    Get a list of the images and pbimages in the continuum working directory
    """
    images = sorted(glob.glob(self.contimagedir + '/I[0-9][0-9].fits'))
    pbimages = sorted(glob.glob(self.contbeamdir + '/B[0-9][0-9].fits'))
    return images, pbimages


def clean_contmosaic_tmp_data(self):
    os.system('rm -rf ' + self.contimagedir + '/*_tmp.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_repr.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_reconv.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_pbcorr.fits')
    os.system('rm -rf ' + self.contimagedir + '/casa*.log')


def clean_polmosaic_tmp_data(self):
    os.system('rm -rf ' + self.polimagedir + '/*_tmp.fits')
    os.system('rm -rf ' + self.polimagedir + '/*_repr.fits')
    os.system('rm -rf ' + self.polimagedir + '/*_reconv.fits')
    os.system('rm -rf ' + self.polimagedir + '/*_pbcorr.fits')
    os.system('rm -rf ' + self.polimagedir + '/casa*.log')
    os.system('rm -rf ' + self.polimagedir + '/Q*.fits')
    os.system('rm -rf ' + self.polimagedir + '/U*.fits')
    os.system('rm -rf ' + self.polbeamdir + '/B*.fits')


def gen_poldirs(self):
    """
    Function to generate the necessary polarisation directories
    """
    if os.path.isdir(self.polworkdir):
        pass
    else:
        os.makedirs(self.polworkdir)

    if os.path.isdir(self.polimagedir):
        pass
    else:
        os.makedirs(self.polimagedir)

    if os.path.isdir(self.polbeamdir):
        pass
    else:
        os.makedirs(self.polbeamdir)

    if os.path.isdir(self.polmosaicdir):
        pass
    else:
        os.makedirs(self.polmosaicdir)


def copy_polimages(self, sb, veri):
    """
    Function to copy the polarisation images of a specific subband to the working directory
    """
    for b in range(40):
        if veri[b]:
            qcube = pyfits.open(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/Qcube.fits'))
            ucube = pyfits.open(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/Ucube.fits'))
            qhdu = qcube[0]
            uhdu = ucube[0]
            qhdr = qhdu.header
            uhdr = uhdu.header
            qplane = qhdu.data[sb,:,:]
            uplane = uhdu.data[sb,:,:]
            newqfreq = qhdr['CRVAL3'] + float(sb) * qhdr['CDELT3']
            newufreq = uhdr['CRVAL3'] + float(sb) * uhdr['CDELT3']
            qhdr.update(NAXIS3=1, CRVAL3=newqfreq)
            uhdr.update(NAXIS3=1, CRVAL3=newufreq)
            pyfits.writeto(self.polimagedir + '/Q' + str(b).zfill(2) + '.fits', data=qplane, header=qhdr)
            pyfits.writeto(self.polimagedir + '/U' + str(b).zfill(2) + '.fits', data=uplane, header=uhdr)


def copy_polbeams(self):
    """
    Find the right beam models in time and frequency for the appropriate beams and copy them over to the working directory
    """
    # Get the right directory with the minimum difference in time with regard to the observation
    beamtimes = sorted(glob.glob(self.beamsrcdir + '*'))
    beamtimes_arr = [float(bt.split('/')[-1][:6]) for bt in beamtimes]
    bt_array = np.unique(beamtimes_arr)
    obstime = float(self.obsid[:6])
    deltat = np.abs(bt_array - obstime)
    loc_min = np.argmin(deltat)
    rightbeamdir = beamtimes[loc_min]

    # Get the frequencies of the beam models
    channs = sorted(glob.glob(os.path.join(rightbeamdir, 'beam_models/chann_[0-9]')))
    freqs = np.full(len(channs), np.nan)
    for b, beam in enumerate(channs):
        hdul = pyfits.open(os.path.join(beam, rightbeamdir.split('/')[-1] + '_00_I_model.fits'))
        freqs[b] = hdul[0].header['CRVAL3']
        hdul.close()

    # Copy the beam models with the right frequency over to the working directory
    for beam in range(40):
        if os.path.isfile(self.polimagedir + '/Q' + str(beam).zfill(2) + '.fits'):
            hdulist = pyfits.open(self.polimagedir + '/Q' + str(beam).zfill(2) + '.fits')
            freq = hdulist[0].header['CRVAL3']
            nchann = np.argmin(np.abs(freqs - freq)) + 1
            os.system('cp ' + os.path.join(rightbeamdir, 'beam_models/chann_' + str(nchann) + '/') + rightbeamdir.split('/')[-1] + '_' + str(beam).zfill(2) + '_I_model.fits ' + self.polbeamdir + '/B' + str(beam).zfill(2) + '.fits')


def get_polfiles(self):
    """
    Get a list of the images and pbimages in the polarisation working directory
    """
    qimages = sorted(glob.glob(self.polimagedir + '/Q[0-9][0-9].fits'))
    uimages = sorted(glob.glob(self.polimagedir + '/U[0-9][0-9].fits'))
    pbimages = sorted(glob.glob(self.polbeamdir + '/B[0-9][0-9].fits'))
    return qimages, uimages, pbimages


def get_common_psf(self, input, format='fits'):
    """
    Common psf for the list of fits files
    """
    beams = []
    bmajes = []
    bmines = []
    bpas = []
    if format == 'fits':
        for f in input:
            ih = pyfits.getheader(f)
            bmajes.append(ih['BMAJ'])
            bmines.append(ih['BMIN'])
            bpas.append(ih['BPA'])
#            beam = Beam.from_fits_header(ih)
#            print(beam)
#            beams.append(beam)
        for i in range(0, len(bmajes) - 1):
            ni = i + 1
            beams = Beams((bmajes[[i, ni]]) * u.deg, (bmines[[i, ni]]) * u.deg, bpas[[i, ni]] * u.deg)
            common = commonbeam.commonbeam(beams)
            bmajes[ni] = common.major/u.deg
            bmines[ni] = common.minor / u.deg
            bpas[ni] = common.pa / u.deg
    elif format == 'array':
        accept_array = np.loadtxt(self.polmosaicdir + '/accept_array.npy')
        bacc_array = np.loadtxt(self.polmosaicdir + '/bacc.npy')
        for b in range(40):
            if input[b]:
                bmajes.extend(get_param(self, 'polarisation_B' + str(b).zfill(2) + '_targetbeams_qu_beamparams')[:, 0, 0])
                bmines.extend(get_param(self, 'polarisation_B' + str(b).zfill(2) + '_targetbeams_qu_beamparams')[:, 1, 0])
                bpas.extend(get_param(self, 'polarisation_B' + str(b).zfill(2) + '_targetbeams_qu_beamparams')[:, 2, 0])
        sb_bad = np.full(24, True, dtype=bool)
        for sb in range(24):
            if np.sum(accept_array[:,sb]) < np.sum(bacc_array):
                sb_bad[sb] = False
        beams_bad = np.tile(sb_bad, int(np.sum(bacc_array)))
        for idx, image in enumerate(beams_bad):
            if image:
                continue
            else:
                bmajes[idx] = np.nan
                bmines[idx] = np.nan
                bpas[idx] = np.nan
        bmajarr = np.array(bmajes)
        bminarr = np.array(bmines)
        bpaarr = np.array(bpas)
        bmajarr = bmajarr[~pd.isnull(bmajarr)]
        bminarr = bminarr[~pd.isnull(bminarr)]
        bpaarr = bpaarr[~pd.isnull(bpaarr)]
        for i in range(0, len(bmajarr) - 1):
            ni = i + 1
            beams = Beams((bmajarr[[i,ni]]/3600.0) * u.deg, (bminarr[[i,ni]]/3600.0) * u.deg, bpaarr[[i,ni]] * u.deg)
            common = commonbeam.commonbeam(beams)
            bmajarr[ni] = (common.major / u.deg) * 3600.0
            bminarr[ni] = (common.minor / u.deg) * 3600.0
            bpaarr[ni] = common.pa / u.deg
#        beams = Beams((bmajarr/3600.0) * u.deg, (bminarr/3600.0) * u.deg, bpaarr * u.deg)
#    common = beams.common_beam()
#    common = commonbeam.common_manybeams_opt(beams, p0=(np.max(bmajes), np.max(bmines), np.median(bpas)))
#    common = commonbeam.common_manybeams_mve(beams)
#    smallest = beams.smallest_beam()
#    print('PSF:\n  Smallest PSF: %s\n  Common PSF: %s', smallest, common)
    print('The smallest common ' + str(common))
    return common


def collect_paramfiles(self):
    """
    Check if single param.npy files are available and combine them into one if yes
    """
    paramfiles = sorted(glob.glob(os.path.join(self.basedir, self.obsid, 'param_[0-9][0-9].npy')))
    if len(paramfiles)==0:
        if os.path.isfile(os.path.join(self.basedir, self.obsid, 'param.npy')):
            print('Main parameter file found and used')
        else:
            print('Main parameter file not found! Further operations not possible!')
            sys.exit()
    else:
        pfirst = np.load(paramfiles[0], allow_pickle=True, encoding='latin1').item()
        for p, pnext in enumerate(paramfiles):
            if pfirst == pnext:
                continue
            else:
                pnext = np.load(paramfiles[p], allow_pickle=True, encoding='latin1').item()
                pfirst.update(pnext)
        np.save(os.path.join(self.basedir, self.obsid, 'param.npy'), pfirst)


def get_param(self, pmname):
    """
    Load a keyword of the parameter file into a variable
    """
    d = np.load(os.path.join(self.basedir, self.obsid, 'param.npy'), allow_pickle=True, encoding='latin1').item()
    values = d[pmname]
    return values