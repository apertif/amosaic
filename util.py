import glob
import os
from configparser import ConfigParser

import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits as pyfits
from radio_beam import Beam, Beams


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
    Funtion to generate the neccessary continuum directories
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
    if self.mode == 'all':
        # copy all the images from the continuum directory
        for image in range(40):
            os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image* ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
    elif self.mode == 'qa':
        # Load the qa-continuum file and only copy the images with good quality
        c_arr = np.full(40, True)
        if os.path.isfile(os.path.join(self.qacontdir, self.obsid, 'dynamicRange.dat')):
            data = ascii.read(os.path.join(self.qacontdir, self.obsid, 'dynamicRange.dat'))
            c_arr[np.where(data['col2'] == 'X')] = False
            for image in range(40):
                if c_arr[image]:
                    os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image* ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
                else:
                    print('Image for beam ' + str(image).zfill(2) + ' not available or validated as bad!')
        else:
            print('No continuum quality assurance available for observation id ' + str(self.obsid) + '. Copying all available images.')
            for image in range(40):
                os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image* ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')
    elif (type(self.mode) == list):
        # Copy only the beams given as a list
        for image in mode:
            os.system('cp ' + os.path.join(self.basedir, self.obsid) + '/' + str(image).zfill(2) + '/continuum/image* ' + self.contimagedir + '/I' + str(image).zfill(2) + '.fits')


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
    images = glob.glob(self.contimagedir + '/I[0-9][0-9].fits')
    pbimages = glob.glob(self.contbeamdir + '/B[0-9][0-9].fits')
    return images, pbimages


def clean_contmosaic_tmp_data(self):
    os.system('rm -rf ' + self.contimagedir + '/*_tmp.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_repr.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_reconv.fits')
    os.system('rm -rf ' + self.contimagedir + '/*_pbcorr.fits')
    os.system('rm -rf ' + self.contimagedir + '/casa*.log')


def get_common_psf(fitsfiles):
    """ common psf for the list of fits files """
    beams = []
    bmajes = []
    bmines = []
    bpas = []
    for f in fitsfiles:
        ih = pyfits.getheader(f)
        bmajes.append(ih['BMAJ'])
        bmines.append(ih['BMIN'])
        bpas.append(ih['BPA'])
        beam = Beam.from_fits_header(ih)
        beams.append(beam)
    beams = Beams(bmajes * u.deg, bmines * u.deg, bpas * u.deg)
    common = beams.common_beam()
    smallest = beams.smallest_beam()
    print('PSF:\n  Smallest PSF: %s\n  Common PSF: %s', smallest, common)
    return common