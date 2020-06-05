#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:21:20 2020

@author: kutkin
"""
import glob
import os
import numpy as np
import subprocess
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve

from radio_beam import Beam, Beams

from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

import logging
logging.basicConfig(level=logging.INFO)


from atools import utils

def fits_transfer_coordinates(fromfits, tofits, overwrite=True):
    """
    transfer RA and Dec from one fits to another
    """
    if not overwrite:
        tofits = tofits.replace('.fits', '_coords_transferred.fits')
    with fits.open(fromfits) as f:
        hdr = f[0].header
        crval1 = f[0].header['CRVAL1']
        crval2 = f[0].header['CRVAL2']
    with fits.open(tofits, mode='update') as hdul:
        hdul[0].header['CRVAL1'] = crval1
        hdul[0].header['CRVAL2'] = crval2
        logging.warning('fits_transfer_coordinates: Udating fits header (CRVAL1/2) for %s', tofits)
        hdul.flush()
    return tofits


def fits_squeeze(ffile, out=None):
    """
    remove extra dimentions from data and modify header
    """

    with fits.open(ffile, mode='update') as hdul:
        # if len(hdul[0].data.shape) == 2:
        #     logging.debug('Data shape is 2D. returning...')
        #     return ffile
        hdul[0].data = np.squeeze(hdul[0].data)
        hdul[0].header['NAXIS'] = 2
        for i in [3, 4]:
            for key in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                hdul[0].header.remove('{}{}'.format(key,i), ignore_missing=True,
                                      remove_all=True)
                hdul[0].header.remove('LTYPE', ignore_missing=True)
                hdul[0].header.remove('LSTART', ignore_missing=True)
                hdul[0].header.remove('LSTEP', ignore_missing=True)
                hdul[0].header.remove('LWIDTH', ignore_missing=True)
                hdul[0].header.remove('LONPOLE', ignore_missing=True)
                hdul[0].header.remove('LATPOLE', ignore_missing=True)
                hdul[0].header.remove('RESTFRQ', ignore_missing=True)
                hdul[0].header.remove('WCSAXES', ignore_missing=True)
        if out is None:
            logging.warning('fits_squeeze: Overwriting FITS header (2D now)')
            hdul.flush()
            out = ffile
        else:
            logging.info('fits_squeeze: Writing new file: %s', out)
            fits.writeto(out, hdul[0].data, hdul[0].header, overwrite=True)
    return out


def fits_operation(fitsfile, other, operation='-', out=None):
    """
    perform operation on fits file and other fits/array/number, 
    keeping header of the original FITS one
    """
    if out is None:
        logging.warning('fits_operation: Overwriting FITS')
        out = fitsfile
    else:
        cmd = 'cp {} {}'.format(fitsfile, out)
        subprocess.call(cmd, shell=True)
    if isinstance(other, str):
        other_data = fits.getdata(other)
    elif isinstance(other, np.ndarray) or isinstance(other, np.float):
        other_data = other
    with fits.open(out, mode='update') as hdul:
        if operation == '-':
            logging.debug('fits_operation: Subtracting data')
            hdul[0].data = hdul[0].data - other_data
        elif operation == '+':
            logging.debug('fits_operation: Adding data')
            hdul[0].data = hdul[0].data + other_data
        elif operation == '*':
            logging.debug('fits_operation: Multiplying data')
            hdul[0].data = hdul[0].data * other_data
        elif operation == '/':
            logging.debug('fits_operation: Dividing data')
            hdul[0].data = hdul[0].data / other_data
        hdul.flush()
    return out


def fits_reconvolve_psf(fitsfile, newpsf, out=None):
    """ Convolve image with deconvolution of (newpsf, oldpsf) """
    if out is None:
        logging.warning('Overwriting FITS')
        out = fitsfile
    else:
        cmd = 'cp {} {}'.format(fitsfile, out)
        subprocess.call(cmd, shell=True)
    newparams = newpsf.to_header_keywords()
    with fits.open(out, mode='update') as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        kern = newpsf.deconvolve(currentpsf).as_kernel(pixscale=hdr['CDELT2']*u.deg)
        hdr.set('BMAJ', newparams['BMAJ'])
        hdr.set('BMIN', newparams['BMIN'])
        hdr.set('BPA', newparams['BPA'])
        hdul[0].data = convolve(hdul[0].data, kern)
        hdul.flush()
    return out


def get_common_psf(fitsfiles):
    """ common psf for the list of fits files """
    beams = []
    bmajes = []
    bmines = []
    bpas = []
    for f in fitsfiles:
        ih = fits.getheader(f)
        bmajes.append(ih['BMAJ'])
        bmines.append(ih['BMIN'])
        bpas.append(ih['BPA'])
        beam = Beam.from_fits_header(ih)
        beams.append(beam)
    beams = Beams(bmajes * u.deg, bmines * u.deg, bpas * u.deg)
    common = beams.common_beam()
    smallest = beams.smallest_beam()
    logging.info('Smallest beam: %s\nCommon beam: %s', smallest, common)
    return common


def main(images, pbimages, pbclip=0.1):

    common_psf = get_common_psf(ffiles)
    
    images = [] # to mosaic
    pbweights = [] # of the pixels
    rmsweights = [] # of the images themself
    # weight_images = []
    for img, pb in zip(ffiles, pbeams):
        logging.info('MOSAIC:\n  Image: %s\n  PBeam: %s', img, pb)
# prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)        
        img = fits_squeeze(img) # remove extra dimentions
        pb = fits_transfer_coordinates(img, pb) # transfer_coordinates
        pb = fits_squeeze(pb) # remove extra dimentions
        with fits.open(img) as f:
            imheader = f[0].header
            imdata = f[0].data
        with fits.open(pb) as f:
            pbheader = f[0].header
            pbdata = f[0].data
# reproject            
            reproj_arr, reproj_footprint = reproject_interp(f[0], imheader) 
        reproj_arr = np.float32(reproj_arr)
        reproj_arr[reproj_arr < pbclip] = np.nan
        pb_regr_repr = pb.replace('.fits', '_repr.fits')
        fits.writeto(pb_regr_repr, reproj_arr, imheader, overwrite=True)
# primary beam weights
        # weight_images.append(pb_regr_repr)
        wg_arr = reproj_arr - pbclip # the edges weight ~0
        wg_arr[np.isnan(wg_arr)] = 0 # the NaNs weight 0
        wg_arr = wg_arr / np.nanmax(wg_arr) # normalize
        pbweights.append(wg_arr)
# convolution with common psf
        reconvolved_image = img.replace('.fits', '_reconv.fits')
        reconvolved_image = fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
# PB correction
        pbcorr_image = reconvolved_image.replace('.fits', '_pbcorr.fits')
        pbcorr_image = fits_operation(reconvolved_image, reproj_arr, operation='/', out=pbcorr_image)
        images.append(pbcorr_image)

# weight the images by RMS noise over the edges 
        l, m = imdata.shape[0]//10,  imdata.shape[1]//10
        mask = np.ones(imdata.shape, dtype=np.bool)
        mask[l:-l,m:-m] = False
        img_noise = np.nanstd(imdata[mask])
        img_weight = 1 / img_noise**2
        rmsweights.append(img_weight)

# merge the image rms weights and the primary beam pixel weights:
    weights = [p*r/max(rmsweights) for p, r in zip(pbweights, rmsweights)]
    
# create the wcs and footprint for output mosaic
    wcs_out, shape_out = find_optimal_celestial_wcs(images, frame=None, auto_rotate=False, reference=None)
        

    array, footprint = reproject_and_coadd(images, wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            input_weights=weights)
    
    array = np.float32(array)
    array_bg = np.float32(array_bg)
# insert common PSF into the header
    psf = common_psf.to_header_keywords() 
    hdr = wcs_out.to_header()
    hdr.insert('RADESYS', ('FREQ', 1.4E9))
    hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
    hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
    hdr.insert('RADESYS', ('BPA', psf['BPA']))

    fits.writeto('mosaic.fits', data=array, 
                 header=hdr, overwrite=True)


if __name__ == "__main__":
    print("MOSAIC tool")
    # t0 = Time.now()
    # main([],[], pbclip=0.1)
    # extime = Time.now() - t0
    # print("Execution time: {:.1f} min".format(extime.to("minute").value))

    