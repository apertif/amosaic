#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:21:20 2020

@author: kutkin
"""
import glob
import os
import sys
import numpy as np
import subprocess
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.colors as mc
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, convolve_fft
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import argparse

from radio_beam import Beam, Beams
from radio_beam import EllipticalGaussian2DKernel

from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

import logging

from scipy.fft import fft2, fftshift, ifft2, ifftshift



def clean_mosaic_tmp_data(path='.'):

    cmd = 'cd {path} && rm *_tmp*.fits'.format(path=path)
    subprocess.call(cmd, shell=True)


def make_tmp_copy(fname):
    base, ext = os.path.splitext(fname)
    tempname = os.path.basename(fname.replace(ext, '_tmp{}'.format(ext)))
    subprocess.call('cp {} {}'.format(fname, tempname), shell=True)
    return tempname


def fits_transfer_coordinates(fromfits, tofits):
    """
    transfer RA and Dec from one fits to another
    """

    with fits.open(fromfits) as f:
        hdr = f[0].header
        crval1 = f[0].header['CRVAL1']
        crval2 = f[0].header['CRVAL2']
    with fits.open(tofits, mode='update') as hdul:
        hdul[0].header['CRVAL1'] = crval1
        hdul[0].header['CRVAL2'] = crval2
        logging.debug('fits_transfer_coordinates: Udating fits header (CRVAL1/2) in %s', tofits)
        hdul.flush()
    return tofits


def fits_squeeze(fitsfile, out=None):
    """
    remove extra dimentions from data and modify header
    """
    if out is None:
        logging.debug('fits_squeeze: Overwriting file %s', fitsfile)
        out = fitsfile
    with fits.open(fitsfile) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
        hdul[0].header['NAXIS'] = 2
        for i in [3, 4]:
            for key in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                header.remove('{}{}'.format(key,i), ignore_missing=True,
                                      remove_all=True)
                header.remove('LTYPE', ignore_missing=True)
                header.remove('LSTART', ignore_missing=True)
                header.remove('LSTEP', ignore_missing=True)
                header.remove('LWIDTH', ignore_missing=True)
                header.remove('LONPOLE', ignore_missing=True)
                header.remove('LATPOLE', ignore_missing=True)
                header.remove('RESTFRQ', ignore_missing=True)
                header.remove('WCSAXES', ignore_missing=True)
        fits.writeto(out, data=data, header=header, overwrite=True)
    return out


def fits_operation(fitsfile, other, operation='-', out=None):
    """
    perform operation on fits file and other fits/array/number,
    keeping header of the original FITS one
    """
    if out is None:
        logging.debug('fits_operation: Overwriting file %s', fitsfile)
        out = fitsfile
    if isinstance(other, str):
        other_data = fits.getdata(other)
    elif isinstance(other, np.ndarray) or isinstance(other, np.float):
        other_data = other
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data
        if operation == '-':
            logging.debug('fits_operation: Subtracting data')
            data -= other_data
        elif operation == '+':
            logging.debug('fits_operation: Adding data')
            data += other_data
        elif operation == '*':
            logging.debug('fits_operation: Multiplying data')
            data *= other_data
        elif operation == '/':
            logging.debug('fits_operation: Dividing data')
            data /= other_data
        fits.writeto(out, data=data, header=hdul[0].header, overwrite=True)
    return out


def fft_psf(bmaj, bmin, bpa, size=3073):
    SIGMA_TO_FWHM = np.sqrt(8*np.log(2))
    fmaj = size / (bmin / SIGMA_TO_FWHM) / 2 / np.pi
    fmin = size / (bmaj / SIGMA_TO_FWHM) / 2 / np.pi
    fpa = bpa + 90
    angle = np.deg2rad(90+fpa)
    fkern = EllipticalGaussian2DKernel(fmaj, fmin, angle, x_size=size, y_size=size)
    fkern.normalize('peak')
    fkern = fkern.array
    return fkern


def reconvolve_gaussian_kernel(img, old_maj, old_min, old_pa, new_maj, new_min, new_pa):
    """
    convolve image with a gaussian kernel without FFTing it
    bmaj, bmin -- in pixels,
    bpa -- in degrees from top clockwise (like in Beam)
    inverse -- use True to deconvolve.
    NOTE: yet works for square image without NaNs
    """
    size = len(img)
    imean = img.mean()
    img -= imean
    fimg = np.fft.fft2(img)
    krel = fft_psf(new_maj, new_min, new_pa, size) / fft_psf(old_maj, old_min, old_pa, size)
    fconv = fimg * ifftshift(krel)
    return ifft2(fconv).real + imean


def fits_reconvolve_psf(fitsfile, newpsf, out=None):
    """ Convolve image with deconvolution of (newpsf, oldpsf) """
    newparams = newpsf.to_header_keywords()
    if out is None:
        logging.debug('fits_reconvolve: Overwriting file %s', fitsfile)
        out = fitsfile
    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        if currentpsf != newpsf:
            kmaj1 = (currentpsf.major.to('deg').value/hdr['CDELT2'])
            kmin1 = (currentpsf.minor.to('deg').value/hdr['CDELT2'])
            kpa1 = currentpsf.pa.to('deg').value
            kmaj2 = (newpsf.major.to('deg').value/hdr['CDELT2'])
            kmin2 = (newpsf.minor.to('deg').value/hdr['CDELT2'])
            kpa2 = newpsf.pa.to('deg').value
            norm = newpsf.to_value() / currentpsf.to_value()
            if len(hdul[0].data.shape) == 4:
                conv_data = hdul[0].data[0,0,...]
            elif len(hdul[0].data.shape) == 2:
                conv_data = hdul[0].data
            # deconvolve with the old PSF
            # conv_data = convolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1, inverse=True)
            # convolve to the new PSF
            conv_data = norm * reconvolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1,
                                                                     kmaj2, kmin2, kpa2)

            if len(hdul[0].data.shape) == 4:
                hdul[0].data[0,0,...] = conv_data
            elif len(hdul[0].data.shape) == 2:
                hdul[0].data = conv_data
            hdr = newpsf.attach_to_header(hdr)
        fits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
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
    logging.info('Smallest PSF: %s', smallest)
    logging.info('Common PSF: %s', common)
    return common


def crop_image(img):
    mask = np.isfinite(img)
    # if img.ndim==3:
        # mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0),mask.any(1)
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def fits_crop(fitsfile, out=None):
    # Load the image and the WCS
    if out is None:
        logging.debug('fits_crop: Overwriting file %s', fitsfile)
        out = fitsfile
    with fits.open(fitsfile) as f:
        hdu = f[0]
        data = f[0].data
        header = f[0].header
        wcs = WCS(header)
# crop the image
        xcen, ycen = header['CRPIX1'], header['CRPIX2']
        mask = np.isfinite(data)
        m, n = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
        col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
        ysize = 2 * max(abs(ycen - row_start), abs(row_end - ycen))
        xsize = 2 * max(abs(xcen - col_start), abs(col_end - xcen))
        # print(xcen, ycen, xsize, ysize)
# Make the cutout, including the WCS
        cutout = Cutout2D(data, position=(xcen, ycen), size=(ysize, xsize), wcs=wcs)
# Put the cutout image in the FITS HDU
        hdu.data = cutout.data
# Update the FITS header with the cutout WCS
        hdu.header.update(cutout.wcs.to_header())
# Write the cutout to a new FITS file
        hdu.writeto(out, overwrite=True)
    return out, cutout


def fits_transpose(fitsfile):
    logging.warning('Transposing data in %s', fitsfile)
    with fits.open(fitsfile, mode='update') as hdul:
        hdul[0].data = hdul[0].data.T
        hdul.flush()
    return fitsfile



def pbcorrect(image, pbimage, pbclip=None, rmnoise=False, out=None):
    tmpimg = make_tmp_copy(image)
    tmppb = make_tmp_copy(pbimage)
    tmpimg = fits_squeeze(tmpimg) # remove extra dimentions
    tmppb = fits_transfer_coordinates(tmpimg, tmppb) # transfer_coordinates
    tmppb = fits_squeeze(tmppb) # remove extra dimentions
    with fits.open(tmpimg) as f:
        imheader = f[0].header
        imdata = f[0].data
    with fits.open(tmppb) as f:
        pbhdu = f[0]
        pbarray = f[0].data
        if pbarray.shape != imdata.shape:
        # autoclip = np.nanmin(f[0].data)
# reproject
            logging.info('Regridding pb-model')
            pbarray, reproj_footprint = reproject_interp(pbhdu, imheader)
        else:
            logging.debug('PB has the same shape. No regridding needed.')
    pbarray = np.float32(pbarray)
    if pbclip is not None:
    # pbclip = pbclip or autoclip
        pbarray[pbarray < pbclip] = np.nan
        logging.info('PB is clipped at %f level', pbclip)
    else:
        logging.info('Not clipping PB array')
    pb_regr_repr = os.path.basename(tmppb.replace('_tmp.fits', '_repr_tmp.fits'))
    fits.writeto(pb_regr_repr, pbarray, imheader, overwrite=True)
    if out is None:
        img_corr = os.path.basename(image.replace('.fits', '_pbcorr.fits'))
        img_uncorr = os.path.basename(image.replace('.fits', '_uncorr.fits'))
    else:
        img_corr = out
        img_uncorr = os.path.basename(image.replace('.fits', '_uncorr.fits'))

    img_corr = fits_operation(tmpimg, pbarray, operation='/', out=img_corr)
    img_uncorr = fits_operation(img_corr, pbarray, operation='*', out=img_uncorr)

    return img_corr, img_uncorr, pbarray


def main(images, pbimages, reference=None, pbclip=0.1, output='mosaic.fits',
         clean_temporary_files=True, rmnoise=False, logger=None):
    if logger is None:
        logger = logging.getLogger('amos')
    common_psf = get_common_psf(images)

    corrimages = [] # to mosaic
    uncorrimages = []
    pbweights = [] # of the pixels
    rmsweights = [] # of the images themself
    # weight_images = []
    for img, pb in zip(images, pbimages):
        logger.info('Image: %s', img)
        logger.info('PBeam: %s', pb)
# prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)

# convolution with common psf
        reconvolved_image = os.path.basename(img.replace('.fits', '_reconv_tmp.fits'))
        reconvolved_image = fits_reconvolve_psf(img, common_psf, out=reconvolved_image)

# PB correction
        pbcorr_image = os.path.basename(img.replace('.fits', '_pbcorr_tmp.fits'))
        pbcorr_image, uncorr_image, pbarray = pbcorrect(reconvolved_image, pb, pbclip=pbclip,
                                          rmnoise=rmnoise, out=pbcorr_image)
# cropping
        cropped_image = os.path.basename(img.replace('.fits', '_mos.fits'))
        cropped_image, cutout = fits_crop(pbcorr_image, out=cropped_image)

        uncorr_cropped_image = os.path.basename(img.replace('.fits', '_uncorr.fits'))
        uncorr_cropped_image, _ = fits_crop(uncorr_image, out=uncorr_cropped_image)

        corrimages.append(cropped_image)
        uncorrimages.append(uncorr_cropped_image)
# primary beam weights
        wg_arr = pbarray #
        wg_arr[np.isnan(wg_arr)] = 0 # the NaNs weight 0
        wg_arr = wg_arr**2 / np.nanmax(wg_arr**2) # normalize
        wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
        pbweights.append(wcut.data)
# weight the images by RMS noise over the edges
        # imdata = np.squeeze(fits.getdata(img))
        # l, m = imdata.shape[0]//10,  imdata.shape[1]//10
        # mask = np.ones(imdata.shape, dtype=np.bool)
        # mask[l:-l,m:-m] = False
        # img_noise = np.nanstd(imdata[mask])
        # img_weight = 1 / img_noise**2
        # rmsweights.append(img_weight)

# merge the image rms weights and the primary beam pixel weights:
    # weights = [p*r/max(rmsweights) for p, r in zip(pbweights, rmsweights)]

# create the wcs and footprint for output mosaic

    logging.info('Mosaicing...')
    wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=reference)

    array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            input_weights=pbweights)
    array2, _ = reproject_and_coadd(uncorrimages, wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            input_weights=pbweights)

    array = np.float32(array)
    array2 = np.float32(array2)

    # plt.imshow(array)

# insert common PSF into the header
    psf = common_psf.to_header_keywords()
    hdr = wcs_out.to_header()
    hdr.insert('RADESYS', ('FREQ', 1.4E9))
    hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
    hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
    hdr.insert('RADESYS', ('BPA', psf['BPA']))

# insert units to header:
    hdr.insert('RADESYS', ('BUNIT', 'JY/BEAM'))


    fits.writeto(output, data=array,
                 header=hdr, overwrite=True)

    fits.writeto(output.replace('.fits', '_uncorr.fits'), data=array2,
                 header=hdr, overwrite=True)

    logging.info('Wrote %s', output)
    if clean_temporary_files:
        logging.debug('Cleaning directory')
        clean_mosaic_tmp_data('.')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("MOSAIC tool")
    parser = argparse.ArgumentParser(description='Mosaic fits images with primary beam correction and weighting')
    parser.add_argument('-g', '--glob', default='', help='Use glob on the directory. DANGEROUS')
    parser.add_argument('-i', '--images', nargs='+')
    parser.add_argument('-b', '--pbeams', nargs='+')
    parser.add_argument('-r', '--reference', help='Reference RA,Dec (ex. "14h02m43,53d47m10s")')
    parser.add_argument('-c', '--clip', type=float, nargs='?', default=None, help='Pbeam clip')
    parser.add_argument('-o', '--output', nargs='?', default='mosaic.fits', help='Output_file_name')

    args = parser.parse_args()

    images = args.images
    pbimages = args.pbeams
    pbclip = args.clip
    glb = args.glob
    if args.reference:
        ra, dec = args.reference.split(',')
        ref = SkyCoord(ra, dec)
    else:
        ref = None

    if glb:
        logging.warning('--glob key is set. looking for files in %s', glb)
        images = sorted(glob.glob('{}/[1-2][0,9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-3][0-9].fits'.format(glb)))
        pbimages = sorted(glob.glob('{}/[1-2][0,9][0-9][0-9][0-9][0-9]_[0-3][0-9]_I_model.fits'.format(glb)))

    t0 = Time.now()
    main(images, pbimages, reference=ref, pbclip=pbclip)
    extime = Time.now() - t0
    print("Execution time: {:.1f} min".format(extime.to("minute").value))

