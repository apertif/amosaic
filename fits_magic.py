import numpy as np
from astropy import units as u
from astropy.convolution import convolve
from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from radio_beam import Beam


def fits_transfer_coordinates(fromfits, tofits):
    """
    transfer RA and Dec from one fits to another
    """

    with pyfits.open(fromfits) as f:
        hdr = f[0].header
        crval1 = f[0].header['CRVAL1']
        crval2 = f[0].header['CRVAL2']
    with pyfits.open(tofits, mode='update') as hdul:
        hdul[0].header['CRVAL1'] = crval1
        hdul[0].header['CRVAL2'] = crval2
        print('fits_transfer_coordinates: Updating fits header (CRVAL1/2) in %s', tofits)
        hdul.flush()
    return tofits


def fits_squeeze(fitsfile, out=None):
    """
    remove extra dimentions from data and modify header
    """
    if out is None:
        print('fits_squeeze: Overwriting file %s', fitsfile)
        out = fitsfile

    with pyfits.open(fitsfile) as hdul:
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
        pyfits.writeto(out, data=data, header=header, overwrite=True)
    return out


def fits_reconvolve_psf(fitsfile, newpsf, out=None):
    """ Convolve image with deconvolution of (newpsf, oldpsf) """
    newparams = newpsf.to_header_keywords()
    if out is None:
        print('fits_reconvolve: Overwriting file %s', fitsfile)
        out = fitsfile
    with pyfits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        print(currentpsf)
        print(newpsf)
        if currentpsf != newpsf:
            kern = newpsf.deconvolve(currentpsf).as_kernel(pixscale=hdr['CDELT2']*u.deg)
            hdr.set('BMAJ', newparams['BMAJ'])
            hdr.set('BMIN', newparams['BMIN'])
            hdr.set('BPA', newparams['BPA'])
            hdul[0].data = convolve(hdul[0].data, kern)
        pyfits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
    return out


def fits_operation(fitsfile, other, operation='-', out=None):
    """
    perform operation on fits file and other fits/array/number,
    keeping header of the original FITS one
    """
    if out is None:
        print('fits_operation: Overwriting file %s', fitsfile)
        out = fitsfile

    if isinstance(other, str):
        other_data = pyfits.getdata(other)
    elif isinstance(other, np.ndarray) or isinstance(other, np.float):
        other_data = other
    with pyfits.open(fitsfile) as hdul:
        data = hdul[0].data
        if operation == '-':
            print('fits_operation: Subtracting data')
            data -= other_data
        elif operation == '+':
            print('fits_operation: Adding data')
            data += other_data
        elif operation == '*':
            print('fits_operation: Multiplying data')
            data *= other_data
        elif operation == '/':
            print('fits_operation: Dividing data')
            data /= other_data
        pyfits.writeto(out, data=data, header=hdul[0].header, overwrite=True)
    return out


def fits_crop(fitsfile, out=None):
    # Load the image and the WCS
    if out is None:
        print('fits_crop: Overwriting file %s', fitsfile)
        out = fitsfile
    with pyfits.open(fitsfile) as f:
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