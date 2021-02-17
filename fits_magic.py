import numpy as np

from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from radio_beam import Beam
from radio_beam import EllipticalGaussian2DKernel

from scipy.fft import ifft2, ifftshift


def fits_transfer_coordinates(fromfits, tofits):
    """
    transfer RA and DEC from one fits to another
    """
    with pyfits.open(fromfits) as f:
        hdr = f[0].header
        crval1 = f[0].header['CRVAL1']
        crval2 = f[0].header['CRVAL2']
    with pyfits.open(tofits, mode='update') as hdul:
        hdul[0].header['CRVAL1'] = crval1
        hdul[0].header['CRVAL2'] = crval2
        hdul.flush()
    return tofits


def fits_squeeze(fitsfile, out=None):
    """
    remove extra dimensions from data and modify header
    """
    if out is None:
        out = fitsfile

    with pyfits.open(fitsfile) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
        hdul[0].header['NAXIS'] = 2
        for i in [3, 4]:
            for key in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                header.remove('{}{}'.format(key,i), ignore_missing=True, remove_all=True)
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
        out = fitsfile
    with pyfits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        currentpsf = Beam.from_fits_header(hdr)
        # if currentpsf != newpsf:
        #     kern = newpsf.deconvolve(currentpsf).as_kernel(pixscale=hdr['CDELT2']*u.deg)
        #     hdr.set('BMAJ', newparams['BMAJ'])
        #     hdr.set('BMIN', newparams['BMIN'])
        #     hdr.set('BPA', newparams['BPA'])
        #     hdul[0].data = convolve(hdul[0].data, kern)
        # pyfits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
        if currentpsf != newpsf:
            kmaj1 = (currentpsf.major.to('deg').value / hdr['CDELT2'])
            kmin1 = (currentpsf.minor.to('deg').value / hdr['CDELT2'])
            kpa1 = currentpsf.pa.to('deg').value
            kmaj2 = (newpsf.major.to('deg').value / hdr['CDELT2'])
            kmin2 = (newpsf.minor.to('deg').value / hdr['CDELT2'])
            kpa2 = newpsf.pa.to('deg').value
            norm = newpsf.to_value() / currentpsf.to_value()
            if len(hdul[0].data.shape) == 4:
                conv_data = hdul[0].data[0, 0, ...]
            elif len(hdul[0].data.shape) == 2:
                conv_data = hdul[0].data
            # deconvolve with the old PSF
            # conv_data = convolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1, inverse=True)
            # convolve to the new PSF
            conv_data = norm * reconvolve_gaussian_kernel(conv_data, kmaj1, kmin1, kpa1, kmaj2, kmin2, kpa2)

            if len(hdul[0].data.shape) == 4:
                hdul[0].data[0, 0, ...] = conv_data
            elif len(hdul[0].data.shape) == 2:
                hdul[0].data = conv_data
            hdr = newpsf.attach_to_header(hdr)
        pyfits.writeto(out, data=hdul[0].data, header=hdr, overwrite=True)
    return out


def fits_operation(fitsfile, other, operation='-', out=None):
    """
    perform operation on fits file and other fits/array/number,
    keeping header of the original FITS one
    """
    if out is None:
        out = fitsfile

    if isinstance(other, str):
        other_data = pyfits.getdata(other)
    elif isinstance(other, np.ndarray) or isinstance(other, np.float):
        other_data = other
    with pyfits.open(fitsfile) as hdul:
        data = hdul[0].data
        if operation == '-':
            data -= other_data
        elif operation == '+':
            data += other_data
        elif operation == '*':
            data *= other_data
        elif operation == '/':
            data /= other_data
        pyfits.writeto(out, data=data, header=hdul[0].header, overwrite=True)
    return out


def fits_crop(fitsfile, out=None):
    # Load the image and the WCS
    if out is None:
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


def get_beam(fitsfile):
    with pyfits.open(fitsfile) as hdu:
        hdu_header = hdu[0].header
        bmaj = hdu_header['BMAJ']
        bmin = hdu_header['BMIN']
    return bmaj, bmin


def get_rms(fitsfile):
    with pyfits.open(fitsfile) as hdu:
        data = hdu[0].data
        rms = np.std(data)
    return rms