import os

import numpy as np
from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

import fits_magic as fm
import utils


class continuum_mosaic:
    """
    Class to produce continuum mosaics.
    """
    module_name = 'CONTINUUM MOSAIC'


    def __init__(self, file_=None, **kwargs):
        self.default = utils.load_config(self, file_)
        utils.set_mosdirs(self)
        self.config_file_name = file_


    def go(self):
        """
        Function to generate the continuum mosaic
        """
        self.cp_data()
        images, pbimages = utils.get_contfiles(self)
        self.make_contmosaic(images, pbimages)


    def cp_data(self):
        """
        Function to generate the needed directories and copy the images and beams over
        """
        utils.gen_contdirs(self)
        utils.copy_contimages(self)
        utils.copy_contbeams(self)


    def make_contmosaic(self, images, pbimages, reference=None, rmnoise=False):
        """
        Function to generate the continuum mosaic
        """
        # Get the common psf
        common_psf = utils.get_common_psf(self, images)
        print('Clipping primary beam response at the %f level', str(self.cont_pbclip))

        corrimages = [] # to mosaic
        uncorrimages = []
        pbweights = [] # of the pixels
        freqs = []
        # weight_images = []
        for img, pb in zip(images, pbimages):
            print('Doing primary beam correction for Beam ' + str(img.split('/')[-1].replace('.fits','').lstrip('I')))
            # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
            with pyfits.open(img) as f:
                imheader = f[0].header
                freqs.append(imheader['CRVAl3'])
                tg = imheader['OBJECT']
        # convolution with common psf
            reconvolved_image = img.replace('.fits', '_reconv_tmp.fits')
            reconvolved_image = fm.fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
        # PB correction
            pbcorr_image = reconvolved_image.replace('.fits', '_pbcorr_tmp.fits')
            tmpimg = utils.make_tmp_copy(reconvolved_image)
            tmppb = utils.make_tmp_copy(pb)
            tmpimg = fm.fits_squeeze(tmpimg)  # remove extra dimentions
            tmppb = fm.fits_transfer_coordinates(tmpimg, tmppb)  # transfer_coordinates
            tmppb = fm.fits_squeeze(tmppb)  # remove extra dimentions
            with pyfits.open(tmpimg) as f:
                imheader = f[0].header
            with pyfits.open(tmppb) as f:
                pbhdu = f[0]
                pbheader = f[0].header
                pbarray = f[0].data
                if (imheader['CRVAL1'] != pbheader['CRVAL1']) or (imheader['CRVAL2'] != pbheader['CRVAL2']) or (imheader['CDELT1'] != pbheader['CDELT1']) or (imheader['CDELT2'] != pbheader['CDELT2']):
                    pbarray, reproj_footprint = reproject_interp(pbhdu, imheader)
                else:
                    pass
            pbarray = np.float32(pbarray)
            pbarray[pbarray < self.cont_pbclip] = np.nan
            pb_regr_repr = tmppb.replace('_tmp.fits', '_repr_tmp.fits')
            pyfits.writeto(pb_regr_repr, pbarray, imheader, overwrite=True)
            img_corr = reconvolved_image.replace('.fits', '_pbcorr.fits')
            img_uncorr = reconvolved_image.replace('.fits', '_uncorr.fits')

            img_corr = fm.fits_operation(tmpimg, pbarray, operation='/', out=img_corr)
            img_uncorr = fm.fits_operation(img_corr, pbarray, operation='*', out=img_uncorr)
        # cropping
            cropped_image = img.replace('.fits', '_mos.fits')
            cropped_image, cutout = fm.fits_crop(img_corr, out=cropped_image)

            uncorr_cropped_image = img.replace('.fits', '_uncorr.fits')
            uncorr_cropped_image, _ = fm.fits_crop(img_uncorr, out=uncorr_cropped_image)

            corrimages.append(cropped_image)
            uncorrimages.append(uncorr_cropped_image)

        # primary beam weights
            wg_arr = pbarray  #
            wg_arr[np.isnan(wg_arr)] = 0  # the NaNs weight 0
            wg_arr = wg_arr ** 2 / np.nanmax(wg_arr ** 2)  # normalize
            wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
            pbweights.append(wcut.data)

        # create the wcs and footprint for output mosaic
        print('Generating primary beam corrected and uncorrected continuum mosaic.')
        wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=reference)

        array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, input_weights=pbweights)
        array2, _ = reproject_and_coadd(uncorrimages, wcs_out, shape_out=shape_out, reproject_function=reproject_interp, input_weights=pbweights)

        array = np.float32(array)
        array2 = np.float32(array2)

        # insert common PSF into the header
        psf = common_psf.to_header_keywords()
        hdr = wcs_out.to_header()
        hdr.insert('RADESYS', ('FREQ', np.nanmean(freqs)))
        hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
        hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
        hdr.insert('RADESYS', ('BPA', psf['BPA']))

        # insert units to header:
        hdr.insert('RADESYS', ('BUNIT', 'JY/BEAM'))

        pyfits.writeto(self.contmosaicdir + '/' + str(tg).upper() + '.fits', data=array, header=hdr, overwrite=True)
        pyfits.writeto(self.contmosaicdir + '/' + str(tg).upper() + '_uncorr.fits', data=array2, header=hdr, overwrite=True)

        utils.clean_contmosaic_tmp_data(self)