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


    def make_contmosaic(self, images, pbimages, reference=None, pbclip=None):
        """
        Function to generate the continuum mosaic
        """

        # Get the common psf
        common_psf = utils.get_common_psf(self, images)

        corrimages = [] # to mosaic
        pbweights = [] # of the pixels
        rmsweights = [] # of the images themself
        freqs = []
        # weight_images = []
        for img, pb in zip(images, pbimages):
            # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
            with pyfits.open(img) as f:
                imheader = f[0].header
                freqs.append(imheader['CRVAl3'])
                tg = imheader['OBJECT']
            img = fm.fits_squeeze(img) # remove extra dimentions
            pb = fm.fits_transfer_coordinates(img, pb) # transfer_coordinates
            pb = fm.fits_squeeze(pb) # remove extra dimensions
            with pyfits.open(img) as f:
                imheader = f[0].header
                imdata = f[0].data
            with pyfits.open(pb) as f:
                pbhdu = f[0]
                autoclip = np.nanmin(f[0].data)
        # reproject
                reproj_arr, reproj_footprint = reproject_interp(pbhdu, imheader)

            pbclip = self.cont_pbclip or autoclip
            print('PB is clipped at %f level', pbclip)
            reproj_arr = np.float32(reproj_arr)
            reproj_arr[reproj_arr < pbclip] = np.nan
            pb_regr_repr = os.path.basename(pb.replace('.fits', '_repr.fits'))
            pyfits.writeto(pb_regr_repr, reproj_arr, imheader, overwrite=True)
        # convolution with common psf
            reconvolved_image = os.path.basename(img.replace('.fits', '_reconv.fits'))
            reconvolved_image = fm.fits_reconvolve_psf(img, common_psf, out=reconvolved_image)
        # PB correction
            pbcorr_image = os.path.basename(reconvolved_image.replace('.fits', '_pbcorr.fits'))
            pbcorr_image = fm.fits_operation(reconvolved_image, reproj_arr, operation='/', out=pbcorr_image)
        # cropping
            cropped_image = os.path.basename(img.replace('.fits', '_mos.fits'))
            cropped_image, cutout = fm.fits_crop(pbcorr_image, out=cropped_image)
            corrimages.append(cropped_image)

        # primary beam weights
            wg_arr = reproj_arr - pbclip # the edges weight ~0
            wg_arr[np.isnan(wg_arr)] = 0 # the NaNs weight 0
            wg_arr = wg_arr / np.nanmax(wg_arr) # normalize
            wcut = Cutout2D(wg_arr, cutout.input_position_original, cutout.shape)
            pbweights.append(wcut.data)

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
        wcs_out, shape_out = find_optimal_celestial_wcs(corrimages, auto_rotate=False, reference=reference)

        array, footprint = reproject_and_coadd(corrimages, wcs_out, shape_out=shape_out,
                                                reproject_function=reproject_interp,
                                                input_weights=weights)
        array = np.float32(array)

        # insert common PSF into the header
        psf = common_psf.to_header_keywords()
        hdr = wcs_out.to_header()
        hdr.insert('RADESYS', ('FREQ', np.nanmean(freqs)))
        hdr.insert('RADESYS', ('BMAJ', psf['BMAJ']))
        hdr.insert('RADESYS', ('BMIN', psf['BMIN']))
        hdr.insert('RADESYS', ('BPA', psf['BPA']))

        pyfits.writeto(self.contmosaicdir + '/' + str(tg).upper() + '.fits', data=array,
                     header=hdr, overwrite=True)

        utils.clean_contmosaic_tmp_data(self)