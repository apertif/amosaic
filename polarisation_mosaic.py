import os

import numpy as np
from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs

import fits_magic as fm
import utils
import glob
import sys


class polarisation_mosaic:
    """
    Class to produce polarisation mosaics in Stokes Q, U and V.
    """
    module_name = 'POLARISATION MOSAIC'


    def __init__(self, file_=None, **kwargs):
        self.default = utils.load_config(self, file_)
        utils.set_mosdirs(self)
        self.config_file_name = file_


    def go(self):
        """
        Function to generate the polarisation mosaics
        """
        utils.gen_poldirs(self)
        utils.collect_paramfiles(self)
        veri = self.check_polimages()
        utils.copy_polimages(self, veri)
        utils.copy_polbeams(self)
        cbeam = utils.get_common_psf(self, veri, format='array')
        for sb in range(24):
            qimages, uimages, pbimages = utils.get_polfiles(self, sb)
            if len(qimages) != 0:
                self.make_polmosaic(qimages, uimages, pbimages, sb, cbeam, pbclip=self.pol_pbclip)
            else:
                print('No data for subband ' + str(sb).zfill(2))
        self.make_polcubes()


    def check_polimages(self):
        """
        Sort out any beams or planes, which are useless for the imaging
        """
        # Collect the beam and noise parameters from the main parameter file
        rms_array = np.full((40, 24, 2), np.nan)
        bmaj_array = np.full((40, 24, 2), np.nan)
        bmin_array = np.full((40, 24, 2), np.nan)
        bpa_array = np.full((40, 24, 2), np.nan)
        for beam in range(0, 40, 1):
            try:
                rms_array[beam, :] = utils.get_param(self, 'polarisation_B' + str(beam).zfill(2) + '_targetbeams_qu_imagestats')[:, 2, :]
                bmaj_array[beam, :] = utils.get_param(self, 'polarisation_B' + str(beam).zfill(2) + '_targetbeams_qu_beamparams')[:, 0, :]
                bmin_array[beam, :] = utils.get_param(self, 'polarisation_B' + str(beam).zfill(2) + '_targetbeams_qu_beamparams')[:, 1, :]
                bpa_array[beam, :] = utils.get_param(self, 'polarisation_B' + str(beam).zfill(2) + '_targetbeams_qu_beamparams')[:, 2, :]
            except KeyError:
                print('Synthesised beam parameters and/or noise statistics of beam ' + str(beam).zfill(2) + ' are not available. Excluding beam!')
        np.savetxt(self.polmosaicdir + '/Qrms.npy', rms_array[:,:,0])
        np.savetxt(self.polmosaicdir + '/Qbmaj.npy', bmaj_array[:,:,0])
        np.savetxt(self.polmosaicdir + '/Qbmin.npy', bmin_array[:,:,0])
        np.savetxt(self.polmosaicdir + '/Qbpa.npy', bpa_array[:,:,0])
        np.savetxt(self.polmosaicdir + '/Urms.npy', rms_array[:, :, 1])
        np.savetxt(self.polmosaicdir + '/Ubmaj.npy', bmaj_array[:,:,1])
        np.savetxt(self.polmosaicdir + '/Ubmin.npy', bmin_array[:,:,1])
        np.savetxt(self.polmosaicdir + '/Ubpa.npy', bpa_array[:,:,1])
        # Create an array for the accepted beams
        accept_array = np.full((40, 24), True)
        # Iterate through the rms and beam sizes of all cubes and filter the images
        for b in range(40):
            for sb in range(24):
                if rms_array[b, sb, 0] > self.pol_rmsclip or np.isnan(rms_array[b, sb, 0]):
                    accept_array[b, sb] = False
                else:
                    continue

            for sb in range(24):
                if rms_array[b, sb, 1] > self.pol_rmsclip or np.isnan(rms_array[b, sb, 1]):
                    accept_array[b, sb] = False
                else:
                    continue

            for sb in range(24):
                if bmin_array[b, sb, 0] > self.pol_bmin or bmin_array[b, sb, 1] > self.pol_bmin:
                    accept_array[b, sb] = False
                else:
                    continue

            for sb in range(24):
                if bmaj_array[b, sb, 0] > self.pol_bmaj or bmaj_array[b, sb, 1] > self.pol_bmaj:
                    accept_array[b, sb] = False
                else:
                    continue
        np.savetxt(self.polmosaicdir + '/accept_array.npy', accept_array)
        # Generate the main array for accepting the beams
        bacc_array = np.full(40, True, dtype=bool)
        badim_array = np.zeros((40))
        # Count number of False for each beam and filter all beams out where more than x planes or more are bad
        for b in range(40):
            badim_array[b] = len(np.where(accept_array[b, :] == False)[0])
            if badim_array[b] > self.pol_badim:
                bacc_array[b] = False
                accept_array[b, :] = False
            else:
                continue
        np.savetxt(self.polmosaicdir + '/badim.npy', badim_array)
        np.savetxt(self.polmosaicdir + '/bacc.npy', bacc_array)
        # Generate the array for accepting the subbands
        sb_acc = np.full(24, True, dtype=bool)
        for sb in range(24):
            if np.sum(accept_array[:, sb]) < np.sum(bacc_array):
                sb_acc[sb] = False
        np.savetxt(self.polmosaicdir + '/sbacc.npy', sb_acc)
        final_acc_arr = np.full((40, 24), True)
        for b in range(40):
            for sb in range(24):
                if bacc_array[b] and sb_acc[sb]:
                    final_acc_arr[b,sb] = True
                else:
                    final_acc_arr[b,sb] = False
        np.savetxt(self.polmosaicdir + '/final_accept.npy', final_acc_arr)
        return final_acc_arr


    def make_polmosaic(self, qimages, uimages, pbimages, sb, psf, reference=None, pbclip=None):
        """
        Function to generate the polarisation mosaic in Q and U
        """

        # Set the directories for the mosaicking
        utils.set_mosdirs(self)
        # Get the common psf
        common_psf = psf

        qcorrimages = [] # to mosaic
        ucorrimages = []  # to mosaic
        qpbweights = [] # of the pixels
        upbweights = []  # of the pixels
        qrmsweights = [] # of the images themself
        urmsweights = []  # of the images themself
        qfreqs = []
        ufreqs = []
        # weight_images = []
        for qimg, uimg, pb in zip(qimages, uimages, pbimages):
            # prepare the images (squeeze, transfer_coordinates, reproject, regrid pbeam, correct...)
            with pyfits.open(qimg) as f:
                qimheader = f[0].header
                qfreqs.append(qimheader['CRVAl3'])
                qtg = qimheader['OBJECT']
            with pyfits.open(uimg) as f:
                uimheader = f[0].header
                ufreqs.append(uimheader['CRVAl3'])
                utg = uimheader['OBJECT']
            qimg = fm.fits_squeeze(qimg) # remove extra dimentions
            uimg = fm.fits_squeeze(uimg)  # remove extra dimentions
            pb = fm.fits_transfer_coordinates(qimg, pb) # transfer_coordinates
            pb = fm.fits_squeeze(pb) # remove extra dimensions
            with pyfits.open(qimg) as f:
                qimheader = f[0].header
                qimdata = f[0].data
            with pyfits.open(uimg) as f:
                uimheader = f[0].header
                uimdata = f[0].data
            with pyfits.open(pb) as f:
                pbhdu = f[0]
                autoclip = np.nanmin(f[0].data)
        # reproject
                qreproj_arr, qreproj_footprint = reproject_interp(pbhdu, qimheader)
                ureproj_arr, ureproj_footprint = reproject_interp(pbhdu, uimheader)

            pbclip = self.pol_pbclip or autoclip
            print('PB is clipped at %f level', pbclip)
            qreproj_arr = np.float32(qreproj_arr)
            ureproj_arr = np.float32(ureproj_arr)
            qreproj_arr[qreproj_arr < pbclip] = np.nan
            ureproj_arr[ureproj_arr < pbclip] = np.nan
            qpb_regr_repr = pb.replace('.fits', '_repr.fits')
            upb_regr_repr = pb.replace('.fits', '_repr.fits')
            pyfits.writeto(qpb_regr_repr, qreproj_arr, qimheader, overwrite=True)
            pyfits.writeto(upb_regr_repr, ureproj_arr, uimheader, overwrite=True)
        # convolution with common psf
            qreconvolved_image = qimg.replace('.fits', '_reconv.fits')
            qreconvolved_image = fm.fits_reconvolve_psf(qimg, common_psf, out=qreconvolved_image)
            ureconvolved_image = uimg.replace('.fits', '_reconv.fits')
            ureconvolved_image = fm.fits_reconvolve_psf(uimg, common_psf, out=ureconvolved_image)
        # PB correction
            qpbcorr_image = qreconvolved_image.replace('_reconv.fits', '_pbcorr.fits')
            qpbcorr_image = fm.fits_operation(qreconvolved_image, qreproj_arr, operation='/', out=qpbcorr_image)
            upbcorr_image = ureconvolved_image.replace('_reconv.fits', '_pbcorr.fits')
            upbcorr_image = fm.fits_operation(ureconvolved_image, ureproj_arr, operation='/', out=upbcorr_image)
        # cropping
            qcropped_image = qimg.replace('.fits', '_mos.fits')
            qcropped_image, qcutout = fm.fits_crop(qpbcorr_image, out=qcropped_image)
            qcorrimages.append(qcropped_image)
            ucropped_image = uimg.replace('.fits', '_mos.fits')
            ucropped_image, ucutout = fm.fits_crop(upbcorr_image, out=ucropped_image)
            ucorrimages.append(ucropped_image)

        # primary beam weights
            qwg_arr = qreproj_arr - pbclip # the edges weight ~0
            qwg_arr[np.isnan(qwg_arr)] = 0 # the NaNs weight 0
            qwg_arr = qwg_arr / np.nanmax(qwg_arr) # normalize
            qwcut = Cutout2D(qwg_arr, qcutout.input_position_original, qcutout.shape)
            qpbweights.append(qwcut.data)
            uwg_arr = ureproj_arr - pbclip # the edges weight ~0
            uwg_arr[np.isnan(uwg_arr)] = 0 # the NaNs weight 0
            uwg_arr = uwg_arr / np.nanmax(uwg_arr) # normalize
            uwcut = Cutout2D(uwg_arr, ucutout.input_position_original, ucutout.shape)
            upbweights.append(uwcut.data)

        # weight the images by RMS noise over the edges
            ql, qm = qimdata.shape[0]//10,  qimdata.shape[1]//10
            qmask = np.ones(qimdata.shape, dtype=np.bool)
            qmask[ql:-ql,qm:-qm] = False
            qimg_noise = np.nanstd(qimdata[qmask])
            qimg_weight = 1 / qimg_noise**2
            qrmsweights.append(qimg_weight)
            ul, um = uimdata.shape[0]//10,  uimdata.shape[1]//10
            umask = np.ones(uimdata.shape, dtype=np.bool)
            umask[ul:-ul,um:-um] = False
            uimg_noise = np.nanstd(uimdata[umask])
            uimg_weight = 1 / uimg_noise**2
            urmsweights.append(uimg_weight)

        # merge the image rms weights and the primary beam pixel weights:
        qweights = [qp*qr/max(qrmsweights) for qp, qr in zip(qpbweights, qrmsweights)]
        uweights = [up * ur / max(urmsweights) for up, ur in zip(upbweights, urmsweights)]

        # create the wcs and footprint for the output mosaic
        qwcs_out, qshape_out = find_optimal_celestial_wcs(qcorrimages, auto_rotate=False, reference=reference)
        uwcs_out, ushape_out = find_optimal_celestial_wcs(ucorrimages, auto_rotate=False, reference=reference)

        qarray, qfootprint = reproject_and_coadd(qcorrimages, qwcs_out, shape_out=qshape_out,
                                                reproject_function=reproject_interp,
                                                input_weights=qweights)
        uarray, ufootprint = reproject_and_coadd(ucorrimages, uwcs_out, shape_out=ushape_out,
                                                reproject_function=reproject_interp,
                                                input_weights=uweights)
        qarray = np.float32(qarray)
        uarray = np.float32(uarray)

        # insert common PSF into the header
        qpsf = common_psf.to_header_keywords()
        qhdr = qwcs_out.to_header()
        qhdr.insert('RADESYS', ('FREQ', np.nanmean(qfreqs)))
        qhdr.insert('RADESYS', ('BMAJ', qpsf['BMAJ']))
        qhdr.insert('RADESYS', ('BMIN', qpsf['BMIN']))
        qhdr.insert('RADESYS', ('BPA', qpsf['BPA']))

        upsf = common_psf.to_header_keywords()
        uhdr = qwcs_out.to_header()
        uhdr.insert('RADESYS', ('FREQ', np.nanmean(ufreqs)))
        uhdr.insert('RADESYS', ('BMAJ', upsf['BMAJ']))
        uhdr.insert('RADESYS', ('BMIN', upsf['BMIN']))
        uhdr.insert('RADESYS', ('BPA', upsf['BPA']))

        pyfits.writeto(self.polmosaicdir + '/' + str(qtg).upper() + '_' + str(sb).zfill(2) + '_Q.fits', data=qarray,
                     header=qhdr, overwrite=True)
        pyfits.writeto(self.polmosaicdir + '/' + str(utg).upper() + '_' + str(sb).zfill(2) + '_U.fits', data=uarray,
                     header=uhdr, overwrite=True)

        utils.clean_polmosaic_tmp_data(self, sb)


    def make_polcubes(self):
        """
        Function to generate the cubes in Q and U from the polarisation mosaics
        """

        # Set the directories for the mosaicking
        utils.set_mosdirs(self)

        # Get the fits files
        Qmoss = sorted(glob.glob(self.polmosaicdir + '/*_[0-9][0-9]_Q.fits'))
        Umoss = sorted(glob.glob(self.polmosaicdir + '/*_[0-9][0-9]_U.fits'))

        # Check if the same number of mosaics is available for Q and U
        Qmoss_chk = []
        Umoss_chk = []
        for qchk in Qmoss:
            qnew = qchk.replace('_Q','')
            Qmoss_chk.append(qnew)
        for uchk in Umoss:
            unew = uchk.replace('_U','')
            Umoss_chk.append(unew)
        if Qmoss_chk == Umoss_chk:
            pass
        else:
            print('Different number of Q and U mosaics. Cannot generate cubes!')
            sys.exit()

        # Find the smallest mosaic image
        allim = sorted(Qmoss + Umoss)
        naxis1 = []
        naxis2 = []
        for mos in allim:
            with pyfits.open(mos) as m:
                imheader = m[0].header
                naxis1.append(imheader['NAXIS1'])
                naxis2.append(imheader['NAXIS2'])
        impix = np.array(naxis1) * np.array(naxis2)
        smim = allim[np.argmin(impix)]

        # Reproject the rest of the images to this one
        # Load the header of the reference image
        hduref = pyfits.open(smim)[0]
        hduref_hdr = hduref.header
        # Reproject the other images to the reference
        for image in allim:
            hdu = pyfits.open(image)[0]
            hdu_hdr = hdu.header
            hduref_hdr['FREQ'] = hdu_hdr['FREQ']
            repr_image = reproject_interp(hdu, hduref_hdr, return_footprint=False)
            pyfits.writeto(image.replace('.fits','_repr.fits'), repr_image, hduref_hdr, overwrite=True)

        # Generate a mask to limit the valid area for all images to the largest common valid one
        allreprims = sorted(glob.glob(self.polmosaicdir + '/*_repr.fits'))
        nall = len(allreprims)
        # Generate an array for all the images
        alldata = np.full((nall, hduref_hdr['NAXIS2'], hduref_hdr['NAXIS1']), np.nan)
        for i, image in enumerate(allreprims):
            hdu = pyfits.open(image)[0]
            hdu_data = hdu.data
            alldata[i,:,:] = hdu_data
        # Generate the mask
        immask = np.sum(alldata, axis=0)
        immask[np.isfinite(immask)] = 1.0
        # Apply the mask
        for m, mimage in enumerate(allreprims):
            mhdu = pyfits.open(mimage)[0]
            mhdu_data = mhdu.data
            mhdu_hdr = mhdu.header
            mdata = mhdu_data*immask
            pyfits.writeto(mimage.replace('_repr.fits','_mask.fits'), mdata, mhdu_hdr, overwrite=True)

        # Finally create the frequency image cubes
        qfinimages = sorted(glob.glob(self.polmosaicdir + '/*Q_mask.fits'))
        ufinimages = sorted(glob.glob(self.polmosaicdir + '/*U_mask.fits'))
        nq = len(qfinimages)
        nu = len(ufinimages)
        qdata = np.full((nq, hduref_hdr['NAXIS2'], hduref_hdr['NAXIS1']), np.nan)
        udata = np.full((nu, hduref_hdr['NAXIS2'], hduref_hdr['NAXIS1']), np.nan)
        freqs = []
        # Generate the Q cube
        for q, qim in enumerate(qfinimages):
            qhdu = pyfits.open(qim)[0]
            qhdu_data = qhdu.data
            qhdu_hdr = qhdu.header
            freqs.append(qhdu_hdr['FREQ'])
            qdata[q,:,:] = qhdu_data
        qhdu_hdr.insert('NAXIS2', ('NAXIS3', len(qfinimages)), after=True)
        qhdu_hdr.insert('CTYPE2', ('CRPIX3', 1.0), after=True)
        qhdu_hdr.insert('CRPIX3', ('CDELT3', 6250000.0), after=True)
        qhdu_hdr.insert('CDELT3', ('CRVAL3', freqs[0]), after=True)
        qhdu_hdr.insert('CRVAL3', ('CTYPE3', 'FREQ-OBS'), after=True)
        pyfits.writeto(self.polmosaicdir + '/Qcube.fits', np.float32(qdata), qhdu_hdr, overwrite=True)
        # Write the frequency file
        with open(self.polmosaicdir + '/freq.txt', 'w') as f:
            for item in freqs:
                f.write("%s\n" % item)
        # Generate the U cube
        for u, uim in enumerate(ufinimages):
            uhdu = pyfits.open(uim)[0]
            uhdu_data = uhdu.data
            uhdu_hdr = uhdu.header
            udata[u,:,:] = uhdu_data
        uhdu_hdr.insert('NAXIS2', ('NAXIS3', len(ufinimages)), after=True)
        uhdu_hdr.insert('CTYPE2', ('CRPIX3', 1.0), after=True)
        uhdu_hdr.insert('CRPIX3', ('CDELT3', 6250000.0), after=True)
        uhdu_hdr.insert('CDELT3', ('CRVAL3', freqs[0]), after=True)
        uhdu_hdr.insert('CRVAL3', ('CTYPE3', 'FREQ-OBS'), after=True)
        pyfits.writeto(self.polmosaicdir + '/Ucube.fits', np.float32(udata), uhdu_hdr, overwrite=True)

        # Write a file with the central coordinates of each pointing used
        coord_arr = np.full((40,3), np.nan)
        coord_arr[:,0] = np.arange(0,40,1)
        for b in range(40):
            if os.path.isfile(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/Qcube.fits')):
                qcube = pyfits.open(os.path.join(self.basedir, self.obsid, str(b).zfill(2), 'polarisation/Qcube.fits'))[0]
                qcube_hdr = qcube.header
                coord_arr[b,1] = qcube_hdr['CRVAL1']
                coord_arr[b,2] = qcube_hdr['CRVAL2']
        np.savetxt(self.polmosaicdir + '/pointings.txt', coord_arr, fmt=['%2s','%1.13e','%1.13e'], delimiter='\t')

        # Remove obsolete files
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_Q_mask.fits')
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_U_mask.fits')
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_Q_repr.fits')
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_U_repr.fits')
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_Q.fits')
        os.system('rm -rf ' + self.polmosaicdir + '*_[0-9][0-9]_U.fits')