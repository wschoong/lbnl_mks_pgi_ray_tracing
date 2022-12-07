import tables
import numpy as np
# import time


# ===============Interpolation ================
def system_matrix_interpolate(sysmat_filename, x_dim=75):
    """Kargs: x_img_pixels, save_fname, """
    sysmat_file = load_h5file(sysmat_filename)
    sysmat = sysmat_file.root.sysmat[:]

    # save_name = sysmat_filename[:-3] + '_interp'
    interp_sysmat = interpolate_system_response(sysmat, x_img_pixels=x_dim)  # save_fname=save_name)
    # sysmat_file.close()
    return interp_sysmat, sysmat_file  # to help close if need be


def interpolate_system_response(sysmat, x_img_pixels=75):  # needed for system_matrix_interpolate
    # n_pixels, n_measurements i.e. (1875, 2304)
    # tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    tot_img_pixels, tot_det_pixels = sysmat.shape  # n_pixels, n_measurements
    y_img_pixels = tot_img_pixels // x_img_pixels

    x_interp_img_pixels = (2 * x_img_pixels-1)
    y_interp_img_pixels = (2 * y_img_pixels-1)
    interp_sysmat = np.zeros([x_interp_img_pixels * y_interp_img_pixels, tot_det_pixels], dtype=sysmat.dtype)

    for row in np.arange(y_img_pixels):  # start from top row, fill in known values and interp in-between x vals
        interp_rid = 2 * row * x_interp_img_pixels  # start
        orig_rid = row * x_img_pixels
        interp_sysmat[interp_rid:interp_rid + x_interp_img_pixels:2, :] = sysmat[orig_rid:orig_rid+x_img_pixels, :]

        interp_sysmat[(interp_rid+1):interp_rid + x_interp_img_pixels:2, :] = \
            (sysmat[orig_rid:(orig_rid + x_img_pixels-1), :] + sysmat[(orig_rid+1):orig_rid + x_img_pixels, :]) * 0.5
    # This can probably be combined with the above
    for row in np.arange(1, y_interp_img_pixels, 2):  # interp y img vals between known values
        interp_rid = row * x_interp_img_pixels
        a_rid = (row-1) * x_interp_img_pixels  # This is skipped by iteration (above rid)
        b_rid = (row+1) * x_interp_img_pixels  # (b)elow rid
        interp_sysmat[interp_rid:interp_rid+x_interp_img_pixels:2, :] = \
            (interp_sysmat[a_rid:a_rid+x_interp_img_pixels:2, :] + interp_sysmat[b_rid:b_rid+x_interp_img_pixels:2, :])\
            * 0.5

        interp_sysmat[(interp_rid + 1):interp_rid + x_interp_img_pixels:2, :] = \
            (interp_sysmat[a_rid:a_rid+x_interp_img_pixels-2:2, :] +
             interp_sysmat[(a_rid+1):a_rid + x_interp_img_pixels:2, :] +
             interp_sysmat[b_rid:b_rid + x_interp_img_pixels - 2:2, :] +
             interp_sysmat[(b_rid + 1):b_rid + x_interp_img_pixels:2, :]) * 0.25

    print("Interpolated Shape: ", interp_sysmat.shape)
    print("Nonzero values (percent): ", 1.0 * np.count_nonzero(interp_sysmat)/interp_sysmat.size)
    # np.save(save_fname, interp_sysmat)
    return interp_sysmat


# ===============Smoothing ================
def smooth_point_response(sysmat, x_img_pixels, *args, **kwargs):  # h5file = True
    # if h5file:
    #    sysmat_file = load_h5file(sysmat_filename)
    #    sysmat = sysmat_file.root.sysmat[:]
    # else:
    #    sysmat = np.load(sysmat_filename)

    size = args[0]
    try:
        fwhm = kwargs['fwhm']
    except Exception as e:
        print(e)
        print("Default FWHM used: 1")
        fwhm = 1

    print("Sysmat Shape:", sysmat.shape)
    # save_name = sysmat_filename[:sysmat_filename.find("SP")+3] + "_F" + str(fwhm) + "S" + str(size)
    # np.save(save_name, gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs))  # size, fwhm of kernel
    # if h5file:
    #     sysmat_file.close()

    fstr = str(int(fwhm)) + "_" + "{:.1f}".format(fwhm)[2:]
    return gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs), "_F" + fstr + "S" + str(size)


def gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs):  # needed for smooth_point_response
    # assumption is that sysmat shape is (n_pixels, n_measurements) i.e. (1875, 2304)
    tot_img_pixels, tot_det_pixels = sysmat.shape  # n_pixels, n_measurements

    view = sysmat.T.reshape([tot_det_pixels, tot_img_pixels // x_img_pixels,  x_img_pixels])
    # TODO: Might not need to transpose in this way
    smoothed_reponse = np.copy(view)
    print("View shape: ", view.shape)

    kern = make_gaussian(*args, **kwargs)  # size, fwhm=1
    ksize = kern.shape[0]
    # print("Kern: ", kern)
    buffer = int(np.floor(ksize/2))  # kernel is square for now

    # resmat * wgts[None,...]  where resmat is the (det_pxl, size, size) block
    for row in np.arange(buffer, (tot_img_pixels // x_img_pixels)-buffer):
        if row % 10 == 0:
            print("Row: ", row)
        upper_edge = row-buffer  # of region to multiply with kernel
        for col in np.arange(buffer, x_img_pixels-buffer):
            left_edge = col-buffer
            smoothed_reponse[:, row, col] = (view[:, upper_edge:upper_edge+ksize, left_edge:left_edge+ksize] *
                                             kern[None, ...]).sum(axis=(1, 2))
    # a1.swapaxes(0,2).swapaxes(0,1).reshape(m2.shape)
    return smoothed_reponse.transpose((1, 2, 0)).reshape(sysmat.shape)


def make_gaussian(size, fwhm=1):  # f, center=None):  # needed for gaussian_smooth_response
    """ Make a centered normalized square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    size = (np.ceil(size)//2 * 2) + 1  # rounds size to nearest odd integer
    x = np.arange(0, size, 1, float)  # size should really be an odd integer
    y = x[:, np.newaxis]

    x0 = y0 = (x[-1] + x[0])/2

    # fwhm = (4 * np.log(2)/6) * (vox**2)  # where vox is the length of a box in pixels
    gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    return gaussian/gaussian.sum()


# ===============Appending ================
def append_responses(files, save_name='appended'):  # sysmat files
    """Append responses that are adjacent in the second dimension"""
    tmp_list = list(range(len(files)))
    for fid, file in enumerate(files):
        if tables.is_hdf5_file(file):
            sysmat_file = load_h5file(file)
            tmp_list[fid] = sysmat_file.root.sysmat[:]
            sysmat_file.close()
        else:
            tmp_list[fid] = np.load(file)

        print("File {f} shape: {s}".format(f=fid, s=tmp_list[fid].shape))
    np.save(save_name, np.vstack(tmp_list))
    print("Final shape: ", np.vstack(tmp_list).shape)


def append_FoVs(files, save_name='appended', first_dim_pxls=(101, 101), after=True):  # TODO: Rewrite this
    """Append responses that are adjacent in the first dimension. After means each file is appended after previous.
    first_dim_pxls is the number of pixels in appended direction for each file"""

    tmp_list = list(range(len(files)))
    second_dim_pxls = list(range(len(files)))  # should all be the same, double check
    tot_pxls = 0
    det_pxls = 0
    meas_pxls = 0

    for fid, file in enumerate(files):
        if tables.is_hdf5_file(file):
            sysmat_file = load_h5file(file)
            arr = sysmat_file.root.sysmat[:]
            sysmat_file.close()
        else:
            arr = np.load(file)
        meas_pxls, det_pxls = arr.shape
        tot_pxls += meas_pxls
        second_dim_pxls[fid] = meas_pxls//first_dim_pxls[fid]
        print("det_pxl: ", det_pxls)
        print("second_dim_pxls: ", second_dim_pxls[fid])
        print("first dim pxls: ", first_dim_pxls[fid])
        tmp_list[fid] = arr.T.reshape([det_pxls, second_dim_pxls[fid], first_dim_pxls[fid]])

        print("File {f} shape: {s}".format(f=fid, s=tmp_list[fid].shape))

    assert np.all(np.array(second_dim_pxls) == second_dim_pxls[0]), "Files don't have same shape in second dimension"
    if after:
        tot_arr = np.concatenate(tmp_list, axis=2)
    else:
        tot_arr = np.concatenate(tmp_list[::-1], axis=2)

    reshaped_arr = tot_arr.transpose((1, 2, 0)).reshape([tot_pxls, det_pxls])
    np.save(save_name, reshaped_arr)
    print("Final shape: ", reshaped_arr.shape)  # TODO: Test this with table measurements


# ===============Other ================
def load_h5file(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def mask_edge_pixels(mods=None):
    """The purpose of this function is to mask edge pixels and to provide a similar mask for system response i.e.
     cut out edge pixels from projection AND system matrix"""
    if mods is None:
        mods = np.arange(16)

    mod_mask = []
    for i in np.arange(16):
        if np.isin(i, mods):
            mask = np.zeros([12, 12])
            mask[1:-1, 1:-1] = 1
        else:
            mask = np.ones([12, 12])
        mod_mask.append(mask)
    proj_mask = np.block([mod_mask[col:col + 4] for col in np.arange(0, len(mod_mask), 4)])
    sysmat_mask = proj_mask.ravel()
    return proj_mask, sysmat_mask


def sysmat_processing(files, npix, *args, interp=True, smooth=True, fname='processed', **kwargs):
    """*args = sze of gaussian filter and **kwargs is fhwm of filter. Files is list of sysmat_files and npix is list
    of dimensions of the spaces"""

    if isinstance(files, str):
        files = [files]

    if len(files) > np.array(npix).size//2:
        print("Checked")
        npix = [npix] * len(files)

    if len(files) == 1:
        npix = [npix]

    print("npix length: ", len(npix))
    print("npix: ", npix)
    print("first entry: ", npix[0])
    store_list = []
    append_str = ''

    for fid, file in enumerate(files):
        npix_x, npix_y = npix[fid]
        if interp:
            sysmat, sysmat_file = system_matrix_interpolate(file, x_dim=npix_x)
            # print("Sysmat shape now:", sysmat.shape)
            npix_x = npix_x + (npix_x - 1)
            npix_y = npix_y + (npix_y - 1)
            sysmat_file.close()
        else:
            if tables.is_hdf5_file(file):
                sysmat_file = load_h5file(file)
                sysmat = sysmat_file.root.sysmat[:]
                sysmat_file.close()  # TODO: Will this cause problems? Copy otherwise
            else:
                sysmat = np.load(file)  # .npy
        print("Interpolation successful!")
        print("sysmat.shape: ", sysmat.shape)
        if smooth:
            sysmat, append_str = smooth_point_response(sysmat, npix_x, *args, **kwargs)

        store_list.append(sysmat)

    fname += append_str
    # if len(store_list) == 1:
    #   processed_array = store_list[0]
    # else:
    processed_array = np.vstack(store_list)
    print("Final shape: ", processed_array.shape)
    np.save(fname, processed_array)

    # TODO: np.savez of system matrix, npix, and total_shape and integrate into recon function workflow
    # smooth_point_response("/home/justin/repos/sysmat/design/2021-03-30-2347_SP0_interp.npy", 201, 7,
    #                       h5file=False, fwhm=2.355 * 1)  # 2.355 * spread defined in gaussian function (uncertainty)


def main():
    # files = ['/home/justin/repos/sysmat/design/2021-04-01-2021_SP0.h5',  # 120 cm from source to collimator
    #          '/home/justin/repos/sysmat/design/2021-04-02-0012_SP0.h5',  # 110 cm
    #          '/home/justin/repos/sysmat/design/2021-04-02-0308_SP0.h5',  # 100
    #          '/home/justin/repos/sysmat/design/2021-04-02-1407_SP0.h5']  # 90
    # npix = np.array([37, 31])  # 3D
    # append_responses(files, save_name='fov3d')

    # files = ['/home/justin/repos/sysmat/design/2021-04-02-1407_SP0.h5']
    # files = ['/home/justin/repos/sysmat/design/2021-04-03-0520_SP0.h5',
    #          '/home/justin/repos/sysmat/design/2021-04-03-1627_SP0.h5']
    # npix = np.array([121, 31]) # 100 mm FoV (not complete)

    # files = ['/home/justin/repos/sysmat/design/2021-03-31-1633_SP0.h5',
    #         '/home/justin/repos/sysmat/design/2021-03-30-2207_SP0.h5']
    # first_dpxls = np.array([19, 21])  # Table
    # append_FoVs(files, first_dim_pxls=first_dpxls, save_name='table_appended')

    # sysmat_processing(files, npix, 7, fwhm=2.355 * 1, fname='100mm_fuller_FoV_processed')
    # Note: This works but forget set npix in recon. FWHM of 2.355 is DOUBLE COUNTING

    # base_folder = '/home/justin/Desktop/system_responses/'
    # sysmat_files = ['obj/100mm_fuller_FoV_processed_F1S7.npy', 'table/table_appended.npy']
    # files = [base_folder + file for file in sysmat_files]
    # append_responses(files, save_name='obj_table')

    # files = '/home/justin/repos/sysmat/design/2021-04-12-1758_SP0.h5'
    # npix = np.array([121, 31])
    # sysmat_processing(files, npix, smooth=False, fname='120mm_wide_FoV_processed_no_smooth')
    # RMS fwhm = 1/np.sqrt(12) * 2.355 = 0.6798
    # sysmat_processing(files, npix, 7, fwhm=0.7, fname='120mm_wide_FoV_processedRMS')

    # files =['/home/justin/repos/sysmat/design/2021-04-07-1433_SP0.h5',  # 130 mm
    #        '/home/justin/repos/sysmat/design/2021-04-12-1758_SP0.h5',  # 120
    #        '/home/justin/repos/sysmat/design/2021-04-05-2233_SP0.h5',  # 110
    #        '/home/justin/repos/sysmat/design/2021-04-03-0520_SP0.h5',  # 100 (care there are 2)
    #        '/home/justin/repos/sysmat/design/2021-04-06-1334_SP0.h5']  # 90
    # npix = np.array([121, 31])
    # sysmat_processing(files, npix, 7, fwhm=0.7, fname='Apr14_3d_wide')

    # files = ['/home/justin/Desktop/system_responses/Thesis/2021-03-27-1529_SP0.h5']
    # npix = np.array([101, 101])
    # sysmat_processing(files, npix, 7, fwhm=2, fname='Apr14_full')

    # files = ['/home/justin/repos/sysmat/design/2021-04-19-2004_SP1.h5']
    # npix = np.array([101, 31])
    # sysmat_processing(files, npix, 7, fwhm=0.7, fname='Apr20_FoV')

    # files = ['/home/justin/repos/sysmat/design/Apr20_FoV_F0_7S7.npy',
    #         '/home/justin/repos/sysmat/design/2021-04-16-1340_SP0.h5']
    # append_responses(files, save_name="Apr20_FoV_beamstop")

    # response = ['/home/justin/repos/sysmat/design/2021-04-19-2004_SP1.h5']
    # npix = np.array([101, 31])
    # sysmat_processing(response, npix, 7, fwhm=0.7, fname='Apr28_FoV')

    files = ['/home/justin/repos/sysmat/design/Apr28_FoV_F0_7S7.npy',
             '/home/justin/repos/sysmat/design/2021-04-23-1259_SP1.h5']
    append_responses(files, save_name="Apr28_FoV_beamstop")


def main2():

    # Start Table appends
    # table_files = ['/home/justin/repos/sysmat/design/system_responses/2021-03-31-1633_SP0.h5',  # -x (beamport)
    #                '/home/justin/repos/sysmat/design/system_responses/2021-03-30-2207_SP0.h5',  # center
    #               '/home/justin/repos/sysmat/design/2021-05-31-2209_SP1.h5']  # +x (beamstop)
    # first_dpxls = np.array([19, 21, 19])  # Table, all z dimensions are 23
    # append_FoVs(table_files, first_dim_pxls=first_dpxls, save_name='june1_table')
    # End Table appends

    region_files = ['/home/justin/repos/sysmat_current/sysmat/design/2021-06-11-1110_SP1.h5',  # FOV
                    # '/home/justin/repos/sysmat/design/2021-04-19-2004_SP1.h5',  # Old FOV
                    '/home/justin/repos/sysmat/design/2021-05-08-2118_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat/design/2021-05-08-1620_SP1.h5',  # Bot FOV
                    '/home/justin/repos/sysmat/design/june1_table.npy',  # Wide table
                    '/home/justin/repos/sysmat/design/2021-05-09-1531_SP1.h5',  # Beam Port
                    '/home/justin/repos/sysmat/design/2021-04-23-1259_SP1.h5']  # Beam Stop

    # append_responses(region_files, save_name="june1_full_response")
    append_responses(region_files, save_name="june15_full_response")


def main3():  # Unfolded response June 22
    region_files = ['/home/justin/repos/sysmat_current/sysmat/design/2021-06-17-1746_SP1.h5',  # FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-1454_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-0957_SP1.h5',  # Bot FOV
                    # '/home/justin/repos/sysmat/design/june1_table.npy',  # Wide table, redo
                    # '/home/justin/repos/sysmat/design/2021-05-09-1531_SP1.h5',  # Beam Port, redo
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-22-0959_SP1.h5']  # Beam Stop

    append_responses(region_files, save_name="june22_full_response")


def main4():  # Folded Response June 30
    region_files = ['/home/justin/repos/sysmat_current/sysmat/design/June30_folded_g1.npy',  # FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-1454_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-0957_SP1.h5',  # Bot FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-22-0959_SP1.h5']  # Beam Stop
    append_responses(region_files, save_name="june30_full_response")


def main5():  # Unfolded Response July 6 (for July 3rd sysmat), correct detector axes
    # TODO: calculate sysmat for other regions. Only FOV done
    region_files = [  # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-03-1015_SP1.h5',  # FOV, unfolded
                    '/home/justin/repos/sysmat_current/sysmat/design/July6_folded_g4.npy',  # FOV, folded
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-1454_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-21-0957_SP1.h5',  # Bot FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-06-22-0959_SP1.h5']  # Beam Stop
    # append_responses(region_files, save_name="july6_full_response")  # unfolded
    append_responses(region_files, save_name="july6_full_response_folded")  # folded


def main6():  # Responses July 20
    region_files = [  # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-03-1015_SP1.h5',  # FOV, unfolded
                    '/home/justin/Desktop/july20/FoV_weights/e_avgC_noSin.h5',  # FOV, e_evg C
                    # '/home/justin/Desktop/july20/FoV_weights/p_avgO_noSin.h5',  # FOV, p_avg C
                    # '/home/justin/repos/sysmat_current/sysmat/design/July6_folded_g4.npy',  # FOV, folded
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-11-0905_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-10-1058_SP1.h5',  # Bot FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-16-1946_SP1.h5']  # Beam Stop
    #  '/home/justin/repos/sysmat_current/sysmat/design/2021-07-17-1727_SP1.h5'  # Beam Port
    append_responses(region_files, save_name="july20_full_response_eavg")  # folded


def main_seng(): # Full FoV, 110 cm source to plane
    region_files = ['/home/justin/repos/sysmat_current/sysmat/design/2021-08-10-2221_SP1.h5',
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-08-05-1116_SP1.h5',
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-08-10-0505_SP1.h5']
    append_responses(region_files, save_name="sysmat_110mm_201x199_sp1_seng")


if __name__ == "__main__":
    # main3()
    # main4()
    # main5()
    # main6()
    main_seng()
