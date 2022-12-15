import matplotlib.pyplot as plt
import time
import numpy as np
import tables
import os
from datetime import datetime
from apertures import Collimator
from sources import Sources
from detector_system import Detector_System
from utils import generate_detector_centers_and_norms


class Imager(object):
    def __init__(self, system_axis=np.array([0, 0, 1])):
        self.system_central_axis = system_axis  # Global axis of system
        self.collimator = Collimator(self)

        self.detector_system = Detector_System(self)

        self.sources = Sources()
        self.subsample = 1  # I need to go through and change all of these
        self.sample_area = 0
        self.sample_step = 0.1  # in cm

        # self.test_generate_ray = 1
        # self.test_prf = 1
        self._ray_det_enter = 0

    def set_collimator(self, **kwargs):
        self.collimator = Collimator(self, **kwargs)

    def create_detector(self, **kwargs):
        self.detector_system.add_detector(**kwargs)
        if not len(self.detector_system.detectors) == 1:
            self.sample_area = self.detector_system.detectors[0].pix_size ** 2

    def set_sources(self, **kwargs):
        self.sources = Sources(**kwargs)

    def generate_ray(self, src_pt, det_pt):  # generates the ray once
        em_dir = norm(det_pt - src_pt)

        dir_col = np.sign(self.collimator.colp[2] - src_pt[2])  # in z-hat, pointing to detector
        min_sc_coll = ((self.collimator.colp[2] - (dir_col * self.collimator.col_half_thickness)) - src_pt[2]) \
                      / em_dir[2] - self.sample_step
        max_sc_coll = ((self.collimator.colp[2] + (dir_col * self.collimator.col_half_thickness)) - src_pt[2]) \
                      / em_dir[2] + self.sample_step

        min_sc_det = (self.detector_system.closest_plane - src_pt[2]) / em_dir[2]
        self._ray_det_enter = min_sc_det//self.sample_step  # in sample steps

        max_sc_det = (self.detector_system.farthest_plane - src_pt[2]) / em_dir[2]  # This is a max length in mm

        # if self.test_generate_ray:
        # ray_coll = src_pt + em_dir * np.arange(min_sc_coll, max_sc_coll
        #                                       + self.sample_step, self.sample_step)[:, np.newaxis]
        # ray_det = src_pt + em_dir * np.arange(min_sc_det, max_sc_det +
        #                                      self.sample_step, self.sample_step)[:, np.newaxis]

        #    print("em_dir: ", em_dir)
        #    print("rays in collimator first {f} and last {l}: f , l".format(f=ray_coll[0], l=ray_coll[-1]))
        #    print("rays in det first {fd} and last {ld}:, fd, ld".format(fd=ray_det[0], ld=ray_det[-1]))
        #    self.test_generate_ray = 0

        return src_pt + em_dir * np.arange(min_sc_coll, max_sc_coll + self.sample_step, self.sample_step)[:, np.newaxis], \
               src_pt + em_dir * np.arange(min_sc_det, max_sc_det + self.sample_step, self.sample_step)[:, np.newaxis], \
               em_dir  # rays_in_collimator, rays_in_detector, em_dir

    def generate_test_response(self, tst_pt, **kwargs):
        try:
            self.subsample = kwargs['subsample']
            print("Found subsample!")
        except Exception as e:
            print(e)
            print("Going with subsample = 1")
            self.subsample = 1

        self.detector_system.initialize_arrays()

        return self._point_response_function(tst_pt, **kwargs)

    def generate_sysmat_response(self, **kwargs):  # subsample = 1, mid = True

        try:
            self.subsample = kwargs['subsample']
        except Exception as e:
            print(e)
            print("Going with subsample = 1")
            self.subsample = 1
        print("Subsample: ", self.subsample)

        self.detector_system.initialize_arrays()
        tot_img_pxls = self.detector_system.projection.size  # This ensures it is first initialized
        print("Total Image Pixels: ", self.detector_system.projection.size)

        current_datetime = datetime.now().strftime("%Y-%m-%d-%H%M")
        save_fname = os.path.join(os.getcwd(), current_datetime + '_SP' + str(self.subsample) + '.h5')

        src_pts = np.prod(self.sources.npix)
        print("Total point source locations: ", src_pts)

        file = tables.open_file(save_fname, mode="w", title="System Response")
        pt_responses = file.create_earray('/', 'sysmat',
                                          atom=tables.atom.Float64Atom(),
                                          shape=(0, tot_img_pxls),
                                          expectedrows=src_pts)

        prog = 0
        perc = 0

        evts_per_percent = np.floor(0.01 * src_pts)

        for src_pt in self.sources.source_pt_iterator():
            pt_responses.append(self._point_response_function(src_pt, **kwargs).ravel()[None])
            file.flush()
            prog += 1
            if prog > 2 * evts_per_percent:
                perc += prog/src_pts * 100  # total points
                print("Current Source Point: ", src_pt)
                print("Progress (percent): ", perc)
                prog = 0
        file.close()

    def _point_response_function(self, src_pt, **kwargs):
        self.detector_system.projection.fill(0.0)  # Clear

        for det_end_pt, det_normal in self.detector_system.generate_det_sample_pts(**kwargs):
            coll_rays, det_rays, em_dir = self.generate_ray(src_pt, det_end_pt)
            attenuation_collimator = self.collimator._collimator_ray_trace(coll_rays)

            # TODO: THIS SECTION TO TEST IF SOLID ANGLE CORRECTION NOT NEEDED (TOP, AUG 5)
            # relative_projection_area = np.abs(np.sum(det_normal * em_dir))  # this is abs(cos(th))
            # solid_angle = relative_projection_area / (self.subsample ** 2)  # relative solid angle of each sampled point
            solid_angle = 1 / (self.subsample ** 2)
            # TODO: THIS SECTION TO TEST IF SOLID ANGLE CORRECTION NOT NEEDED (BOT, AUG 5)

            # if self.test_prf:
            #    print()
            #    print("PRF test")
            #    print("Attenuation Collimator: ", attenuation_collimator)
            #    print("Relative Projection Area: ", relative_projection_area)
            #    print("Solid_angle: ", solid_angle)
            #    print("Att * Solid Angle: ", attenuation_collimator * solid_angle)
            #    self.test_prf = 0
            self.detector_system._ray_projection(det_rays,
                                                 em_dir,
                                                 coll_att=attenuation_collimator * solid_angle)
        return self.detector_system.projection


def norm(array):
    arr = np.array(array)
    return arr / np.sqrt(np.dot(arr, arr))


def ang2arr(angle_degrees):  # This means the beam-axis (+x) is the reference point for slit angles
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle), 0])


def test(separate=True):
    start = time.time()
    system = Imager()

    # ==================== Collimator ====================
    system.collimator.colp = np.array([0, 0, -130])  # 100 is the default

    # Vertical Slits
    slit1 = np.array([0, 0, 0])
    system.collimator.add_aperture('slit', size=2, loc=slit1)

    h_offset = 48  # horizontal distance between vertical slits
    slit2 = np.array([h_offset, 0, 0])
    system.collimator.add_aperture('slit', loc=slit2)
    system.collimator.add_aperture('slit', loc=-slit2)

    # Outside slits
    joint = (203.2 / 2) - 50.39  # y -coordinate of slit connection, 60 degree slits
    slit3 = np.array([h_offset, joint, 0])  # right side
    slit4 = np.array([h_offset, -joint, 0])  # right side
    system.collimator.add_aperture('slit', slit_ax=ang2arr(60.), x_min=h_offset, loc=slit3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-60.), x_min=h_offset, loc=slit4)

    # Left side outside slits
    ls3 = np.array([-h_offset, joint, 0])
    ls4 = np.array([-h_offset, -joint, 0])
    system.collimator.add_aperture('slit', slit_ax=ang2arr(120.), x_max=-h_offset, loc=ls3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-120.), x_max=-h_offset, loc=ls4)

    # y = 0 slits
    rtd_slits = np.array([h_offset, 0, 0])  # (r)ight (t)hirty (d)egree slits
    system.collimator.add_aperture('slit', slit_ax=ang2arr(30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)

    # Inner Slits
    isv_offset = (203.2 / 2) - 80.94  # inner slit vertical offsets
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, -isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, -isv_offset, 0]))

    # ====== Open Sides ======
    # Bottom Opening
    cw = 203.2
    bot_colly = -cw / 2
    open_width = (5.95 + 61.75)
    table_posy = bot_colly - open_width
    system.collimator.add_aperture('slit', size=open_width, slit_ax=ang2arr(0), aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   y_min=table_posy, y_max=bot_colly,
                                   loc=np.array([0, bot_colly - (open_width / 2), 0]))  # bottom opening to table

    # Right Side Opening
    cw = 203.2  # collimator physical width
    r_open_wid = 2000  # open to right of collimator
    right_opening = (cw / 2) + (r_open_wid / 2)
    system.collimator.add_aperture('slit', size=r_open_wid, aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   x_min=cw / 2, x_max=(cw / 2) + r_open_wid,
                                   loc=np.array([right_opening, 0, 0]))  # +x side opening

    # Left Side opening
    l_open_wid = 2000  # open to right of collimator
    left_opening = -((cw / 2) + (l_open_wid / 2))
    system.collimator.add_aperture('slit', size=l_open_wid, aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   x_min=-((cw / 2) + l_open_wid), x_max=-(cw / 2),
                                   loc=np.array([left_opening, 0, 0]))  # -x side opening

    # ==================== Detectors ====================
    # layout = np.array([4, 4])
    # system.detector_system.layout = layout  # could just put in __init__ of Detector_System

    x = np.array([1, 0, 0])
    # y = np.array([0, 1, 0])
    x_displacement = 25.447  # This is the displacement of the center of the detectors from the center of the collimator according to the 3D model
    distance_mod_plane = np.array([0, 0, -260]) + (x_displacement * x)

    mod_centers, directions = generate_detector_centers_and_norms(system.detector_system.layout,
                                                                  det_width=53.2,
                                                                  focal_length=420.9)  # Det_width and focal_length also from 3D model

    for det_idx, det_center in enumerate(mod_centers + distance_mod_plane):
        print("Set det_center: ", det_center)
        system.create_detector(det_id=det_idx, center=det_center, det_norm=directions[det_idx])
    print("Farthest Plane: ", system.detector_system.farthest_plane)

    # ==================== Sources ====================

    em_pt = np.array([-10, 0, -20])  # change this value to generate a projection at different locations keeping in mind the axes conventions

    # ==================== Attenuation ====================
    # system.collimator.mu = 0.04038 * (10 ** 2)
    # system.collimator.mu = 1000

    # ==================== Run and Display ====================
    system.sample_step = 0.1  # in cm, default
    system.subsample = 1  # samples per detector pixel.

    point_response = system.generate_test_response(em_pt, subsample=system.subsample)
    print('It took ' + str(time.time() - start) + ' seconds.')

    layout = np.array([4, 4])

    if not separate:
        ax = plt.axes()
        for plt_index in np.arange(layout[1] * layout[0]):
            row = plt_index // layout[1]
            col = plt_index % layout[0]
            section = point_response[(12 * row):(12 * (row + 1)), (12 * col):(12 * (col + 1))]
            point_response[(12 * row):(12 * (row + 1)), (12 * col):(12 * (col + 1))] = section

        data = point_response
        im = ax.imshow(data)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im)
        plt.show()
        return

    fig, ax = plt.subplots(layout[0], layout[1])

    for plt_index in np.arange(layout[1] * layout[0]):
        row = plt_index // layout[1]
        col = plt_index % layout[0]

        data = point_response[(12 * row):(12 * (row+1)), (12 * col):(12 * (col+1))]

        im = ax[row, col].imshow(data, origin='lower')
        # plt.colorbar(im, ax=ax[row, col])
        ax[row, col].set_yticks([])
        ax[row, col].set_xticks([])

    fig.tight_layout()
    plt.show()


def main():
    start = time.time()
    system = Imager()

    # ==================== Collimator ====================
    system.collimator.colp = np.array([0, 0, -130])  # 100 is the default

    # Vertical Slits
    slit1 = np.array([0, 0, 0])
    system.collimator.add_aperture('slit', size=2, loc=slit1)

    h_offset = 48  # horizontal distance between vertical slits
    slit2 = np.array([h_offset, 0, 0])
    system.collimator.add_aperture('slit', loc=slit2)
    system.collimator.add_aperture('slit', loc=-slit2)

    # Outside slits
    joint = (203.2 / 2) - 50.39  # y -coordinate of slit connection, 60 degree slits
    slit3 = np.array([h_offset, joint, 0])  # right side
    slit4 = np.array([h_offset, -joint, 0])  # right side
    system.collimator.add_aperture('slit', slit_ax=ang2arr(60.), x_min=h_offset, loc=slit3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-60.), x_min=h_offset, loc=slit4)

    # Left side outside slits
    ls3 = np.array([-h_offset, joint, 0])
    ls4 = np.array([-h_offset, -joint, 0])
    system.collimator.add_aperture('slit', slit_ax=ang2arr(120.), x_max=-h_offset, loc=ls3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-120.), x_max=-h_offset, loc=ls4)

    # y = 0 slits
    rtd_slits = np.array([h_offset, 0, 0])  # (r)ight (t)hirty (d)egree slits
    system.collimator.add_aperture('slit', slit_ax=ang2arr(30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)

    # Inner Slits
    isv_offset = (203.2 / 2) - 80.94  # inner slit vertical offsets
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, -isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, -isv_offset, 0]))

    # ====== Open Sides ======
    # Bottom Opening
    cw = 203.2  # collimator width in each dimension  # Delete this for faster calculation
    bot_colly = -cw / 2
    open_width = (5.95 + 61.75)  # plate thickness + bottom of tungsten to plate
    table_posy = bot_colly - open_width
    system.collimator.add_aperture('slit', size=open_width, slit_ax=ang2arr(0), aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   y_min=table_posy, y_max=bot_colly,
                                   loc=np.array([0, bot_colly - (open_width / 2), 0]))  # bottom opening to table

    # Right Side Opening
    cw = 203.2  # collimator physical width
    r_open_wid = 2000  # open to right of collimator
    right_opening = (cw/2) + (r_open_wid/2)
    system.collimator.add_aperture('slit', size=r_open_wid, aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   x_min=cw/2, x_max=(cw/2) + r_open_wid,
                                   loc=np.array([right_opening, 0, 0]))  # +x side opening

    # Left Side opening
    l_open_wid = 2000  # open to right of collimator
    left_opening = -((cw / 2) + (l_open_wid / 2))
    system.collimator.add_aperture('slit', size=l_open_wid, aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   x_min=-((cw / 2) + l_open_wid), x_max=-(cw / 2),
                                   loc=np.array([left_opening, 0, 0]))  # -x side opening

    # ==================== Detectors ====================
    # layout = np.array([4, 4])
    # system.detector_system.layout = layout  # could just put in __init__ of Detector_System

    x = np.array([1, 0, 0])
    # y = np.array([0, 1, 0])
    x_displacement = 25.447
    distance_mod_plane = np.array([0, 0, -260]) + (x_displacement * x)

    mod_centers, directions = generate_detector_centers_and_norms(system.detector_system.layout,
                                                                  det_width=53.2,
                                                                  focal_length=420.9)

    for det_idx, det_center in enumerate(mod_centers + distance_mod_plane):
        print("Set det_center: ", det_center)
        system.create_detector(det_id=det_idx, center=det_center, det_norm=directions[det_idx])
    print("Farthest Plane: ", system.detector_system.farthest_plane)

    # ==================== Sources ====================
    # sc stands for source center
    # system.sources.sc = np.array([0, -10, -40])  # 90 mm to collimator
    # system.sources.sc = np.array([0, -10, -30])  # 100 mm to collimator
    system.sources.sc = np.array([0, -10, -20])  # This is at 110 mm collim to source distance (default)
    # system.sources.sc = np.array([0, -10, -10])  # 120 mm to collimator

    system.sources.vsze = 1
    system.sources.npix = np.array([201, 61])  # FOV


    # ==================== Attenuation ====================
    # system.collimator.mu = 0.04038 * (10 ** 2)
    # system.collimator.mu = 1000

    # ==================== Run and Display ====================
    system.sample_step = 0.1  # in cm, default
    system.subsample = 2  # samples per detector pixel. TODO: Check if this is right before running

    system.generate_sysmat_response(subsample=system.subsample)
    print('It took ' + str(time.time() - start) + ' seconds.')


if __name__ == "__main__":
    main()  # generate system response, 20 x 20 cm FoV with ~1 mm pitch takes 24-48 hours 
    # test(separate=False)  # try this instead to see projections of a source at em_pt. Separate puts each detector into its own plot. Depending on distance from collimator takes ~4-30 seconds.
