import numpy as np
# from scipy.sparse import lil_matrix
# from fast_histogram import histogram2d

#  Global origin (0,0,0) should be in the center of the collimator on the detector-collimator axis
# Beam in x-hat direction, vertical is y-hat. And the detector and collimator are in the -z direction


class Detector_System(object):  # detectors must be defined from upper left to upper right and then down each row

    def __init__(self, container, det_rows=4, det_cols=4):
        self.imager = container
        self.detectors = []
        self.num_detectors = len(self.detectors)

        self.sample_area = 0
        self.farthest_plane = 0
        self.closest_plane = np.inf
        self.layout = np.array([det_rows, det_cols])

        self.intersect_table = np.zeros([3, 1])
        self.projection = np.zeros([12 * det_rows, 12 * det_cols])

    def add_detector(self, **kwargs):
        new_detector = Detector(self, **kwargs)  # MODIFIED
        self.detectors.append(new_detector)
        if not len(self.detectors) == 1:
            self.sample_area = self.detectors[0].pix_size

        far_plane = (new_detector.c[2]) + (np.sign(new_detector.c[2]) * (new_detector.thickness+new_detector.pix_size))
        close_plane = (new_detector.c[2]) - \
                      (np.sign(new_detector.c[2]) * (np.max(new_detector.npix) * new_detector.pix_size/2))  # Added
        if np.abs(far_plane) > self.farthest_plane:
            self.farthest_plane = far_plane
        if np.abs(close_plane) < self.closest_plane:
            self.closest_plane = close_plane

        self.num_detectors += 1

    def generate_det_sample_pts(self, **kwargs):
        for det in self.detectors:
            for one_pt in det._detector_sample_pts(**kwargs):
                yield one_pt, det.norm

    def initialize_arrays(self):  # det_num, enter, total_intersection
        self.intersect_table = np.zeros([self.num_detectors, 3], dtype=np.int32)
        self.intersect_table[:, 0] = np.arange(self.num_detectors)
        # self.imager.projection = np.zeros(self.detectors[0].npix[::-1] * self.layout)
        self.projection = np.zeros(self.detectors[0].npix[::-1] * self.layout, dtype=np.float64)

    def _ray_projection(self, ray, em_dir, coll_att=1):  # The goal of this function is to check for any intersections,
        # and then run it in order

        for det_num, det in enumerate(self.detectors):
            self.intersect_table[det_num, 1:] = det._check_intersection(ray)

        valid_modules = self.intersect_table[self.intersect_table[:, 1] > 0]
        prev_intersection = 0

        for row, (mod_id, enter, total_intersection) in enumerate(valid_modules[valid_modules[:, 1].argsort()]):
            prev_att = np.exp(-prev_intersection * self.imager.sample_step *
                              self.detectors[0].mu * self.detectors[0].rho)

            cur_det = self.detectors[mod_id]

            start_row = cur_det.mod_id[0] * cur_det.npix[0]
            start_col = cur_det.mod_id[1] * cur_det.npix[1]

            # self.imager.projection[start_row:(start_row+cur_det.npix[0]), start_col:(start_col+cur_det.npix[1])] += \
            self.projection[start_row:(start_row + cur_det.npix[0]), start_col:(start_col + cur_det.npix[1])] += \
                    cur_det._crystal_interaction_probabilities(ray, em_dir,
                                                               self.imager.sample_step,
                                                               enter, total_intersection,
                                                               prefactor=prev_att) * coll_att
            prev_intersection += total_intersection


class Detector(object):
    # mu = 100.0  # mm^2/g mass attenuation coefficient
    # rho = 1/1000.0  # density g/mm^3

    mu = 0.03809 * (10 ** 2)  # mm^2/g
    rho = 7.4 / (10 ** 3)  # density g/mm^3
    # mu/rho = 4.038 E-2 cm^2/g  (LSO:Ce at 4 MeV), density = 7.4 g/cm^3
    # https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html for Lu2SiO4

    def __init__(self, container,  # MODIFIED
                 center=(0, 0, -200),
                 # mm Coordinates of the center face of the detector (facing from the object plane)
                 det_norm=(0, 0, 1),  # unit vector facing collimator
                 det_thickness=20,  # in mm
                 npix_1=12, npix_2=12,
                 pxl_sze=4,  # in mm
                 # ax_1=(1, 0, 0), ax_2=(0, 1, 0),  # July 3 -Commented out
                 det_id=0):
        self.det_system = container  # MODIFIED
        self.c = np.array(center)
        # self.norm = np.array(det_norm)  # TODO: Moved
        self.thickness = det_thickness
        self.npix = np.array((npix_1, npix_2))
        self.pix_size = pxl_sze

        # self.axes = np.array([ax_1, ax_2])  # July 3 - Commented out

        # JULY 3 ADDED TOP
        self._norm = None
        self.axes = np.array([(1, 0, 0), (0, 1, 0)])
        self.norm = np.array(det_norm)
        # JULY 3 ADDED BOT

        self.det_id = det_id
        self.mod_id = np.array([det_id//self.det_system.layout[0], det_id % self.det_system.layout[1]])  # row, col
        # MODIFIED

        self.hist_ax0 = (np.arange(-self.npix[0]/2., self.npix[0]/2. + 1)) * self.pix_size
        self.hist_ax1 = (np.arange(-self.npix[1]/2., self.npix[1]/2. + 1)) * self.pix_size

    # ADDITION TOP - July 3, 2021
    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        n = np.array(value)
        v_dir = np.array((0, 1, 0))  # vertical direction
        h_axis = np.cross(v_dir, n)  # horizontal direction
        new_v_dir = np.cross(n, h_axis)
        self.axes = np.array([norm(h_axis), norm(new_v_dir)])
        self._norm = norm(n)

    # ADDITION BOT - July 3, 2021
    
    # =================== Only used with angle_generator.py ===================
    # TOP
    def face_pts(self, back=False):  # back is True means back plane, TODO: Add to detector_system.py
        ax0_scalars = np.arange((-self.npix[0] / 2. + 0.5),
                                (self.npix[0] / 2. + 0.5)) * self.pix_size

        ax1_scalars = np.arange((-self.npix[1] / 2. + 0.5),
                                (self.npix[1] / 2. + 0.5))[::-1] * self.pix_size
    # Reversed ordering

        ax0_vec = np.outer(ax0_scalars, self.axes[0])
        ax1_vec = np.outer(ax1_scalars, self.axes[1])

        centers = self.c +  (ax1_vec[:, np.newaxis] + ax0_vec[np.newaxis, :]).reshape(-1, 3)

        # return centers.reshape(self.npix[0], self.npix[1]) + (back * (-1) * self.thickness * self.norm)
        return centers + (back * (-1) * self.thickness * self.norm)
    # BOT
    # =================== Only used with angle_generator.py ===================

    def _detector_sample_pts(self, mid=True, subsample=1):  # end_pts
        # For me this meant starting from upper left (facing collimator) and going right then down each row
        subpixels = subsample
        for ax1 in np.arange((-self.npix[1] * subpixels) / 2 + 0.5,
                             (self.npix[1] * subpixels) / 2 + 0.5)[::-1] * self.pix_size / subpixels:
            for ax0 in np.arange((-self.npix[0] * subpixels) / 2 + 0.5,
                                 (self.npix[0] * subpixels) / 2 + 0.5) * self.pix_size / subpixels:
                yield ax0 * self.axes[0] + ax1 * self.axes[1] + self.c + (mid * (-0.5) * self.norm)

    def _check_intersection(self, ray_array):  # the ray should be generated elsewhere and only once from ray sampler
        intersection = (np.abs(np.dot(ray_array - self.c, self.axes[0])) < self.hist_ax0[-1]) & \
                       (np.abs(np.dot(ray_array - self.c, self.axes[1])) < self.hist_ax1[-1]) & \
                       (np.dot(ray_array - self.c, -self.norm) >= 0) & \
                       (np.dot(ray_array - self.c, -self.norm) <= self.thickness)

        total_intersection = np.sum(intersection)

        enter = ((total_intersection > 0) * np.argmax(intersection)) - (total_intersection <= 0)
        # negative 1 if it doesn't intersect

        return [enter, total_intersection]

    def _crystal_interaction_probabilities(self, ray_array, em_dir, step, enter, total_intersection, prefactor=1.0):
        inside_rays = ray_array[enter:(enter+total_intersection)]
        prob_interact = np.exp(-self.mu * self.rho * step * np.arange(inside_rays.shape[0])) * \
                        (self.pix_size ** 2) * \
                        (1 - np.exp(-self.mu * self.rho * step)) / \
                        (4 * np.pi * (step * np.arange(self.det_system.imager._ray_det_enter + enter,
                                                       self.det_system.imager._ray_det_enter + enter +
                                                       total_intersection)) ** 2)
        # comment: prob_interact = (prob doesn't interact to step X) * (prob interacts in step X+1) * (solid angle)
        # * (prob doesn't interact with collimator)
        # solid angle = (prefactor * (area_pixels/solid sphere)), prefactor -> glancing angle on detector surface

        return np.histogram2d(np.dot(inside_rays - self.c, self.axes[0]),
                              np.dot(inside_rays - self.c, self.axes[1]),
                              bins=(self.hist_ax0, self.hist_ax1),
                              weights=prefactor * prob_interact)[0].T[::-1]  # normalize by subpixels


def norm(array):
    arr = np.array(array)
    return arr / np.sqrt(np.dot(arr, arr))
