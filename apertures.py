import numpy as np


class Collimator(object):
    # mu = 100.0  # mm^2/g mass attenuation coefficient
    # rho = 1 / 1000.0  # density g/mm^3

    mu = 0.04038 * (10**2)  # mm^2/g
    rho = 19.3 / (10**3)  # density g/mm^3
    # mu/rho = 4.038 E-2 cm^2/g  (Tungsten at 4 MeV), density = 19.3 g/cm^3

    # (4.057, 4.4 MeV), (4.222, 6.1 MeV) * (10 ** -2) cm^2/g

    _sup_aper = ('slit', 'pinhole')

    def __init__(self, container,
                 x_limits=(-100.0, 100.0),  # in mm
                 y_limits=(-100.0, 100.0),  # in mm
                 coll_norm=np.array([0, 0, 1]),  # normal facing object
                 coll_plane=np.array([0, 0, -100]),  # in mm
                 coll_thickness=75  # in mm
                 ):
        self.imager = container
        self.xlim = np.array(x_limits)
        self.ylim = np.array(y_limits)
        self.norm = norm(coll_norm)
        self.colp = np.array(coll_plane)
        self.col_half_thickness = coll_thickness/2

        self.apertures = []

        self.debug = 0

    def add_aperture(self, aperture, **kwargs):
        if aperture not in self._sup_aper:
            raise ValueError('Aperture type {f} is not supported. '
                             'Supported aperture types: {a}'.format(f=aperture, a=str(self._sup_aper)[1:-1]))
        if aperture == 'slit':
            self.apertures.append(Slit(self, **kwargs))
        # if aperture == 'pinhole':
        #    self.apertures.append(Pinhole(**kwargs))

    def _collimator_ray_trace(self, ray):
        ray_in_collimator = ray[np.abs(np.dot((ray - self.colp), self.norm)) < self.col_half_thickness]
        attenuation = np.zeros(ray_in_collimator.shape[0])

        for aper in self.apertures:
            attenuation += aper.ray_pass(ray_in_collimator)

        # return np.any(attenuation > 0)  # EXTREME, uncomment to only see pattern
        return np.exp(-self.rho * self.mu * np.sum((attenuation == 0)) * self.imager.sample_step)


class Slit(object):
    # Beam ->+x
    __slots__ = ['collimator', 'sze', 'c', 'f', 'tube', 'x_lim', 'y_lim', 'a', 'h_ang', 'colz', 'opening']

    def __init__(self, container, size=2, loc=(0, 0, 0), cen_ax=(0, 0, 1), slit_ax=(0, 1, 0), aper_angle=20,
                 chan_length=2,
                 x_min=-np.inf,
                 x_max=np.inf,
                 y_min=-np.inf,
                 y_max=np.inf):
        self.collimator = container
        self.sze = size/2.  # Minimum half-width of aperture in mm aka channel width
        self.c = np.array(loc) + self.collimator.colp  # Some point along center slit plane
        self.f = np.array(cen_ax)  # Normal to plane of aperture (focus)
        self.tube = chan_length  # Total length
        self.x_lim = np.sort([x_min, x_max])
        self.y_lim = np.sort([y_min, y_max])

        self.a = np.array(slit_ax)  # axis of slit
        self.opening = np.cross(self.a, self.f)  # Vector that is in direction of the slit opening

        self.h_ang = np.deg2rad(aper_angle/2.)  # Half of the opening angle of the slit in radians

    def ray_pass(self, ray):  # Ray should be an array of 3d points that correspond to the passing ray
        ray_check = np.zeros(ray.shape[0])  # TODO: Care here
        near_slit = (ray[:, 0] >= self.x_lim[0]) & (ray[:, 0] <= self.x_lim[1]) & \
                    (ray[:, 1] >= self.y_lim[0]) & (ray[:, 1] <= self.y_lim[1])

        if not np.count_nonzero(near_slit):  # i.e. if any points near slit, continue to rest of function
            return ray_check  # TODO: If this works, significant speed up of code expected. This should return zeros

        proj_f = np.abs(np.dot(ray - self.c, self.f))  # project onto normal
        proj_u = np.abs(np.dot(ray - self.c, self.opening))  # project along opening
        # ray_check = np.zeros(proj_f.size)  # NOTE: Previously here

        # near_slit = (ray[:, 0] >= self.x_lim[0]) & (ray[:, 0] <= self.x_lim[1]) & \  # Originally here
        #             (ray[:, 1] >= self.y_lim[0]) & (ray[:, 1] <= self.y_lim[1])
        # near_slit checks if the ray is inside the 2D square that encloses the slit or slit segment

        chn_out = ((proj_f - (self.tube / 2.)) > 0)  # Projection onto slit axis is outside channel

        ray_check[chn_out & near_slit] = proj_u[chn_out & near_slit] < \
            (((proj_f[chn_out & near_slit] - (self.tube / 2.)) * np.tan(self.h_ang)) + self.sze)

        ray_check[~chn_out & near_slit] = (proj_u[~chn_out & near_slit] < self.sze)  # Inside channel

        # I.E. when the ray is inside the slit you get a 1, 0 when outside
        return ray_check


def norm(array):
    arr = np.array(array)
    return arr / np.sqrt(np.dot(arr, arr))
