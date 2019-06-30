
"""Utility functions for computing single galaxy clustering redshfits using
a k-dimensional tree in galaxy parameter space.
"""

from __future__ import division, print_function, absolute_import

from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree, minkowski_distance

from the_wizz.pdf_maker_utils import _collapse_multiplex

# TODO:
#     Add option to pickle and loaded pickled CatalogKDtree objects.


def collapse_ids_to_single_estimate(hdf5_pairs_group, pair_data, pdf_maker_obj,
                                    unknown_data, args):
    """This is the heart of the-wizz. It enables the matching of a set of
    catalog ids to the ids stored as pairs to the spectroscopic
    objects. The result of this calculation is a intermediary data product
    containing the density of unknown objects around each reference object
    stored in the PDFMaker data structure class. This specific version is
    for when all the spectra have been pre-loaded in anticipation of running
    a large number of sub-samples as is the case with kdtree recovery.
    ----------------------------------------------------------------------------
    Args:
        hdf5_pairs_group: hdf5 group object containing the pair ids for a fixed
            annulus.
        unknown_data: open fits data containing object ids and relivent weights
        args: ArgumentParser.parse_args object returned from
            input_flags.parse_input_pdf_args
    Returns:
        None
    """
    print("\tpre-loading unknown data...")
    if args.unknown_weight_name is not None:
        unknown_data = unknown_data[
            unknown_data[args.unknown_weight_name] != 0]
    id_array = unknown_data[args.unknown_index_name]
    id_args_array = id_array.argsort()
    id_array = id_array[id_args_array]
    rand_ratio = (
        unknown_data.shape[0]/(1.*hdf5_pairs_group.attrs['n_random_points']))
    if args.unknown_stomp_region_name is not None:
        id_array = [id_array[
            unknown_data[args.unknown_stomp_region_name][id_args_array] ==
            reg_idx]
            for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])]
        tmp_n_region = np.array(
            [id_array[reg_idx].shape[0]
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.int_)
        rand_ratio = ((tmp_n_region /
                       (1.*hdf5_pairs_group.attrs['n_random_points'])) *
                      (hdf5_pairs_group.attrs['area'] /
                       hdf5_pairs_group.attrs['region_area']))
    ave_weight = 1.0
    weight_array = np.ones(unknown_data.shape[0], dtype=np.float32)
    if args.unknown_weight_name is not None:
        weight_array = unknown_data[args.unknown_weight_name][id_args_array]
        ave_weight = np.mean(weight_array)
    if args.unknown_stomp_region_name is not None:
        weight_array = [weight_array[
            unknown_data[args.unknown_stomp_region_name][id_args_array] ==
            reg_idx]
            for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])]
        ave_weight = np.array(
            [weight_array[reg_idx].mean()
             for reg_idx in xrange(hdf5_pairs_group.attrs['n_region'])],
            dtype=np.float_)
    n_reference = len(hdf5_pairs_group)
    reference_unknown_array = np.empty(n_reference, dtype=np.float32)
    pool = Pool(args.n_processes)
    if args.unknown_stomp_region_name is not None:
        pool_iter = pool.imap(
            _collapse_multiplex,
            [(data_set,
              id_array[pdf_maker_obj.reference_region_array[pair_idx]],
              weight_array[pdf_maker_obj.reference_region_array[pair_idx]],
              args.use_inverse_weighting)
             for pair_idx, data_set in enumerate(pair_data)],
            chunksize=np.int(np.where(args.n_processes > 1,
                             np.log(len(pair_data)), 1)))
    else:
        pool_iter = pool.imap(
            _collapse_multiplex,
            [(data_set, id_array, weight_array, args.use_inverse_weighting)
             for pair_idx, data_set in enumerate(pair_data)],
            chunksize=np.int(np.where(args.n_processes > 1,
                                      np.log(len(pair_data)), 1)))
    print("\t\tcomputing/storing pair count...")
    for pair_idx, reference_value in enumerate(pool_iter):
        reference_unknown_array[pair_idx] = reference_value
    pool.close()
    pool.join()
    pdf_maker_obj.set_reference_unknown_array(reference_unknown_array)
    pdf_maker_obj.scale_random_points(rand_ratio, ave_weight)
    return None


def create_match_data(input_catalog, mag_name_list, other_name_list,
                      use_as_colors):
    kdtree_data_array = np.empty((input_catalog.shape[0], len(mag_name_list)))
    for mag_idx, mag_name in enumerate(mag_name_list):
        kdtree_data_array[:, mag_idx] = input_catalog[mag_name]
        if mag_idx > 0 and use_as_colors:
            kdtree_data_array[:, mag_idx - 1] -= kdtree_data_array[:, mag_idx]
    if use_as_colors:
        kdtree_data_array = np.delete(kdtree_data_array, -1, 1)
    if len(other_name_list) > 0:
        other_data_array = np.empty((input_catalog.shape[0],
                                     len(other_name_list)))
        for other_idx, other_name in enumerate(other_name_list):
            other_data_array[:, other_idx] = input_catalog[other_name]
        kdtree_data_array = np.concatenate(
            (kdtree_data_array, other_data_array), axis=1)
    return kdtree_data_array


class CatalogKDTree(object):
    """Convience class for creating a dataset suitable for a KDTree search, and
    wrapping the scipy KDTree object.
    """
    def __init__(self, input_array):
        """__init__ method preps the internal data storage and creates the
        KDTree.
        ------------------------------------------------------------------------
        Args:
            input_catalog: astropy.io.fits catalog object containing the
                columns of interest
            column_name_list: list of string names of catalog columns to
               consider for the KDTree
            id_column_name: string name of the column containing the indices
        """
        self._internal_array = input_array
        self._normalize_data()
        self._initialize_tree()

    def __call__(self, input_array, k):
        """Given input properties of an object, return the KDTree, array indices
        of the k nearest neighbors.
        ------------------------------------------------------------------------
        Args:
            input_array: float array of object properties (eg fluxes in survey
                bands)
            k: int number of nearest neighbors to return.
        Returns:
            tuple;
                array of integer array indices of objects
                list of quartile and max distances
        """
        tmp_array = (input_array - self._mean_array)/self._std_array
        d, i = self._kd_tree.query(tmp_array, k)
        return i, d[[int(k/4.), int(k/2.), int(3.*k/4.), -1]]

    def k_nearest_ball_point(self, input_array, max_dist):
        """Method to return the KDTree indicies from all points within a fixed
        distance of the point requested. The distance is expressed in sigma of
        the stored data array, i.e. a value of 1 returns all points within 1
        sigma.
        ------------------------------------------------------------------------
        Args:
            input_array: float array of object properties (eg fluxes in survey
                bands)
            max_dist: Maximum radial distance to search from the input point.
        Returns:
            int array of array indices
        """
        tmp_array = (input_array - self._mean_array) / self._std_array
        return self._kd_tree.query_ball_point(tmp_array, max_dist)

    def _initialize_tree(self):
        """Internal method for intilizing the KDTree object.
        -----------------------------------------------------------------------
        Args:
            self
        Returns:
            None
        """
        self._kd_tree = cKDTree(self._internal_array)
        return None

    def _normalize_data(self):
        """Internal method for scaling the data columns stored to a standard
        normal distribution of mean zero and standard deviation of 1.
        ------------------------------------------------------------------------
        Args:
            self
        Returns:
            None
        """
        self._mean_array = self._internal_array.mean(axis=0)
        self._std_array = self._internal_array.std(axis=0)
        for col_idx in xrange(self._internal_array.shape[1]):
            self._internal_array[:, col_idx] = (
                (self._internal_array[:, col_idx] -
                 self._mean_array[col_idx]) / self._std_array[col_idx])
        return None

    def get_mean_array(self):
        return self._mean_array

    def get_std_array(self):
        return self._std_array


class SphericalKDTree(object):
    """
    SphericalKDTree(RA, DEC, leaf_size=16)

    A binary search tree based on scipy.spatial.cKDTree that works with
    celestial coordinates. Provides methods to find pairs within angular
    apertures (ball) and annuli (shell). Data is internally represented on a
    unit-sphere in three dimensions (x, y, z).

    Parameters
    ----------
    RA : array_like
        List of right ascensions in degrees.
    DEC : array_like
        List of declinations in degrees.
    leafsize : int
        The number of points at which the algorithm switches over to
        brute-force.
    """

    def __init__(self, RA, DEC, leafsize=16):
        # convert angular coordinates to 3D points on unit sphere
        pos_sphere = self._position_sky2sphere(RA, DEC)
        self.tree = cKDTree(pos_sphere, leafsize)

    @staticmethod
    def _position_sky2sphere(RA, DEC):
        """
        _position_sky2sphere(RA, DEC)

        Maps celestial coordinates onto a unit-sphere in three dimensions
        (x, y, z).

        Parameters
        ----------
        RA : float or array_like
            Single or list of right ascensions in degrees.
        DEC : float or array_like
            Single or list of declinations in degrees.

        Returns
        -------
        pos_sphere : array like
            Data points (x, y, z) representing input points on the unit-sphere,
            shape of output is (3,) for a single input point or (N, 3) for a
            set of N input points.
        """
        ras_rad = np.deg2rad(RA)
        decs_rad = np.deg2rad(DEC)
        try:
            pos_sphere = np.empty((len(RA), 3))
        except TypeError:
            pos_sphere = np.empty((1, 3))
        cos_decs = np.cos(decs_rad)
        pos_sphere[:, 0] = np.cos(ras_rad) * cos_decs
        pos_sphere[:, 1] = np.sin(ras_rad) * cos_decs
        pos_sphere[:, 2] = np.sin(decs_rad)
        return np.squeeze(pos_sphere)

    @staticmethod
    def _distance_sky2sphere(dist_sky):
        """
        _distance_sky2sphere(dist_sky)

        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.

        Parameters
        ----------
        dist_sky : float or array_like
            Single or list of separations in celestial coordinates.

        Returns
        -------
        dist_sphere : float or array_like
            Celestial separation converted to (x, y, z) Euclidean distance.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere):
        """
        _distance_sphere2sky(dist_sphere)

        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.

        Parameters
        ----------
        dist_sphere : float or array_like
            Single or list of Euclidean distances in (x, y, z) space.

        Returns
        -------
        dist_sky : float or array_like
            Euclidean distance converted to celestial angular separation.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def query_radius(self, RA, DEC, r):
        """
        query_radius(RA, DEC, r)

        Find all data points within an angular aperture r around a reference
        point with coordiantes (RA, DEC) obeying the spherical geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        r : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with sepration < r.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        point_sphere = self._position_sky2sphere(RA, DEC)
        # find all points that lie within r
        r_sphere = self._distance_sky2sphere(r)
        idx = self.tree.query_ball_point(point_sphere, r_sphere)
        # compute pair separation
        dist_sphere = minkowski_distance(self.tree.data[idx], point_sphere)
        dist = self._distance_sphere2sky(dist_sphere)
        return idx, dist

    def query_shell(self, RA, DEC, rmin, rmax):
        """
        query_radius(RA, DEC, r)

        Find all data points within an angular annulus rmin <= r < rmax around
        a reference point with coordiantes (RA, DEC) obeying the spherical
        geometry.

        Parameters
        ----------
        RA : float
            Right ascension of the reference point in degrees.
        DEC : float
            Declination of the reference point in degrees.
        rmin : float
            Minimum separation of data points from the reference point.
        rmax : float
            Maximum separation of data points from the reference point.

        Returns
        -------
        idx : array_like
            Positional indices of matching data points in the search tree data
            with rmin <= sepration < rmax.
        dist : array_like
            Angular separation of matching data points from reference point.
        """
        # find all points that lie within rmax
        idx, dist = self.query_radius(RA, DEC, rmax)
        # remove pairs with r >= rmin
        dist_mask = dist >= rmin
        idx = np.compress(dist_mask, idx)
        dist = np.compress(dist_mask, dist)
        return idx, dist


if __name__ == "__main__":

    N = 10000

    # test distance conversion
    for dist_sky in np.linspace(0.0, 360.0):
        dist_sphere = SphericalKDTree._distance_sky2sphere(dist_sky)
        back_trans = SphericalKDTree._distance_sphere2sky(dist_sphere)
        dist_sky_wrapped = np.minimum(dist_sky, 360.0 - dist_sky)
        assert(np.isclose(dist_sky_wrapped, back_trans))

    # test data conversion
    rand_ra = np.random.uniform(0.0, 180.0, size=N)
    rand_delta_dec = np.random.uniform(0.0, 180.0, size=N)
    # compute two points that lie on a great circle
    pos_sky1 = (rand_ra, 90.0 - rand_delta_dec / 2.0)
    pos_sky2 = (180.0 + rand_ra, 90.0 - rand_delta_dec / 2.0)
    pos_sphere1 = SphericalKDTree._position_sky2sphere(*pos_sky1)
    pos_sphere2 = SphericalKDTree._position_sky2sphere(*pos_sky2)
    # these points must be separated by dist_sky
    dist_sphere = minkowski_distance(pos_sphere1, pos_sphere2)
    back_trans = SphericalKDTree._distance_sphere2sky(dist_sphere)
    assert(np.isclose(rand_delta_dec, back_trans).all())

    # random sky coordinates
    RAs = np.random.uniform(0.0, 360.0, size=N)
    DECs = np.random.uniform(70.0, 89.99, size=N)
    rmin, rmax = 0.5, 1.0  # degrees
    point_sky = (44.0, 90.0)  # northern pole
    # compute using brute force
    pos_sphere = SphericalKDTree._position_sky2sphere(RAs, DECs)
    point_sphere = SphericalKDTree._position_sky2sphere(*point_sky)
    dist_sphere = minkowski_distance(pos_sphere, point_sphere)
    dist_sky = SphericalKDTree._distance_sphere2sky(dist_sphere)
    mask = (dist_sky >= rmin) & (dist_sky < rmax)
    idx_bruteforce = set(np.arange(N)[mask])
    dist_bruteforce = dist_sky[mask]
    # test SphericalKDTree
    tree = SphericalKDTree(RAs, DECs)
    idx_kd, dist_kd = tree.query_shell(
        point_sky[0], point_sky[1], rmin, rmax)
    idx_kd = set(idx_kd)
    assert(dist_kd.min() >= rmin)
    assert(dist_kd.max() < rmax)
    assert(idx_kd == idx_bruteforce)
