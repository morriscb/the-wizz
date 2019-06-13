
import h5py
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree




class PairMaker(object):
    """
    """

    def __init__(self, r_mins, r_maxs):
        pass

    def run(self, catalog):
        """
        """
        pass

    def _convert_radec_to_xyz(self, ras, decs):
        """Convert RA/DEC positions to points on the unit sphere.

        Parameters
        ----------
        ras : `numpy.ndarray`, (N,)
            Right assertion coordinate in radians
        decs : `numpy.ndarray`, (N,)
            Declination coordinate in radians

        Returns
        -------
        vectors : `numpy.ndarray`, (N, 3)
            Array of points on the unit sphere.
        """
        vectors = np.empty((len(ras), 3))

        vectors[:, 2] = np.sin(decs)

        sintheta = np.cos(decs)
        vectors[:, 0] = np.cos(ras) * sintheta
        vectors[:, 1] = np.sin(ras) * sintheta

        return vectors

    def _make_kdtree(self, ras, decs):
        """Create a spatial search tree given input ra/decs.

        Parameters
        ----------
        ras : `numpy.ndarray`, (N,)
            Right assertion coordinate in radians
        decs : `numpy.ndarray`, (N,)
            Declination coordinate in radians

        Returns
        -------
        spatial_tree : `scipy.spatial.cKDTree`
            searchable spatial tree on the unit-sphere.
        """
        vectors = self._convert_radec_to_xyz(ras, decs)
        return cKDTree(vectors)
