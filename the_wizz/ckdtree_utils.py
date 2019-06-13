import numpy as np
from itertools import compress
from scipy.spatial import minkowski_distance, cKDTree

import stomp


class ccKDTree(cKDTree):

    __doc__ = cKDTree.__doc__

    def __init__(self, *args, **kwargs):
        super(ccKDTree, self).__init__(*args, **kwargs)

    def query_shell_point(self, x, r_min, r_max, p=2., eps=0, n_jobs=1):
        """
        query_hypershell_point(self, x, r_min, r_max, p=2., eps=0)

        Find all points within distance ``r_min <= r < r_max`` of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r_min : positive float
            The minimum radius of points to return.
        r_max : positive float
            The maximum radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r_max / (1 + eps)``, and branches
            are added in bulk if their furthest points are nearer than
            ``r_max * (1 + eps)``.
        n_jobs : int, optional
            Number of jobs to schedule for parallel processing. If -1 is given
            all processors are used. Default: 1.

        Returns
        -------
        d : list of floats or array of lists
            If x is a single point, returns a list of the distances to the
            neighbors of x. If x is an array of points, returns an object array
            of shape tuple containing lists of distances.
        i : list of ints or array of lists
            If x is a single point, returns a list of the indices of the
            neighbors of x. If x is an array of points, returns an object array
            of shape tuple containing lists of neighbors.

        Notes
        -----
        Performance penalty from using python code to reject points with
        ``r < r_min``.

        """
        # check input
        if r_min >= r_max:
            raise ValueError("r_max must be larger than r_min")
        # query normal ball point up to outer radius
        idx_array = self.query_ball_point(x, r_max, p, eps, n_jobs)
        # reject points with r_min < r
        # performance note:
        # list(itertools.compress(list, mask)) beats np.asarray(list)[mask]
        if type(idx_array) is list:  # only a single point in x
            r = minkowski_distance(x, self.data[idx_array], p)
            mask = r >= r_min
            d = list(compress(r, mask))
            i = list(compress(idx_array, mask))
        else:  # a list of points in x
            d = []
            i = []
            for x_element, idx_list in enumerate(zip(x, idx_array)):
                r = minkowski_distance(x_element, self.data[idx_list], p)
                mask = r >= r_min
                d.append(list(compress(r, mask)))
                i.append(list(compress(idx_list, mask)))
        # return same output type as query_ball_point
        return d, i


