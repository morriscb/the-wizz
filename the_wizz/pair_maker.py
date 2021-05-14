
from astropy.cosmology import Planck15
from multiprocessing import Lock, Pool
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


class PairMaker(object):
    """Class for computing distance weighted correlations of a reference sample
    with known redshift against a sample with unknown redshifts.

    Parameters
    ----------
    r_mins : `list` of `float`s
        List of bin edge minimums in Mpc.
    r_maxes : `list` of `float`s
        List of bin edge maximums in Mpc.
    z_min : `float`
        Minimum redshift of the reference sample to consider.
    z_max : `float`
        Maximum redshift of the reference sample to consider.
    weight_power : `float`
        Expected power-law slope of the projected correlation function. Used
        for signal matched weighting.
    distance_metric : `astropy.cosmology.LambdaCDM.<distance>`
        Cosmological distance metric to use for all calculations. Should be
        either comoving_distance or angular_diameter_distance. Defaults to
        the Planck15 cosmology and comoving metric.
    output_pairs : `string`
        Name of a directory to write raw pair counts and distances to. Spawns
        a multiprocessing child task to write out data.
    n_write_proc : `int`
        If an output file name is specified, this sets the number of
        subprocesses to spawn to write the data to disk.
    n_write_clean_up : `int`
        If an output file name is specified, this sets the number reference
        objects to process before cleaning up the subprocess queue. Controls
        the amount of memory on the main processes.
    """
    def __init__(self,
                 r_mins,
                 r_maxes,
                 z_min=0.01,
                 z_max=5.00,
                 weight_power=-0.8,
                 distance_metric=None,
                 output_pairs=None,
                 n_write_proc=2,
                 n_write_clean_up=10000,
                 n_z_bins=64):

        if not isinstance(r_mins, list):
            self.r_mins = [r_mins]
        else:
            self.r_mins = r_mins

        if not isinstance(r_maxes, list):
            self.r_maxes = [r_maxes]
        else:
            self.r_maxes = r_maxes
        self.r_min = np.min(r_mins)
        self.r_max = np.max(r_maxes)
        self.z_min = z_min
        self.z_max = z_max
        self.n_z_bins = n_z_bins
        self.n_write_clean_up = n_write_clean_up
        self.n_write_proc = n_write_proc

        if distance_metric is None:
            distance_metric = Planck15.comoving_distance
        self.distance_metric = distance_metric

        self.weight_power = weight_power

        self.output_pairs = output_pairs

    def run(self, reference_catalog, unknown_catalog):
        """Find the (un)weighted pair counts between reference and unknown
        catalogs.

        Parameters
        ----------
        reference_catalog : `dict`
            Catalog of objects with known redshift to count pairs around.
            Dictionary contains:

            ``"ra"``
                RA position in degrees (`numpy.ndarray`, (N,))
            ``"dec"``
                Dec position in degrees (`numpy.ndarray`, (N,))
            ``"id"``
                Unique identifier in the catalog (`numpy.ndarray`, (N,))
            ``"redshift"``
                Redshift of the reference object (`numpy.ndarray`, (N,))
        unknown_catalog : `dict`
            Catalog of objects with unknown redshift to count around the
            reference objects.
            Dictionary contains:

            ``"ra"``
                RA position in degrees (`numpy.ndarray`, (N,))
            ``"dec"``
                Dec position in degrees (`numpy.ndarray`, (N,))
            ``"id"``
                Unique identifier in the catalog (`numpy.ndarray`, (N,))
            ``"weight"``
                OPTIONAL: If setting use_unkn_weights flag, weight to apply
                to each unknown object. (`numpy.ndarray`, (N,))

        Returns
        -------
        output_data : `pandas.DataFrame`
            Summary data produced from the pair finding, cross-correlation.
            Contains a summary of the N_pairs and requested distance weights
            per reference object.
        """
        unkn_vects = self._convert_radec_to_xyz(
            np.radians(unknown_catalog["ra"]),
            np.radians(unknown_catalog["dec"]))
        unkn_tree = cKDTree(unkn_vects)
        unkn_ids = unknown_catalog["id"]
        total_unknown = len(unkn_ids)
        try:
            unkn_weights = unknown_catalog["weight"]
        except KeyError:
            unkn_weights = np.ones(total_unknown, dtype=np.float32)
        ave_weight = np.mean(unkn_weights)

        redshifts = reference_catalog["redshift"]
        z_mask = np.logical_and(redshifts >= self.z_min,
                                redshifts < self.z_max)
        ref_ids = reference_catalog["id"][z_mask]
        ref_vects = self._convert_radec_to_xyz(
            np.radians(reference_catalog["ra"][z_mask]),
            np.radians(reference_catalog["dec"][z_mask]))
        redshifts = reference_catalog["redshift"][z_mask]
        dists = self.distance_metric(redshifts).value
        try:
            ref_regions = reference_catalog["region"][z_mask]
        except KeyError:
            ref_regions = np.zeros(len(ref_ids), dtype=np.uint32)

        output_data = []

        self.subprocs = []
        if self.output_pairs is not None:
            locks = dict()
            for idx in range(1, self.n_z_bins + 1):
                locks[idx] = Lock()
            self.write_pool = Pool(self.n_write_proc,
                                   initializer=pool_init,
                                   initargs=(locks,))
            redshift_args = redshifts.argsort()
            area_cumsum = np.cumsum(self.r_max / dists[redshift_args])
            area_bin_edges = np.linspace(area_cumsum[0],
                                         area_cumsum[-1],
                                         self.n_z_bins + 1)
            bin_edge_idxs = np.searchsorted(
                area_cumsum,
                area_bin_edges)
            self.z_bin_edges = redshifts[redshift_args[bin_edge_idxs]]
            self.z_bin_edges[0] = self.z_min
            self.z_bin_edges[-1] = self.z_max

        for ref_vect, redshift, dist, ref_id, ref_region in zip(ref_vects,
                                                                redshifts,
                                                                dists,
                                                                ref_ids,
                                                                ref_regions):
            # Query the unknown tree.
            unkn_idxs = np.array(self._query_tree(ref_vect, unkn_tree, dist))
            if len(unkn_idxs) == 0:
                #Didn't find any pairs
                output_row = dict([("ref_id", ref_id),
                           ("redshift", redshift),
                           ("region", ref_region)])

                for r_min, r_max in zip(self.r_mins, self.r_maxes):
                    scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
                    output_row["%s_count" % scale_name] = 0
                    output_row["%s_weight" % scale_name] = 0.

            else:
                # Compute angles and convert them to cosmo distances.
                matched_unkn_vects = unkn_vects[unkn_idxs]
                dots = np.dot(matched_unkn_vects, ref_vect)
                dot_mask = dots < np.cos(self.r_min / dist)
                matched_unkn_dists = np.arccos(dots[dot_mask]) * dist

                # Bin data and return counts/sum of weights in bins.
                output_row = self._compute_bin_values(
                    ref_id,
                    ref_region,
                    redshift,
                    unkn_ids[unkn_idxs[dot_mask]],
                    matched_unkn_dists,
                    unkn_weights[unkn_idxs[dot_mask]])
            output_row["tot_sample"] = total_unknown
            output_row["ave_unkn_weight"] = ave_weight
            output_data.append(output_row)

        output_data_frame = pd.DataFrame(output_data)

        if self.output_pairs is not None:
            self._clean_up()
            self.write_pool.close()
            self.write_pool.join()

            for region in np.unique(output_data_frame["region"]):
                mask = output_data_frame["region"] == region
                sub_df = output_data_frame[mask]
                sub_df.to_parquet("%s/region=%i/reference_data.parquet" %
                                  (self.output_pairs, region),
                                  compression='gzip',
                                  index=False)

        return output_data_frame

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

    def _query_tree(self, ref_vect, unkn_tree, dist):
        """Query the kdtree for all points within the maximum r value at a
        given redshift/distance.

        Parameters
        ----------
        ref_vecct : `numpy.ndarray`, (3,)
            Position to center ball tree search on.
        unkn_tree : `scipy.spatial.cKDTree`
            Searchable kdtree containing points to correlate with.
        dist : `float`
            Distance from observer to the reference object at redshift, z.

        Returns
        -------
        output_indexes : `list` of `int`s
            List of integer index lookups into the array the tree was created
            with.
        """
        theta_max = self.r_max / dist
        return unkn_tree.query_ball_point(
            ref_vect,
            np.sqrt(2 - 2 * np.cos(theta_max)))

    def _compute_bin_values(self,
                            ref_id,
                            region,
                            redshift,
                            unkn_ids,
                            unkn_dists,
                            unkn_weights):
        """Bin data and construct output dict.

        If an output data file is specified, send the raw pairs off to be
        written to disk.

        Parameters
        ----------
        ref_id : `int`
            Unique identifier for the reference object.
        redshift : `float`
            Redshift of the reference object.
        unkn_ids : `numpy.ndarray`, (N,)
            Unique ids of all objects with unknown redshift that are within
            the distance r_min to r_max
        unkn_dists : `numpy.ndarray`, (N,)
            Distances in Mpc from the reference to the unknown objects between
            r_min and r_max.
        unkn_weights : `numpy.ndarray`, (N,)
            Weights for each object with unknown redshift.


        Returns
        -------
        output_row : `dict`
            Dictionary containing the values:

            ``"ref_id"``
                Unique reference id (`int`)
            ``"redshift"``
                Reference redshift (`float`)
            ``"region"``
                Spatial region the reference belongs to (`int`)
            ``"[scale_name]_count"``
                Number of unknown objects with the annulus around the
                reference for annulus [scale_name]. (`int`)
            ``"[scale_name]_weight"``
                Weighted number  unknown objects with the annulus around the
                reference for annulus [scale_name]. (`float`)
        """
        output_row = dict([("ref_id", ref_id),
                           ("redshift", redshift),
                           ("region", region)])

        if self.output_pairs is not None and len(unkn_ids) > 0:
            z_bin = np.digitize(redshift, self.z_bin_edges)
            self._subproc_write(ref_id, region, z_bin, unkn_ids, unkn_dists)

        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
            r_mask = np.logical_and(unkn_dists >= r_min, unkn_dists < r_max)

            bin_unkn_ids = unkn_ids[r_mask]
            bin_unkn_dists = unkn_dists[r_mask]
            bin_unkn_weights = unkn_weights[r_mask]

            output_row["%s_count" % scale_name] = len(bin_unkn_ids)
            output_row["%s_weight" % scale_name] = (
                bin_unkn_weights *
                distance_weight(bin_unkn_dists, self.weight_power)).sum()

        return output_row

    def _subproc_write(self, ref_id, region, z_bin, unkn_ids, unkn_dists):
        """Construct a dataformate of values to be written to disk via a
        subprocess.

        Parameters
        ----------
        ref_id : `int`
            Unique identifier for the reference object.
        region : `int`
            Spatial region this reference belongs to.
        unkn_ids : `numpy.ndarray`, (N,)
            Unique ids of all objects with unknown redshift that are within
            the distance r_min to r_max
        unkn_dists : `numpy.ndarray`, (N,)
            Distances in Mpc from the reference to the unknown objects between
            r_min and r_max.
        """
        scale_name = "Mpc%.2ft%.2f" % (self.r_min, self.r_max)
        output_dict = dict(
            [("ref_id", ref_id),
             ("region", region),
             ("z_bin", z_bin),
             ("file_name", self.output_pairs),
             ("scale_name", scale_name),
             ("unkn_id", unkn_ids),
             ("dists", unkn_dists)])

        if len(self.subprocs) >= self.n_write_clean_up:
            self._clean_up()
        self.subprocs.append(self.write_pool.apply_async(
            write_pairs,
            (output_dict,),
            error_callback=error_callback))

    def _clean_up(self):
        """Cleanup subprocesses.
        """
        for subproc in self.subprocs:
            subproc.get()
        del self.subprocs
        self.subprocs = []


def distance_weight(dists, power=-0.8):
    """Convert raw distances into a signal matched weight for the
    correlation.

    All weights with distances below 0.001 Mpc will be set to a value of
    0.001 ** ``weight_power``.

    Parameters
    ----------
    dists : `numpy.ndarray`, (N,)
        Distances in Mpc.
    power : `float`
        Exponent to raise distance to the power of.

    Returns
    -------
    weights : `numpy.ndarray`, (N,)
        Output weights.
    """
    return np.where(dists > 0.01, dists ** power, 0.01 ** power)


def pool_init(locks):
    """Initializer for enabling locking for multiprocessing.Pool.

    Copies a dict of locks into a global to prevent multiple writes to the
    same output file.

    Parameters
    ----------
    locks : `dict`
        A dictionary with integer keys mapping to `multiprocessing.Lock`
        objects.
    """
    global lock_dict
    lock_dict = locks


def write_pairs(data):
    """Write raw pairs produced by pair maker to disk.

    Ids are loss-lessly compressed distances are stored as log, keeping 3
    decimal digits.

    Parameters
    ----------
    data : `dict`
        Dictionary of data produced by the PairMaker class.
        Dictionary has should have following keys:

        ``"file_name"``
            File name of the file to write to. (`str`)
        ``"ref_id"``
            Id of the reference object. (`int`)
        ``"scale_names"``
            Names of the scales run in pair_maker. Formated e.g.
            'Mpc1.00t10.00' (`list`)
        ``"'scale_name'_ids"``
            Unique ids of unknown objects within annulus 'scale_name' around
            the reference object (`numpy.ndarray`, (N,))
        ``"'scale_name'_dists"``
            Distance to unknown object with id in 'scale_name'_ids
            (`numpy.ndarray`, (N,))
    """
    ids = data["unkn_id"]
    comp_log_dists = compress_distances(data["dists"])

    n_pairs = len(ids)
    ref_ids = np.full(n_pairs, data["ref_id"], dtype=np.uint64)
    regions = np.full(n_pairs, data["region"], dtype=np.uint32)
    z_bins = np.full(n_pairs, data["z_bin"], dtype=np.uint32)

    id_sort_args = ids.argsort()
    output_table = pa.Table.from_batches([pa.RecordBatch.from_arrays(
        [pa.array(ref_ids),
         pa.array(regions),
         pa.array(z_bins),
         pa.array(ids[id_sort_args]),
         pa.array(comp_log_dists[id_sort_args])],
        ["ref_id", "region", "z_bin", "unkn_id", "comp_log_dist"])])
    lock_dict[data["z_bin"]].acquire()
    pq.write_to_dataset(output_table,
                        root_path=data["file_name"],
                        compression="gzip",
                        partition_cols=["region", "z_bin"])
    lock_dict[data["z_bin"]].release()


def error_callback(exception):
    """Simple function to propagate errors from multiprocessing.Process
    objects.

    Parameters
    ----------
    exception : `Exception`
    """
    raise exception


def compress_distances(dists):
    """Log and convert distances to int type.

    Compression is lossy, keeping only 4 decimals in the log.

    Parameters
    ----------
    dists : `numpy.ndarray`
        Distances in Mpc to convert to int for compression.

    Returns
    -------
    comp_dists : `numpy.ndarray`
        Integer array representing Mpc distances. Can be converted back to
        distance by using `decompress_distances`.
    """
    return (np.log(dists) * 10 ** 4).astype(np.int32)


def decompress_distances(comp_dists):
    """Convert dists from int to float and unlog them.

    Parameters
    ----------
    comp_dists : `numpy.ndarray`
        Integer representations of log distances produced by the function
        `compres_distances`.

    Returns
    -------
    dists : `numpy.ndarray`
        Float uncompressed data.
    """
    return np.exp(comp_dists * 10 ** -4)
