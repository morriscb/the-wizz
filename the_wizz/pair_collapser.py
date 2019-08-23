
from glob import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd

from .pair_maker import decompress_distances, distance_weight


class PairCollapser:
    """Collapse raw ids and distances of unknown objects into an over-density
    estimate around each reference contained in the pair file.

    Parameters
    ----------
    parquet_pairs : `str`
        Location of the parquet pair file output by PairMaker
    r_mins : `list`
        Minimum radius for pair binning. Must be paired with a r_max
    r_mins : `list`
        Maximum radius for pair binning. Must be paired with a r_min
    weight_power : `float`
        Power law slope to weight distances in weighted pairs by. Defaults to
        -0.8
    n_proc : `int`
        Number of sub processes to use when computing the over-densities.
        Parallelization is over the redshift bins stored per region in the pair
        file.
    """

    def __init__(self,
                 parquet_pairs,
                 r_mins,
                 r_maxes,
                 weight_power=-0.8,
                 n_proc=0):
        self.parquet_pairs = parquet_pairs
        self.r_mins = r_mins
        self.r_maxes = r_maxes
        self.weight_power = weight_power
        self.n_proc = n_proc

    def run(self,
            unknown_catalog):
        """Compute the over-density around reference objects stored in the
        pair file given the ids and weights in the input unknown_catalog.

        Parameters
        ----------
        unknown_catalog : `dict`
            Catalog of objects with unknown redshift to cross-correlate against
            the set of reference objects stored in the pair file. Input catalog
            should cover the same area as that covered by the references and
            be regionated similarly.
            Dictionary should contain contains:

            ``"id"``
                Unique identifier in the catalog (`numpy.ndarray`, (N,))
            ``"region"``
                Spatial region the unknown objects belong to. Optional only
                if the pair file contains one region.
                (`numpy.ndarray`, (N,))
            ``"weight"``
                OPTIONAL: If setting use_unkn_weights flag, weight to apply
                to each unknown object. (`numpy.ndarray`, (N,))

        Returns
        -------
        output : `pandas.DataFrame`
            Summary data produced from the pair finding, cross-correlation.
            Contains a summary of the N_pairs and requested distance weights
            per reference object.
        """
        unkn_ids = unknown_catalog["id"]
        try:
            unkn_regions = unknown_catalog["region"]
        except KeyError:
            print("WARNING: not taking advantage of the spatial regionation "
                  "in the pair file may lead to long run times.")
            unkn_regions = np.zeros(len(unkn_ids), dtype=np.uint32)
        try:
            unkn_weights = unknown_catalog["weight"]
        except KeyError:
            unkn_weights = np.ones(len(unkn_ids))

        unique_regions = np.unique(unkn_regions)

        output = []
        for region in unique_regions:
            print("Starting region %i..." % region)
            z_bin_paths = glob("%s/region=%i/z_bin=*" %
                               (self.parquet_pairs, region))
            region_mask = unkn_regions == region
            region_ids = unkn_ids[region_mask]
            region_sort = region_ids.argsort()
            region_weights = unkn_weights[region_mask][region_sort]
            region_ids = region_ids[region_sort]
            region_ave_weight = np.mean(region_weights)

            process_data = [
                {"unkn_ids": region_ids,
                 "unkn_weights": region_weights,
                 "tot_sample": len(region_ids),
                 "ave_unkn_weight": region_ave_weight,
                 "r_mins": self.r_mins,
                 "r_maxes": self.r_maxes,
                 "file_name": self.parquet_pairs,
                 "region": "region=%i" % region,
                 "z_bin": z_bin,
                 "weight_power": self.weight_power}
                for z_bin in z_bin_paths]
            if self.n_proc > 0:
                pool = Pool(self.n_proc)
                region_output = pool.imap_unordered(
                    collapse_pairs,
                    process_data,
                    chunksize=int(np.ceil(
                        0.5 * len(z_bin_paths) / self.n_proc)))
                pool.close()
                pool.join()
                output.extend(region_output)
            else:
                for data in process_data:
                    output.append(collapse_pairs(data))

        return pd.concat(output)


def collapse_pairs(data):
    """Collapse unknown pairs into a single estimate for each reference in
    a redshift bin in in the pair file.

    Parameters
    ----------
    data : `dict`
        Dictionary of data to process.
        Dictionary contains:

        ``"unkn_ids"``
            Integer ids of objects with unknown redshift.
            (`numpy.ndarray`, (N,))
        ``"unkn_weights"``
            Weights for each individual object with unknown redshift.
            (`numpy.ndarray`, (N,))
        ``"tot_sample"``
            Number of unknown objects in this region. (`int`)
        ``"r_mins"``
            Minimum distance in Mpc of the annuli. (`list` of `float`)
        ``"r_maxes"``
            Maximum distance in Mpc of the annuli. (`list` of `float`)
        ``"file_name"``
            Location of the parquet directory. (`str`)
        ``"z_bin"``
            Full path name to the parquet file, redshift bin to be run. (`str`)
        ``"weight_power"``
            Power law exponent to weight galaxies by distance. (`float`)

    Returns
    -------
    output : `pandas.DataFrame`
        Over-densities around each of the reference objects in this redshift
        file.
    """
    output = []
    r_mins = data["r_mins"]
    r_maxes = data["r_maxes"]
    reference_data = pd.read_parquet("%s/%s/reference_data.parquet" %
                                     (data["file_name"],
                                      data["region"]))
    reference_data.set_index("ref_id", inplace=True)

    pair_data = pd.read_parquet("%s" % data["z_bin"])
    pair_data.set_index("ref_id", inplace=True)
    unkn_ids = data["unkn_ids"]
    unkn_weights = data["unkn_weights"]

    for ref_id in pair_data.index.unique():
        output_row = collapse_pairs_ref_id(reference_data.loc[ref_id],
                                           pair_data.loc[ref_id],
                                           unkn_ids,
                                           unkn_weights,
                                           r_mins,
                                           r_maxes,
                                           data["weight_power"])
        output_row["ref_id"] = ref_id
        output_row["tot_sample"] = data["tot_sample"]
        output_row["ave_unkn_weight"] = data["ave_unkn_weight"]
        output.append(output_row)

    return pd.DataFrame(output)


def collapse_pairs_ref_id(ref_row,
                          pair_data,
                          unkn_ids,
                          unkn_weights,
                          r_mins,
                          r_maxes,
                          weight_power):
    """Collapse pairs around one reference object by masking in the ids of the
    input objects with unknown redshift.

    Parameters
    ----------
    ref_row : `pandas.DataFrame`
        DataFrame representing one reference object.
    pair_data : `pandas.DataFrame`
        Ids and distances for all the unknown objects around this reference.
    unkn_ids : `numpy.ndarray`
        Integer ids of unknown objects to mask into pair data.
    unkn_weights : `numpy.ndarray`
        Weights of unknown objects.
    r_mins : `list` of `float`
        Minimum distance for bin annuli
    r_maxes : `list` of `float`
        Maximum distance for bin annuli
    weight_power : `float`
        Power law slope to weight pair distances by.

    Returns
    -------
    output : `dict`
        Output values for a individual reference object.
        Dictionary containing:

        ``"redshift"``
            Redshift of the reference object. (`float`)
        ``"region"``
            Region this reference object belongs to. (`int`)
        ``"[scale_name]_counts"``
            Number of pairs around the reference within the annulus
            ``scale_name`` (`int`)
        ``"[scale_name]_weights"``
            Sum of weights of pairs around the reference within the annulus
            ``scale_name`` (`float`)
    """
    output = dict()
    output["redshift"] = ref_row["redshift"]
    output["region"] = ref_row["region"]
    for r_min, r_max in zip(r_mins, r_maxes):
        scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
        output["%s_counts" % scale_name] = 0
        output["%s_weights" % scale_name] = 0.0

    pair_dists = decompress_distances(pair_data["comp_log_dist"].to_numpy())
    pair_ids = pair_data["unkn_id"].to_numpy()

    (start_idx, end_idx) = find_trim_indexes(pair_ids, unkn_ids)
    if start_idx == end_idx:
        return output
    sub_unkn_ids = unkn_ids[start_idx: end_idx]
    sub_unkn_weights = unkn_weights[start_idx: end_idx]

    (start_idx, end_idx) = find_trim_indexes(sub_unkn_ids, pair_ids)
    if start_idx == end_idx:
        return output
    sub_pair_ids = pair_ids[start_idx: end_idx]
    sub_pair_dists = pair_dists[start_idx: end_idx]

    if len(sub_unkn_ids) > len(sub_pair_ids):
        matched_dists, matched_weights = find_pairs(
            sub_pair_ids, sub_unkn_ids, sub_pair_dists, sub_unkn_weights)
    else:
        matched_weights, matched_dists = find_pairs(
            sub_unkn_ids, sub_pair_ids, sub_unkn_weights, sub_pair_dists)

    matched_dist_weights = matched_weights * distance_weight(matched_dists,
                                                             weight_power)
    for r_min, r_max in zip(r_mins, r_maxes):
        scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
        dist_mask = np.logical_and(matched_dists >= r_min,
                                   matched_dists < r_max)
        output["%s_counts" % scale_name] = dist_mask.sum()
        output["%s_weights" % scale_name] = \
            matched_dist_weights[dist_mask].sum()
    return output


def find_trim_indexes(trim_to, trim_from):
    """Find indices of trim_to where it is fully contained in trim_from.

    Parameters
    ----------
    trim_to : `numpy.ndarray`
        Sorted integer array to find locations of the min and max value in
        ``trim_from``
    trim_from : `numpy.ndarray`
        Sorted integer array to trim to the min and max value of ``trim_to``.

    Returns
    -------
    output : `tuple`, (`int`, `int`)
        Min and max index where the values of trim_to will be fully contained
        within trim_from.
    """
    start_idx = np.searchsorted(trim_from, trim_to[0], side="left")
    end_idx = np.searchsorted(trim_from, trim_to[-1], side="right")
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(trim_from):
        end_idx = len(trim_from)
    return (start_idx, end_idx)


def find_pairs(input_ids, match_ids, input_weights, match_weights):
    """Find the values in input_ids within match_ids and mask the weight
    array's.

    Parameters
    ----------
    input_ids : `numpy.ndarray`
        Sorted integer array of ids to find in ``match_ids``
    match_ids : `numpy.ndarray`
        Stored integer array of ids.
    input_weights : `numpy.ndarray`
        Weights of each of the input ids.
    match_weights : `numpy.ndarray`
        Weights of each of the match ids.
    
    """
    sort_idxs = np.searchsorted(match_ids, input_ids)
    sort_mask = np.logical_and(sort_idxs < len(match_ids), sort_idxs >= 0)
    matched_mask = np.equal(match_ids[sort_idxs[sort_mask]],
                            input_ids[sort_mask])
    return (input_weights[sort_mask][matched_mask],
            match_weights[sort_idxs[sort_mask]][matched_mask])
