
from glob import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd

from .pair_maker import decompress_distances, distance_weight


class PairCollapser:
    """
    """

    def __init__(self,
                 parquet_pairs,
                 r_mins,
                 r_maxes,
                 z_min=0.01,
                 z_max=5.00,
                 weight_power=-0.8,
                 n_proc=0):
        self.parquet_pairs = parquet_pairs
        self.r_mins = r_mins
        self.r_maxes = r_maxes
        self.z_min = z_min
        self.z_max = z_max
        self.weight_power = weight_power
        self.n_proc = n_proc

    def run(self,
            unknown_catalog):
        """
        """
        unkn_ids = unknown_catalog["id"]
        try:
            unkn_regions = unknown_catalog["region"]
        except KeyError:
            unkn_regions = np.zeros(len(unkn_ids), dtype=np.uint32)
        try:
            unkn_weights = unknown_catalog["weights"]
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

            process_data = [
                {"unkn_ids": region_ids,
                 "unkn_weights": region_weights,
                 "tot_sample": len(region_ids),
                 "r_mins": self.r_mins,
                 "r_maxes": self.r_maxes,
                 "file_name": self.parquet_pairs,
                 "region": "region=%i" % region,
                 "z_bin": z_bin}
                for z_bin in z_bin_paths]
            if self.n_proc > 0:
                with Pool(self.n_proc) as pool:
                    region_output = pool.imap_unordered(collapse_pairs,
                                                        process_data)
                output.extend(region_output)
            else:
                for data in process_data:
                    output.append(collapse_pairs(data))

        return pd.concat(output)


def collapse_pairs(data):
    """
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
                                           r_maxes)
        output_row["ref_id"] = ref_id
        output_row["tot_sample"] = data["tot_sample"]
        output.append(output_row)

    return pd.DataFrame(output)


def collapse_pairs_ref_id(ref_row,
                          pair_data,
                          unkn_ids,
                          unkn_weights,
                          r_mins,
                          r_maxes):
    """
    """
    output = dict()
    output["redshift"] = ref_row["redshift"]
    output["region"] = ref_row["region"]
    for r_min, r_max in zip(r_mins, r_maxes):
        scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
        output["%s_counts" % scale_name] = 0
        output["%s_weights" % scale_name] = 0.0

    pair_dists = decompress_distances(pair_data["comp_log_dist"].to_numpy())
    pair_ids = pair_data["unkn_ids"].to_numpy()

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

    matched_dist_weights = matched_weights * distance_weight(matched_dists)
    for r_min, r_max in zip(r_mins, r_maxes):
        scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
        dist_mask = np.logical_and(matched_dists >= r_min,
                                   matched_dists < r_max)
        output["%s_counts" % scale_name] = dist_mask.sum()
        output["%s_weights" % scale_name] = \
            matched_dist_weights[dist_mask].sum()
    return output


def find_trim_indexes(trim_to, trim_from):
    """
    """
    start_idx = np.searchsorted(trim_from, trim_to[0], side="left")
    end_idx = np.searchsorted(trim_from, trim_to[-1], side="right")
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(trim_from):
        end_idx = len(trim_from)
    return (start_idx, end_idx)


def find_pairs(input_ids, match_ids, input_weights, match_weights):
    """
    """
    sort_idxs = np.searchsorted(match_ids, input_ids)
    sort_mask = np.logical_and(sort_idxs < len(match_ids), sort_idxs >= 0)
    matched_mask = np.equal(match_ids[sort_idxs[sort_mask]],
                            input_ids[sort_mask])
    return (input_weights[sort_mask][matched_mask],
            match_weights[sort_idxs[sort_mask]][matched_mask])
