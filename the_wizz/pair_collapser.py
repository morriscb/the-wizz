
from multiprocessing import Pool
import numpy as np
import pandas as pd

from .pair_maker import decompress, distance_weight


def collapse_pairs(data):
    """
    """
    output = {"id": ,
              "redshift": ,
              "region": ,}
    r_mins = data["r_mins"]
    r_maxes = data["r_maxes"]
    pair_data = pd.read_parquet("%s/region=%i/ref_id=%i" % 
                                (data["file_name"],
                                 data["region"],
                                 data["ref_id"]))
    unkn_ids = data["unkn_ids"]
    unkn_weights = data["unkn_weights"]

    pair_dists = decompress(pair_data["comp_log_dists"])
    pair_dist_mask = np.logical_and(pair_dists >= np.min(r_mins),
                                    pair_dists < np.max(r_maxes))
    pair_ids = pair_data["ids"][pair_dist_mask]
    pair_dists = pair_dists[pair_dist_mask]

    (start_idx, end_idx) = find_trim_indexes(pair_ids, unkn_ids)
    if start_idx == end_idx:
        for r_min, r_max in zip(r_mins, r_maxes):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
            output["%s_counts" % scale_name] = 0
            output["%s_weights" % scale_name] = 0.0
        return output
    sub_unkn_ids = unkn_ids[start_idx: end_idx]
    sub_unkn_weights = unkn_weights[start_idx: end_idx]

    (start_idx, end_idx) = find_trim_indexes(sub_unkn_ids, pair_ids)
    if start_idx == end_idx:
        for r_min, r_max in zip(r_mins, r_maxes):
            scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
            output["%s_counts" % scale_name] = 0
            output["%s_weights" % scale_name] = 0.0
        return output
    sub_pair_ids = pair_ids[start_idx: end_idx]
    sub_pair_dists = pair_dists[start_idx: end_idx]

    if len(sub_unkn_ids) > len(sub_pair_ids):
        matched_dists, matched_weights = find_pairs(
            sub_pair_ids, sub_unkn_ids, sub_pair_dists, sub_unkn_weights)
    else:
        matched_weights, matched_dists = find_pairs(
            sub_unkn_ids, sub_pair_ids, sub_unkn_weights, sub_pair_dists)
    for r_min, r_max in zip(r_mins, r_maxes):
        scale_name = "Mpc%.2ft%.2f" % (r_min, r_max)
        dist_mask = np.logical_and(matched_dists >= r_min,
                                   matched_dists < r_max)
        output["%s_counts" % scale_name] = dist_mask.sum()
        output["%s_weights" % scale_name] = 
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
