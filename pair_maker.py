#! /usr/local/bin/python

"""This is the main program for running the pair finder and creating the data
file that contains the raw pair information between the reference and unknown
sample. It should be run with the complete photometric, unknown catalog of
interest to allow users to later subselect samples from this catalog. See
input_flags.py for a list of options or use --help from the command line.
"""

from __future__ import division, print_function, absolute_import

import sys

import numpy as np
import stomp

from the_wizz import core_utils
from the_wizz import stomp_utils
from the_wizz import input_flags
from the_wizz import pair_maker_utils

if __name__ == "__main__":
    print("")
    print("the-wizz has begun conjuring: running pair maker...")
    # Load the command line arguments.
    args = input_flags.parse_input_pair_args()
    input_flags.print_args(args)
    # Create the output hdf5 file where we will store the output for the
    # pair finding including raw pairs, area, unmasked fraction. We do this
    # first to soft fail rather than run through the code.
    # Load the stomp geometry coving the area of spectroscopic overlap.
    stomp_map = stomp.Map(args.stomp_map)
    # We request regionation for use with spatial bootstrapping. The
    # resolution found for regionation also sets the max resolution of the
    # different STOMP tree maps.
    stomp_map.InitializeRegions(args.n_regions)
    print("Created %i Regions at resolution %i..." %
          (stomp_map.NRegion(), stomp_map.RegionResolution()))
    # Load the sample with known redshifts.
    (reference_vector, reference_ids, reference_tree_map) = \
        stomp_utils.load_reference_sample(
            args.reference_sample_file, stomp_map, args)
    # Load the unknown sample from disc. Assumed data type is fits.
    unknown_itree = stomp_utils.load_unknown_sample(args.unknown_sample_file,
                                                    stomp_map, args)
    # We also wish to subtract a random sample from density estimate. This
    # function creates a set of uniform data points on the geometry of the
    # stomp map.
    random_tree = None
    if args.n_randoms > 0:
        random_tree = stomp_utils.create_random_data(
            args.n_randoms * unknown_itree.NPoints(), stomp_map)
    # Now that we have everything set up we can send our data off to the pair
    # finder.
    pair_finder = pair_maker_utils.RawPairFinder(
        unknown_itree, reference_vector, reference_ids, reference_tree_map,
        stomp_map, args.output_pair_hdf5_file, random_tree,
        create_hdf5_file=True, input_args=args)
    # We need to tell the pair finder what scale we would like to run over
    # before we begin.
    min_scale_list = args.min_scale.split(',')
    max_scale_list = args.max_scale.split(',')
    if len(min_scale_list) != len(max_scale_list):
        print("Number of min scales requested does not match number of max"
              "sales. Exitting.")
        sys.exit()
    print("Running", len(min_scale_list), "scales...")
    # Loop over the min/max scales requested.
    for min_scale, max_scale in zip(min_scale_list, max_scale_list):
        print("Running scale: %s to %s" % (min_scale, max_scale))
        # Pair finder does what it says. It also computes the areas, unmasked
        # fractions for each reference object.
        pair_finder.find_pairs(np.float_(min_scale), np.float_(max_scale))
    # Clean up the large items before we exit.
    del random_tree
    del unknown_itree
    del reference_vector, reference_ids, reference_tree_map
    del pair_finder
    print("Done!")
    # And that's it. We are done.
