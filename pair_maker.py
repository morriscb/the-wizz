
import _core_utils
import _stomp_utils
import input_flags
import _pair_maker_utils
import h5py
import numpy as np
import stomp
import sys


"""
This is the main program for running the pair finder and creating the data file
that contains the raw pair information between the target and unknown sample.
"""


if __name__ == "__main__":
    
    print("")
    print("The wiZZ has begun conjuring!")
    
    ### load the command line arguments 
    args = input_flags.parse_input_pair_args()
    
    ### load the stomp geometry coving the area of spectroscopic overlap
    ### TODO:
    ###     create an external program that allows for the creation of an
    ###     addaptive map. This will require some testing and visualization
    ###     software for Python. Low priority, maybe at ver1.0. Easy fix would
    ###     be a test of min, max RA, DEC and testing the area of that and how
    ###     much area remains after that. User can set a threshold based on how
    ###     much area coverage they know there should be.
    stomp_map = stomp.Map(args.stomp_map)
    ### We request regionation for use with spatial bootstrapping. The
    ### resolution found for regionation also sets the 
    stomp_map.InitializeRegions(args.n_regions)
    
    ### load the sample with known redshifts 
    target_vector, target_ids = _stomp_utils.load_target_sample(
        args.target_sample_file, stomp_map, args)
    
    ### load the unknown sample from disc. Assumed data type is fits though hdf5
    ### will be allowed in later versions
    unknown_itree = _stomp_utils.load_unknown_sample(args.unknown_sample_file,
                                                    stomp_map, args)
    
    ### We also wish to subtract a random sample from density estimate. This
    ### function creates a set of uniform data points on the geometry of the
    ### stomp map.
    if args.n_randoms > 0:
        random_tree = _stomp_utils.create_random_data(
            args.n_randoms * unknown_itree.NPoints(), stomp_map)
    
    ### now create the output hdf5 file where we will store the output for the
    ### pair finding including raw pairs, area, unmasked fraction.
    output_pair_hdf5_file = _core_utils.create_hdf5_file(
        args.output_pair_hdf5_file, args)
    
    ### now that we have everything set up we can send our data off to the pair
    ### finder
    pair_finder = _pair_maker_utils.RawPairFinder(unknown_itree, target_vector,
                                                  target_ids, stomp_map)
    
    ### We need to tell the pair finder what scale we would like to run over
    ### before we begin. 
    min_scale_list = args.min_scale.split(',')
    max_scale_list = args.max_scale.split(',')
    
    print min_scale_list
    print max_scale_list
    
    if len(min_scale_list) != len(max_scale_list):
        print("Number of min scales requested does not match number of max"
              "sales. Exitting.")
        sys.exit()
    
    ### loop over the min/max scales requested.
    for scale_idx in xrange(len(min_scale_list)):
        print("Running scale: %s to %s" % (min_scale_list[scale_idx],
                                           max_scale_list[scale_idx]))
        ### Pair finder does what it says. It also computes the areas, unmasked
        ### fractions for target object.
        pair_finder.find_pairs(np.float_(min_scale_list[scale_idx]),
                               np.float_(max_scale_list[scale_idx]))
        
        ### This is an optional part of the pair finder. It takes as an argument
        ### the a stomp tree map containing uniform random points as generated
        ### by the stomp. For non-uniform randoms it is recommened that the
        ### user run this software with said randoms as the unknown sample.
        if args.n_randoms > 0: 
            pair_finder.random_loop(np.float_(min_scale_list[scale_idx]),
                                    np.float_(max_scale_list[scale_idx]),
                                    random_tree)
        
        ### Now that we have all of our pairs found and in memory, we want to
        ### write them to something more permanent.
        pair_finder.write_to_hdf5(output_pair_hdf5_file,
                                  'kpc%st%s' % (min_scale_list[scale_idx],
                                                max_scale_list[scale_idx]))
    
    output_pair_hdf5_file.close()
    ### And that's it. We are done.
    