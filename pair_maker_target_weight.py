
import _core_utils
import _stomp_utils
import input_flags
import _pair_maker_utils
import h5py
import numpy as np
import stomp
import sys


"""
This program is used to calculate a relative weight for each spectroscpic
object. Dense structures in the target dataset can cause undue bias in the
resultant recovery and even cause negagive correlations for samples that anti-
correlation with said structures. This code attempts to mitigate this by
computing the over-density of spectroscopic objects relative to themseleves and
using this as a weight for supressing the effect of over-dense structures on
the recovery. This may even help with surveys with strange masks.
"""


if __name__ == "__main__":
    
    print("")
    print("The-wiZZ has begun conjuring: running pair maker target weight...")
    
    ### load the command line arguments 
    args = input_flags.parse_input_pair_args()
    
    input_flags.print_args(args)
    
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
    print("Created %i Regions at resolution %i..." %
          (stomp_map.NRegion(), stomp_map.RegionResolution()))
    
    ### load the sample with known redshifts 
    target_vector, target_ids = _stomp_utils.load_target_sample(
        args.target_sample_file, stomp_map, args)
    
    ### Instead of loading an unknown sample from disk, we load the target
    ### objects themselves as the unknown sample.
    unknown_itree = stomp.IndexedTreeMap(stomp_map.RegionResolution(), 200)
    for target_idx, target_obj in enumerate(target_vector):
        tmp_i_ang = stomp.IndexedAngularCoordinate(
            target_obj.UnitSphereX(), target_obj.UnitSphereY(),
            target_obj.UnitSphereZ(), target_idx)
        unknown_itree.AddPoint(tmp_i_ang)
    
    ### We also wish to subtract a random sample from density estimate. This
    ### function creates a set of uniform data points on the geometry of the
    ### stomp map.
    if args.n_randoms > 0:
        random_tree = _stomp_utils.create_random_data(
            args.n_randoms * unknown_itree.NPoints(), stomp_map)
    
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
        
    over_density_list = []
    ### loop over the min/max scales requested.
    output_file = _core_utils.create_ascii_file(args.output_hdf5_file, args)
    for scale_idx in xrange(len(min_scale_list)):
        output_file.writelines('#type%i = kpc%st%s\n' %
                               (scale_idx,
                                min_scale_list[scale_idx],
                                max_scale_list[scale_idx]))
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
            
        over_density_array = np.empty(len(pair_finder._pair_list))
        for target_idx, target_id_list in enumerate(pair_finder._pair_list):
            
            over_density_array[target_idx] = (
                len(target_id_list) * args.n_randoms /
                pair_finder._n_random_per_target[target_idx])
        over_density_list.append(over_density_array)
    
    for target_idx in xrange(target_vector.size()):
        line = ''
        for scale_idx in xrange(len(over_density_list)):
            line += '%.8e ' % over_density_list[scale_idx][target_idx]
        output_file.writelines(line + '\n')
    output_file.close()
    ### And that's it. We are done.
    