

import h5py
import numpy as np
import stomp


class RawPairFinder(object):
    
    """
    Main class for calculating and storing the indexes of nearby pairs for the 
    target and unknown samples. It handles both the real data and random
    samples in slightly different pair loops since we don't need to store
    indices for random samples.
    """
    
    def __init__(self, unknown_itree, target_vector, target_ids,
                 stomp_map):
        """
        Initialization for the pair finding software. Utilizes the STOMP
        sphereical pixelization library to find and store all close pairs into
        a HDF5 data file.
        args:
            unknown_itree: stomp IndexedTreeMap object containing the unknown
                object sample. The data structure is that of a searchable
                quad-tree
            target_vector: stomp CosmoVector object containing the target
                objects
            target_ids: numpy.array containing the index number of the target
                objects
            stomp_map: stomp Map object specifying the geometry of the survey
        """
        
        self._unknown_itree = unknown_itree
        self._target_vect = target_vector
        self._target_ids = target_ids
        self._region_ids = np.empty_like(target_ids, dtype = np.uint16)
        self._stomp_map = stomp_map
    
    def _reset_array_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the unknown sample.
        """   
        self._area_array = np.empty(self._target_vect.size())
        self._unmasked_array = np.empty(self._target_vect.size())
        self._bin_resolution = np.empty(self._target_vect.size(),
                                        dtype = np.uint32)
        self._pair_list = []
        self._pair_invdist_list = []
        for idx in xrange(self._target_vect.size()):
            self._pair_list.append([])
            self._pair_invdist_list.append([])
            
        return None
    
    def _reset_random_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the random sample following the geometry of the unknown
        sample.  
        """ 
        self._n_random_per_target = (np.ones_like(self._target_ids,
                                                  dtype = np.uint32) * -99)
        self._n_random_invdist_per_target = (
            np.ones_like(self._target_ids, dtype = np.float32) * -99)
        
        return None
            
    def find_pairs(self, min_scale, max_scale):
        """
        Main functionality of the RawPairFinder class. Given the input data,
        we find all the close pairs between the target, known redshift objects
        and the unknown redshift oobjects. Stores the raw pair indices and also
        the inverse distance weight.
        Args:
            min_scale: float value of the minimum scale to run the pair finder.
                Units are physical kpc
            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
        Returns:
           None
        """
        
        self._reset_array_data()
        
        radial_bin = stomp.RadialBin(min_scale/1000.0, max_scale/1000.0, 0.01)
        
        print("Finding real pairs...")
        
        for target_idx, target_obj in enumerate(self._target_vect):
            
            
            self._region_ids[target_idx] = self._stomp_map.FindRegion(
                target_obj)
            radial_bin.SetRedshift(target_obj.Redshift())
            max_ang = stomp.Cosmology.ProjectedAngle(target_obj.Redshift(),
                                                     max_scale / 1000.0)
            radial_bin.CalculateResolution(target_obj.Lambda() - max_ang,
                                           target_obj.Lambda() + max_ang)
            
            target_pix = stomp.Pixel(target_obj, radial_bin.Resolution())
            
            covering_pix_vect = stomp.PixelVector()
            target_pix.WithinAnnulus(target_obj, radial_bin.Resolution(),
                                     radial_bin, covering_pix_vect)
            
            self._bin_resolution[target_idx] = radial_bin.Resolution()
            
            unmasked_frac = 0
            area = 0
            
            for pix in covering_pix_vect:
                
                tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
                if tmp_unmasked <= 0.0:
                    continue
                
                unmasked_frac += tmp_unmasked
                area += tmp_unmasked * pix.Area(radial_bin.Resolution())
                dist = (target_pix.Ang()).AngularDistance(pix.Ang())
                
                tmp_i_ang_vect = stomp.IAngularVector()
                self._unknown_itree.Points(tmp_i_ang_vect, pix)
                for i_ang in tmp_i_ang_vect:
                    self._pair_list[target_idx].append(i_ang.Index())
                    self._pair_invdist_list[target_idx].append(
                        np.float32(1. / dist))
            self._area_array[target_idx] = area
            self._unmasked_array[target_idx] = unmasked_frac
            
        return None
            
    def random_loop(self, min_scale, max_scale, random_tree):
        """
        Function for computing and storing the number of random objects, created
        to follow the same geometry as the unknown sample, against of the target
        sample. Stores the raw number counts and also an inverse weighted number
        count.
        args:
            min_scale: float value of the minimum scale to run the pair finder.
                Units are physical kpc
            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
            n_randoms: int value factor of randoms to run. The total number
                randoms used will be n_randoms * (# unknown points)
        Returns:
            None
        """
        
        self._reset_random_data()
        
        self._n_random_points = random_tree.NPoints()
        
        radial_bin = stomp.RadialBin(min_scale/1000.0, max_scale/1000.0, 0.01)
        
        print("Finding random pairs...")
        
        for target_idx, target_obj in enumerate(self._target_vect):
            
            
            self._region_ids[target_idx] = self._stomp_map.FindRegion(
                target_obj)
            radial_bin.SetRedshift(target_obj.Redshift())
            max_ang = stomp.Cosmology.ProjectedAngle(target_obj.Redshift(),
                                                     max_scale / 1000.0)
            radial_bin.CalculateResolution(target_obj.Lambda() - max_ang,
                                                 target_obj.Lambda() + max_ang)
            
            target_pix = stomp.Pixel(target_obj, radial_bin.Resolution())
            
            covering_pix_vect = stomp.PixelVector()
            target_pix.WithinAnnulus(target_obj, radial_bin.Resolution(),
                                     radial_bin, covering_pix_vect)
            
            n_points = 0
            inv_dist = 0.0
            
            for pix in covering_pix_vect:
                
                tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
                if tmp_unmasked <= 0.0:
                    continue
                
                dist = (target_pix.Ang()).AngularDistance(pix.Ang())
                tmp_n_points = random_tree.NPoints(pix)
                n_points += tmp_n_points
                inv_dist += tmp_n_points / dist
            
            self._n_random_per_target[target_idx] = n_points
            self._n_random_invdist_per_target[target_idx] = np.float32(inv_dist)
            
        return None
    
    def write_to_hdf5(self, hdf5_file, scale_name):
        
        ### TODO:
        ###     Write more descriptive information about columns and settings
        ###     used in the run. Possibly write out all of the current arguments
        ###     in the Argparser object into the data file.
        
        """
        Method to write the raw pairs to an HDF5 file. These "pair files" are
        the heart of The-wiZZ and allow for quick computation and recomputation
        of clustering redshift recovery PDFs.
        Args:
            hdf5_file: Open HDF5 file object from h5py
            scale_name: Name of the specific scale that was run. This will end
                up being the name of the HDF5 group for the stored data.
        Returns:
            None
        """
        
        tmp_grp = hdf5_file.create_group('%s' % (scale_name))
        tmp_grp.attrs.create(
               'n_unknown', self._unknown_itree.NPoints())
        
        try:
            tmp_grp.attrs.create(
               'n_random_points', self._n_random_points)
        except AttributeError:
            pass
        
        for target_idx, target in enumerate(self._target_vect):
            tmp_target_grp = tmp_grp.create_group(
                '%i' % self._target_ids[target_idx])
            tmp_target_grp.create_dataset(
                'ids', data = np.array(self._pair_list[target_idx],
                                       dtype = np.uint32))
            tmp_target_grp.create_dataset(
                'inv_dist', data = np.array(self._pair_invdist_list[target_idx],
                                            dtype = np.float32))
            tmp_target_grp.attrs.create('redshift',
                                      target.Redshift())
            tmp_target_grp.attrs.create('unmasked_frac',
                                      self._unmasked_array[target_idx])
            tmp_target_grp.attrs.create('bin_resolution',
                                      self._bin_resolution[target_idx])
            tmp_target_grp.attrs.create('area', self._area_array[target_idx])
            tmp_target_grp.attrs.create('region', self._region_ids[target_idx])
            try:
                tmp_target_grp.attrs.create(
                    'rand', self._n_random_per_target[target_idx])
                tmp_target_grp.attrs.create(
                    'rand_inv_dist',
                    self._n_random_invdist_per_target[target_idx])
            except AttributeError:
                continue
            
        return None
            