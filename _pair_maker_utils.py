

import __core__
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
        class for the unknown sample
        """   
        self._area_array = np.empty(self._target_vect.size())
        self._unmasked_array = np.empty(self._target_vect.size())
        self._bin_resolution = np.empty(self._target_vect.size(),
                                        dtype = np.uint32)
        self._pair_list = []
        self._pair_dist_list = []
        for idx in xrange(self._target_vect.size()):
            self._pair_list.append([])
            self._pair_dist_list.append([])
            
        return None
    
    def _reset_random_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the random sample following the geometry of the unknown
        sample.
            
        """ 
        self._n_random_per_target = (np.ones_like(self._target_ids,
                                                  dtype = np.uint32) * -99)
        
        return None
            
    def find_pairs(self, min_scale, max_scale):
        """
        args:
            min_scale: float value of the minimum scale to run the pair finder.
                Units are physical kpc
            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
            n_randoms: int value factor of randoms to run. The total number
                randoms used will be n_randoms * (# unknown points)
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
                
                tmp_i_ang_vect = stomp.IAngularVector()
                self._unknown_itree.Points(tmp_i_ang_vect, pix)
                for i_ang in tmp_i_ang_vect:
                    self._pair_list[target_idx].append(i_ang.Index())
                    self._pair_dist_list[target_idx].append(
                        i_ang.AngularDistance(target_pix.Ang()))
            self._area_array[target_idx] = area
            self._unmasked_array[target_idx] = unmasked_frac
            
    def random_loop(self, min_scale, max_scale, random_tree):
        """
        args:
            min_scale: float value of the minimum scale to run the pair finder.
                Units are physical kpc
            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
            n_randoms: int value factor of randoms to run. The total number
                randoms used will be n_randoms * (# unknown points)
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
            
            for pix in covering_pix_vect:
                
                tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
                if tmp_unmasked <= 0.0:
                    continue
                
                n_points += random_tree.NPoints(pix)
            
            self._n_random_per_target[target_idx] = n_points
            
        return None
    
    def write_to_hdf5(self, hdf5_file, scale_name):
        """
        
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
            
            tmp_data_set = tmp_grp.create_dataset(
                '%i' % self._target_ids[target_idx],
                data = np.array(self._pair_list[target_idx],
                                dtype = np.uint32))
            tmp_data_set.attrs.create('redshift',
                                      target.Redshift())
            tmp_data_set.attrs.create('unmasked_frac',
                                      self._unmasked_array[target_idx])
            tmp_data_set.attrs.create('bin_resolution',
                                      self._bin_resolution[target_idx])
            tmp_data_set.attrs.create('area', self._area_array[target_idx])
            tmp_data_set.attrs.create('region', self._region_ids[target_idx])
            try:
                tmp_data_set.attrs.create(
                    'rand', self._n_random_per_target[target_idx])
            except AttributeError:
                continue
        return None
            