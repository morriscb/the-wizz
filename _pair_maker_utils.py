

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
        self._region_area = np.empty(self._stomp_map.NRegion(),
                                     dtype = np.float32)
        for reg_idx in xrange(self._stomp_map.NRegion()):
            self._region_area[reg_idx] = self._stomp_map.RegionArea(reg_idx)
    
    def _reset_array_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the unknown sample.
        """   
        self._area_array = np.zeros(self._target_vect.size(),
                                    dtype = np.float32)
        self._unmasked_array = np.zeros(self._target_vect.size(),
                                        dtype = np.float32)
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
        self._n_random_per_target = np.zeros_like(self._target_ids,
                                                  dtype = np.uint32)
        self._n_random_invdist_per_target = np.zeros_like(self._target_ids,
                                                          dtype = np.float32)
        
        return None
    
    def _compute_unknown_weight(self, dist):
        """
        Convienence function for computing the weighting of an unknown object
        as a function of projected physical Mpc from the target object. This
        function is declared here for easy modification by intersted users.
        The normal behavior is inverse distance.
        Args:
            dist: float array of distances in projected physical Mpc
        Rreturns:
            float array of weights
        """
        return np.where(dist > 1e-8, 1. / dist, 1e8)
            
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
            
            for pix in covering_pix_vect:
                
                if pix.Resolution() < self._stomp_map.RegionResolution():
                    
                    pix_vect = stomp.PixelVector()
                    pix.SubPix(self._stomp_map.RegionResolution(),
                               pix_vect)
                    for sub_pix in pix_vect:
                        if (self._stomp_map.FindRegion(sub_pix) ==
                            self._region_ids[target_idx]):
                            self._store_target_unknown_pixel(
                                target_idx, target_obj, sub_pix)
                else:
                    if (self._stomp_map.FindRegion(pix) ==
                        self._region_ids[target_idx]):
                        self._store_target_unknown_pixel(
                            target_idx, target_obj, pix)
            
        return None
    
    def _store_target_unknown_pixel(self, target_idx, target_obj, pix):
        """
        Internal class function for finding the number of unknown objects and
        their ids in a single stomp pixel.
        Args:
            target_idx: int array index of the target object
            target_obj: stomp.CosmoCoordinate object containing the spatial and
                redshift information of the considered target object.
            pix: stomp.Pixel object to compute the number of randoms in.
        Returns:
            None
        """
        
        tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
        
        if tmp_unmasked <= 0.0:
            return None
        
        self._unmasked_array[target_idx] += tmp_unmasked
        self._area_array[target_idx] += (tmp_unmasked *
                                         pix.Area(pix.Resolution()))
        weight = self._compute_unknown_weight(
            target_obj.ProjectedRadius(pix.Ang()))
                
        tmp_i_ang_vect = stomp.IAngularVector()
        self._unknown_itree.Points(tmp_i_ang_vect, pix)
        for i_ang in tmp_i_ang_vect:
            self._pair_list[target_idx].append(i_ang.Index())
            self._pair_invdist_list[target_idx].append(
                np.float32(weight))
            
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
            
            for pix in covering_pix_vect:
                
                if pix.Resolution() < self._stomp_map.RegionResolution():
                    pix_vect = stomp.PixelVector()
                    pix.SubPix(self._stomp_map.RegionResolution(),
                               pix_vect)
                    for sub_pix in pix_vect:
                        if (self._stomp_map.FindRegion(sub_pix) ==
                            self._region_ids[target_idx]):
                            self._store_target_random_pixel(
                                target_idx, target_obj, sub_pix,
                                random_tree)
                else:
                    if (self._stomp_map.FindRegion(pix) ==
                        self._region_ids[target_idx]):
                        self._store_target_random_pixel(
                            target_idx, target_obj, pix,
                            random_tree)
            
        return None
    
    def _store_target_random_pixel(self, target_idx, target_obj, pix,
                                   random_tree):
        """
        Internal class function for finding the number of randoms in a single
        stomp pixel.
        Args:
            target_idx: int array index of the target object
            target_obj: stomp.CosmoCoordinate object containing the spatial and
                redshift information of the considered target object.
            pix: stomp.Pixel object to compute the number of randoms in.
        Returns:
            None
        """
        tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
        if tmp_unmasked <= 0.0:
            return None
        
        weight = self._compute_unknown_weight(
            target_obj.ProjectedRadius(pix.Ang()))
        tmp_n_points = random_tree.NPoints(pix)
        self._n_random_per_target[target_idx] += tmp_n_points
        self._n_random_invdist_per_target[target_idx] += np.float32(
            tmp_n_points * weight)
        
        return None
    
    def write_to_hdf5(self, hdf5_file, scale_name):
        
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
        
        tmp_grp.attrs.create('area', self._stomp_map.Area())
        tmp_grp.attrs.create('n_region', self._stomp_map.NRegion())
        tmp_grp.attrs.create('region_area',
                             self._region_area)
        
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
            
            sort_args = np.argsort(self._pair_list[target_idx])
            
            tmp_target_grp.create_dataset(
                'ids', data = np.array(self._pair_list[target_idx],
                                       dtype = np.uint32)[sort_args],
                maxshape = (None,), compression = 'lzf', shuffle = True)
            tmp_target_grp.create_dataset(
                'inv_dist', data = np.array(self._pair_invdist_list[target_idx],
                                            dtype = np.float32)[sort_args],
                maxshape = (None,), compression = 'lzf', shuffle = True)
            
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
    

class TargetPairFinder(RawPairFinder):
    
    """
    Derived class for calculating and storing the over-density of the target 
    objects around themselves. It handles both the real data and random
    samples in slightly different pair loops since we don't need to store
    indices for random samples.
    """
    
    def _reset_array_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the target sample.
        """
        
        self._area_array = np.zeros(self._target_vect.size(),
                                    dtype = np.float32)
        self._unmasked_array = np.zeros(self._target_vect.size(),
                                        dtype = np.float32)
        self._bin_resolution = np.zeros(self._target_vect.size(),
                                        dtype = np.uint32)
        self._redshift_array = np.empty(self._target_vect.size(),
                                        dtype = np.float32)
        
        self._pair_array = np.zeros(self._target_vect.size(),
                                    dtype = np.float32)
        
        return None
    
    def _reset_random_data(self):
        """
        Utility function that creates/resets the interal data storage of the 
        class for the random sample following the geometry of the unknown
        sample.  
        """ 
        self._n_random_per_target = np.zeros_like(self._target_ids,
                                                  dtype = np.uint32)
        
        return None
    
    def find_pairs(self, max_scale):
        """
        Main functionality of the TargetPairFinder class. Given the input data,
        we compute the number of target objects within a fixed angular scale
        around themselves (aka an auto-correlation).
        
        Args:

            max_scale: float value of the maximum scale to run the pair finder.
                Units are physical kpc
        Returns:
           None
        """
        
        self._reset_array_data()
        
        radial_bin = stomp.RadialBin(1e-8, max_scale/1000.0, 0.01)
        
        print("Finding real pairs...")
        
        max_dist_sq = (max_scale / 1000.0) ** 2
        
        for target_idx, target_obj in enumerate(self._target_vect):
            
            self._region_ids[target_idx] = self._stomp_map.FindRegion(
                target_obj)
            radial_bin.SetRedshift(target_obj.Redshift())
            max_ang = stomp.Cosmology.ProjectedAngle(target_obj.Redshift(),
                                                     max_scale / 1000.0)
            radial_bin.CalculateResolution(target_obj.Lambda() - max_ang,
                                           target_obj.Lambda() + max_ang)
            
            self._redshift_array[target_idx] = target_obj.Redshift()
            target_pix = stomp.Pixel(target_obj, radial_bin.Resolution())
            target_dist = stomp.Cosmology_ComovingDistance(
                target_obj.Redshift())
            
            covering_pix_vect = stomp.PixelVector()
            target_pix.WithinRadius(max_ang, covering_pix_vect)
            
            self._bin_resolution[target_idx] = radial_bin.Resolution()
            
            for pix in covering_pix_vect:
                
                if pix.Resolution() < self._stomp_map.RegionResolution():
                    pix_vect = stomp.PixelVector()
                    pix.SubPix(self._stomp_map.RegionResolution(),
                               pix_vect)
                    for sub_pix in pix_vect:
                        if (self._stomp_map.FindRegion(sub_pix) ==
                            self._region_ids[target_idx]):
                            self._store_target_unknown_pixel(
                                target_idx, target_obj, target_dist,
                                max_dist_sq, sub_pix)
                else:
                    if (self._stomp_map.FindRegion(pix) ==
                        self._region_ids[target_idx]):
                        self._store_target_unknown_pixel(
                            target_idx, target_obj, target_dist,
                            max_dist_sq, pix)
            
        return None
    
    def _store_target_unknown_pixel(self, target_idx, target_obj, target_dist,
                                    max_dist_sq, pix):
        """
        Internal class function for finding the number of unknown objects and
        their ids in a single stomp pixel.
        Args:
            target_idx: int array index of the target object
            target_obj: stomp.CosmoCoordinate object containing the spatial and
                redshift information of the considered target object.
            pix: stomp.Pixel object to compute the number of randoms in.
        Returns:
            None
        """
        
        tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
        
        if tmp_unmasked <= 0.0:
            return None
        
        self._unmasked_array[target_idx] += tmp_unmasked
        self._area_array[target_idx] += (tmp_unmasked *
                                         pix.Area(pix.Resolution()))
        proj_dist_sq = target_obj.ProjectedRadius(pix.Ang()) ** 2
                
        tmp_cos_ang_vect = stomp.CosmoVector()
        self._unknown_itree.Points(tmp_cos_ang_vect, pix)
        for cos_ang in tmp_cos_ang_vect:
            dist_sq = (
                (target_dist -
                 stomp.Cosmology_ComovingDistance(cos_ang.Redshift())) ** 2 +
                proj_dist_sq)
            if dist_sq > max_dist_sq or dist_sq < 1e-16:
                return None
            self._pair_array[target_idx] += 1.0
            
        return None
    
    def random_loop(self, max_scale, random_tree):
        
        self._reset_random_data()
        
        self._n_random_points = random_tree.NPoints()
        
        radial_bin = stomp.RadialBin(1e-8, max_scale/1000.0, 0.01)
        
        max_dist_sq = (max_scale / 1000.0) ** 2
        
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
            target_dist = stomp.Cosmology_ComovingDistance(
                target_obj.Redshift())
            
            covering_pix_vect = stomp.PixelVector()
            target_pix.WithinRadius(max_ang, covering_pix_vect)
            
            for pix in covering_pix_vect:
                
                if pix.Resolution() < self._stomp_map.RegionResolution():
                    pix_vect = stomp.PixelVector()
                    pix.SubPix(self._stomp_map.RegionResolution(),
                               pix_vect)
                    for sub_pix in pix_vect:
                        if (self._stomp_map.FindRegion(sub_pix) ==
                            self._region_ids[target_idx]):
                            self._store_target_random_pixel(
                                target_idx, target_obj, target_dist,
                                max_dist_sq, sub_pix, random_tree)
                else:
                    if (self._stomp_map.FindRegion(pix) ==
                        self._region_ids[target_idx]):
                        self._store_target_random_pixel(
                            target_idx, target_obj, target_dist, max_dist_sq,
                            pix,random_tree)
            
        return None
    
    def _store_target_random_pixel(self, target_idx, target_obj, target_dist,
                                   max_dist_sq, pix, random_tree):
        """
        Internal class function for finding the number of randoms in a single
        stomp pixel.
        Args:
            target_idx: int array index of the target object
            target_obj: stomp.CosmoCoordinate object containing the spatial and
                redshift information of the considered target object.
            pix: stomp.Pixel object to compute the number of randoms in.
        Returns: 
            None
        """
        
        tmp_unmasked = self._stomp_map.FindUnmaskedFraction(pix)
        if tmp_unmasked <= 0.0:
            return None
        
        proj_dist_sq = target_obj.ProjectedRadius(pix.Ang()) ** 2
        
        tmp_cos_ang_vect = stomp.CosmoVector()
        random_tree.Points(tmp_cos_ang_vect, pix)
        for cos_ang in tmp_cos_ang_vect:
            dist_sq = (
                (target_dist -
                 stomp.Cosmology_ComovingDistance(cos_ang.Redshift())) ** 2 +
                proj_dist_sq)
            if dist_sq > max_dist_sq or dist_sq < 1e-16:
                return None
            self._n_random_per_target[target_idx] += 1.0
        
        return None
    
    def compute_target_overdensity(self):
        
        self._target_overdensity = np.where(
            self._n_random_per_target > 0,
            self._pair_array / self._n_random_per_target - 1.0,
            0.0)
        
        return None
    
    def write_to_hdf5(self, hdf5_file, group_name):
        
        """
        Method to write the over-density and raw number counts. These "pair files" are
        the heart of The-wiZZ and allow for quick computation and recomputation
        of clustering redshift recovery PDFs.
        Args:
            hdf5_file: Open HDF5 file object from h5py
            scale_name: Name of the specific scale that was run. This will end
                up being the name of the HDF5 group for the stored data.
        Returns:
            None
        """
        
        tmp_grp = hdf5_file.create_group('%s' % (group_name))
        
        tmp_grp.attrs.create('area', self._stomp_map.Area())
        tmp_grp.attrs.create('n_region', self._stomp_map.NRegion())
        tmp_grp.attrs.create('region_area',
                             self._region_area)
        
        try:
            tmp_grp.attrs.create(
               'n_random_points', self._n_random_points)
        except AttributeError:
            pass
        
        tmp_grp.create_dataset('n_target', data = self._pair_array,
                               compression = 'lzf')
        tmp_grp.create_dataset('redshift', data = self._redshift_array,
                               compression = 'lzf')
        tmp_grp.create_dataset('unmasked_frac', data = self._unmasked_array,
                               compression = 'lzf')
        tmp_grp.create_dataset('bin_resolution', self._bin_resolution,
                               compression = 'lzf')
        tmp_grp.create_dataset('area', self._bin_resolution,
                               compression = 'lzf')
        tmp_grp.create_dataset('region', self._bin_resolution,
                               compression = 'lzf')
        
        try:
            tmp_grp.create_dataset('n_random', data = self._n_random_per_target,
                                   compression = 'lzf')
            tmp_grp.create_dataset('ovrden', data = self._target_overdensity,
                                   compression = 'lzf')
        except AttributeError:
            pass
            
        return None
            