
import argparse
from copy import copy
import corner
from astropy.cosmology import WMAP5
import emcee
import matplotlib.pyplot as pl
import numpy as np
import pickle
import subprocess
import time
import sys

from glob import glob
from scipy.integrate import romberg

cb_set1 = np.array([(228,26,28),(55,126,184),(77,175,74),
                       (152,78,163),(255,127,0),(255,255,51),
                       (166,86,40),(247,129,191),(153,153,153)])/255.0
cb_dark2 = np.array([[27, 158, 119], [217, 95, 2], [117, 112, 179],
                        [231, 41, 138], [102, 166, 30], [230, 171, 2],
                        [166, 118, 29], [102, 102, 102]])/255.0

verbose = False

def compute_pz_norm(phi, z_min = 0.01, z_max = 3.0):
    
    tmp_phi = copy(phi)
    while np.any(tmp_phi[:,1] < 0.0) and len(tmp_phi) > 1:
        min_arg = tmp_phi[:,1].argmin()
        if min_arg == 0:
            new_phi = np.zeros((tmp_phi.shape[0] - 1, tmp_phi.shape[1]))
            new_phi[0,0] = (
                np.sum(tmp_phi[:2, 0] / tmp_phi[:2, 2] ** 2) /
                np.sum(1.0 / new_phi[:2, 2] ** 2))
            new_phi[0,1] = (
                np.sum(tmp_phi[:2, 1] / tmp_phi[:2, 2] ** 2) /
                np.sum(1.0 / tmp_phi[:2, 2] ** 2))
            new_phi[0,2] = np.sqrt(
                1.0 / np.sum(1.0 / tmp_phi[:2, 2] ** 2))
            new_phi[1:, :] = tmp_phi[2:, :]
        elif min_arg == tmp_phi.shape[0] - 1:
            new_phi = np.zeros((tmp_phi.shape[0] - 1, tmp_phi.shape[1]))
            new_phi[-1,0] = (
                np.sum(tmp_phi[-2:, 0] / tmp_phi[-2:, 2] ** 2) /
                np.sum(1.0 / tmp_phi[-2:, 2] ** 2))
            new_phi[-1,1] = (
                np.sum(tmp_phi[-2:, 1] / tmp_phi[-2:, 2] ** 2) /
                np.sum(1.0 / tmp_phi[-2:, 2] ** 2))
            new_phi[-1,2] = np.sqrt(
                1.0 / np.sum(1.0 / tmp_phi[-2:, 2] ** 2))
            new_phi[:-1, :] = tmp_phi[:-2, :]
        else:
            new_phi = np.zeros((tmp_phi.shape[0] - 2, tmp_phi.shape[1]))
            new_phi[min_arg - 1, 0] = (
                np.sum(tmp_phi[min_arg - 1:min_arg + 2, 0] /
                       tmp_phi[min_arg - 1:min_arg + 2, 2] ** 2) /
                np.sum(1.0 / tmp_phi[min_arg - 1:min_arg + 2, 2] ** 2))
            new_phi[min_arg - 1, 1] = (
                np.sum(tmp_phi[min_arg - 1:min_arg + 2, 1] /
                        tmp_phi[min_arg - 1:min_arg + 2, 2] ** 2) /
                np.sum(1.0 / tmp_phi[min_arg - 1:min_arg + 2, 2] ** 2))
            new_phi[min_arg - 1, 2] = np.sqrt(
                1.0 / np.sum(1.0 / tmp_phi[min_arg - 1:min_arg + 2, 2] ** 2))
            new_phi[:min_arg - 1, :] = tmp_phi[:min_arg - 1, :]
            new_phi[min_arg:, :] = tmp_phi[min_arg + 2:, :]
        tmp_phi = new_phi
        
    int_phi = np.append(0.0, np.append(tmp_phi[:,1], 0.0))
    int_z = np.append(z_min, np.append(tmp_phi[:,0], z_max))
    norm = np.trapz(int_phi, int_z)
    if norm < 0.0:
        print("WARNING: Normalization still negative. %.8e" % norm)
        norm = (np.sum(np.where(phi[:,1] >= 0, phi[:,1] / phi[:,2] ** 2, 0.0)) /
                np.sum(1.0 / phi[:,2] ** 2) * (z_max - z_min))
        
    return norm


class MagLimFitter(object):
    
    def __init__(self, phi, phi_cov, weights, x0, z_min, z_max, n_z_bins):
        
        self._phi = phi
        self._phi_cov = phi_cov
        print(self._phi.shape)
        print(self._phi_cov.shape)
        
        self._hold_phi = copy(self._phi)
        self._hold_phi_cov = copy(self._phi_cov)
        
        self._z_min = z_min
        self._z_max = z_max
        self._n_z_bins = n_z_bins
        self._n_bins = self._phi.shape[0] / n_z_bins
        self._weights = np.ones(self._phi.shape[0])
        for bin_idx in xrange(self._n_bins ):
            self._weights[bin_idx * n_z_bins:
                          (bin_idx + 1) * n_z_bins] = weights[bin_idx]
        
        self._pz_norm = np.ones(self._phi.shape[0])
        for bin_idx in xrange(self._n_bins):
            self._pz_norm[bin_idx * self._n_z_bins:
                          (bin_idx + 1) * self._n_z_bins] *= (
                self._weights[bin_idx] /
                compute_pz_norm(self._phi[bin_idx * self._n_z_bins:
                                          (bin_idx + 1) * self._n_z_bins, :]))
        self._phi[:,1] *= self._pz_norm
        self._phi[:,2] *= self._pz_norm
        self._phi_cov *= np.outer(self._pz_norm, self._pz_norm)
        
        self._I_matrix = np.zeros((self._n_z_bins, self._n_z_bins),
                                    np.float_)
        for diag in xrange(self._n_z_bins):
            self._I_matrix[diag, diag] = 1
        self._trans_matrix = copy(self._I_matrix)
        for bin_idx in xrange(self._n_bins - 2):
            self._trans_matrix = np.concatenate(
                (self._trans_matrix, self._I_matrix), axis = 1)
        self._trans_matrix = np.concatenate(
            (self._trans_matrix, -1. * self._I_matrix), axis = 1)
        
        print self._trans_matrix.shape
        print self._trans_matrix
        
        self._n_parm = len(x0)
        self.set_fit(x0)
        
        print(self._phi.shape)
    
    def bias(self, z):
        return self._raw_bias(z)
    
    def set_fit(self, x):
        self._x = x
        self._raw_bias = np.poly1d(x)
        
    def set_error(self, err):
        self._err = err
    
    def _fix_bias(self):
        bias_array = self.bias(self._hold_phi[:,0])
        self._phi[:,1] = self._hold_phi[:,1] / bias_array
        self._phi[:,2] = self._hold_phi[:,2] / bias_array
        self._phi_cov = self._hold_phi_cov / np.outer(bias_array, bias_array)
        
        self._pz_norm = np.ones(self._phi.shape[0])
        for bin_idx in xrange(self._n_bins):
            self._pz_norm[bin_idx * self._n_z_bins:
                          (bin_idx + 1) * self._n_z_bins] *= (
                1.0 /
                compute_pz_norm(self._phi[bin_idx * self._n_z_bins:
                                          (bin_idx + 1) * self._n_z_bins, :]))
        self._phi[:,1] *= self._pz_norm
        self._phi[:,2] *= self._pz_norm
        self._phi_cov *= np.outer(self._pz_norm, self._pz_norm)
        
        self._tmp_phi = self._phi[:,1] * self._weights
        self._tmp_phi_cov = self._phi_cov * np.outer(self._weights,
                                                     self._weights)
        
        sum_phi = np.dot(
            self._trans_matrix[:,:(self._n_bins - 1) * self._n_z_bins],
            self._tmp_phi[:(self._n_bins - 1) * self._n_z_bins])
        sum_phi_cov = np.dot(
            self._trans_matrix[:,:(self._n_bins - 1) * self._n_z_bins],
            np.dot(self._tmp_phi_cov[:(self._n_bins - 1) * self._n_z_bins,
                                     :(self._n_bins - 1) * self._n_z_bins],
                   self._trans_matrix[:,:(self._n_bins - 1) *
                                        self._n_z_bins].transpose()))
        
        summed_phi_norm = compute_pz_norm(
            np.array([self._phi[:self._n_z_bins,0], sum_phi,
                      np.sqrt(sum_phi_cov.diagonal())]).transpose())
        self._tmp_phi[:(self._n_bins - 1) * self._n_z_bins] /= summed_phi_norm
        self._tmp_phi_cov[
            :(self._n_bins - 1) * self._n_z_bins,
            :(self._n_bins - 1) * self._n_z_bins] /= summed_phi_norm ** 2
        
        diff = np.dot(self._trans_matrix, self._tmp_phi)
        cov = np.dot(
            self._trans_matrix,
            np.dot(self._tmp_phi_cov, self._trans_matrix.transpose()))
        
        return (diff, np.linalg.inv(cov))
        
    def get_fit(self):
        return self._x
    
    def get_error(self):
        return self._err
    
    def get_covariance(self):
        return self._covar
        
    def lnprior(self, var):
        """
        Prior for emcee MCMC fit. Prior is just a top hat.
        """
        bias = self.bias(self._phi[:self._n_z_bins, 0])
        if (np.all(bias >= 1) and np.all(bias < 10)):
            return 0.0
        return -np.inf
    
    def lnlike(self, var):
        """
        Likelihood for emcee MCMC. Likelihood is a simple chi^2 using the data
        and data covariance.
        """
        diff, inv_cov = self._fix_bias()
        return -0.5 * np.dot(diff, np.dot(inv_cov, diff))
     
    def lnprob(self, var):
        """
        Probability for emcee MCMC fitter.
        """
        self.set_fit(var)
        lp = self.lnprior(var)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(var)
        
    def fit(self, max_MCMC, burn_in_precent):
        
        ndim, nwalkers = self._n_parm, 100
        wander = np.where(np.array([value / 10.0 for value in self._x]) != 0,
                          np.array([value / 10.0 for value in self._x]), 1e-8)
        pos = [self._x + wander*np.random.randn(ndim) for i in xrange(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        
        print("Running MCMC...")
        sampler.run_mcmc(pos, max_MCMC, rstate0 = np.random.get_state())
        print("Done.")
        
        samples = sampler.flatchain[int(burn_in_precent * max_MCMC):, :]
        lnprob = sampler.flatlnprobability[int(burn_in_precent * max_MCMC):]
        
        print("best fit:", samples[np.argmax(lnprob)])
        ans = samples.mean(axis = 0)
        std = samples.std(axis = 0)
        self.set_fit(ans)
        self.set_error(std)
        
        self._covar = np.zeros((ans.shape[0], ans.shape[0]))
        for idx1 in xrange(self._covar.shape[0]):
            for idx2 in xrange(self._covar.shape[1]):
                self._covar[idx1, idx2] += np.sum(
                    (samples[:,idx1] - ans[idx1]) *
                    (samples[:,idx2] - ans[idx2]))
        self._covar /= 1. * samples.shape[0] - 1.
                                              
        if verbose:
            value_string = "fit "
            for v_idx, value in enumerate(ans):
                value_string += "param%i: %.4e, " % (v_idx, value)
            print(value_string)
            
            fig = corner.corner(samples, bins = 50,
                                labels=["param%i" % idx
                                        for idx in xrange(ans.shape[0])],
                                quantiles=[0.16, 0.5, 0.84])
            fig.savefig("bias_corner.pdf")
            
            print np.dot(self._trans_matrix[
                               :,:(self._n_bins - 1) * self._n_z_bins],
                         self._phi[:(self._n_bins - 1) * self._n_z_bins,1]).shape
            
            pl.figure(figsize = (11, 8.5))
            pl.axhline(0.0, ls = ':', color = 'k')
            pl.errorbar(self._phi[:self._n_z_bins,0],
                        self._phi[(self._n_bins - 1) * self._n_z_bins:, 1],
                        self._phi[(self._n_bins - 1) * self._n_z_bins:, 2],
                        color = cb_set1[1], ls = '--', label = 'raw')
        
            tmp_error = np.dot(self._trans_matrix,
                    np.dot(self._tmp_phi_cov,
                           self._trans_matrix.transpose())).diagonal()
            print tmp_error
            pl.plot(self._phi[:self._n_z_bins,0],
                    np.dot(self._trans_matrix, self._tmp_phi),
                    color = cb_set1[2], label = 'diff')
            pl.plot(self._phi[:self._n_z_bins,0],
                    self.bias(self._phi[:self._n_z_bins,0]),
                    color = cb_set1[3], ls = '--', label = 'bias')
            pl.legend(loc = 0)
            pl.savefig('test_fit.pdf')
            
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi_file_list', required = True,
                        type = str, help = 'CSV list of file names. The last '
                        'file should be the global sample. All files should '
                        'have the same shape that is run over the same z_min, '
                        'z_max and number of bins.')
    parser.add_argument('--phi_boot_file_list', required = True,
                        type = str, help = 'CSV list of file names containing '
                        'bootstraps. These should be the same number as the '
                        'phi files and each should have the same number of '
                        'bootstrap realizations drawn from the same regions.')
    parser.add_argument('--phi_weight_values', default = None,
                        type = str, help = 'CSV list of file values to weight '
                        'weight each phi by (usually total N). Defaults to a '
                        'weight of 1 for each phi.')
    parser.add_argument('--output_fit_file', required = True,
                        type = str, help = 'Output fit values.')
    parser.add_argument('--output_phi_file', required = True,
                        type = str, help = 'Output of phis of fitted data.')
    parser.add_argument('--z_min', default = 0.01,
                        type = str, help = 'Minimum redshift that the recovery '
                        'was run over.')
    parser.add_argument('--z_max', default = 1.0,
                        type = str, help = 'Maximum redshift that the recovery '
                        'was run over.')
    parser.add_argument('--fit_start', default = None,
                        type = str, help = 'CSV list for starting values. '
                        'b0,b1,b2,...bn')
    parser.add_argument('--max_MCMC', default = 10000,
                        type = int, help = 'Maximum MCMC steps.')
    parser.add_argument('--burn_in_frac', default = 0.3,
                        type = float, help = 'Fraction of steps to throw out.')
    parser.add_argument('--verbose', action = 'store_true',
                        help = 'Verbose prints check plots')
    args = parser.parse_args()
    
    verbose = args.verbose
    
    n_files = len(args.phi_file_list.split(','))
    
    for phi_idx, phi_file_name in enumerate(args.phi_file_list.split(',')):
        if phi_idx == 0:
            phi_array = np.loadtxt(phi_file_name)
            continue
        phi_array = np.concatenate((phi_array, np.loadtxt(phi_file_name)))
    n_z_bins = phi_array.shape[0] / n_files
                                   
    for phi_idx, phi_file_name in enumerate(args.phi_boot_file_list.split(',')):
        if phi_idx == 0:
            phi_boot_array = np.loadtxt(phi_file_name)
            continue
        phi_boot_array = np.concatenate((phi_boot_array, np.loadtxt(phi_file_name)))
    phi_cov_array = np.cov(phi_boot_array, rowvar = 1)
    
    phi_weights = np.array([float(value)
                            for value in args.phi_weight_values.split(',')])
    
    x0 = np.array([float(value) for value in args.fit_start.split(',')])
    maglim_fitter = MagLimFitter(phi_array, phi_cov_array, phi_weights, x0,
                                 args.z_min, args.z_max, n_z_bins)
    maglim_fitter.fit(args.max_MCMC, args.burn_in_frac)
    
    print("best fit:", maglim_fitter.get_fit())
    print("best fit error:", maglim_fitter.get_error())
    
    output_header = '#input_flags:\n'
    for arg in vars(args):
        output_header += '#\t%s : %s\n' % (arg, getattr(args, arg))
    
    output_file = open(args.output_fit_file, 'w')
    output_file.writelines(output_header)
    ans = maglim_fitter.get_fit()
    err = maglim_fitter.get_error()
    covar = maglim_fitter.get_covariance()
    
    output_header = 'input_flags:\n'
    for arg in vars(args):
        output_header += '\t%s : %s\n' % (arg, getattr(args, arg))
    for idx in xrange(len(ans)):
        output_header += 'row_param%i\n' % idx
    
    covar = maglim_fitter.get_covariance()
    for var_idx in xrange(len(ans)):
        output_string = '%.8e %.8e ' % (ans[var_idx], err[var_idx])
        for covar_idx in xrange(len(ans)):
            output_string += '%.8e ' % covar[var_idx, covar_idx]
        output_string += '\n'
        output_file.writelines(output_string)
        
    np.savetxt(args.output_phi_file, maglim_fitter._phi, header = output_header)
    
    
    