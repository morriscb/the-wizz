
from astropy.cosmology import Planck15, z_at_value
import astropy.units as u
import numpy as np
import pandas as pd
import pickle


def linear_bins(z_min, z_max, n_bins):
    """Create bins linear or redshift.

    Parameters
    ----------
    z_min : `float`
        Minimum redshift.
    z_max : `float`
        Maximum redshift
    n_bins : `int`
        Number of redshift bins to create

    Returns
    -------
    bin_edges : `numpy.ndarray`, (n_bins + 1,)
        Redshift bin edges.
    """
    return np.linspace(z_min, z_max, n_bins + 1)


def log_bins(z_min, z_max, n_bins):
    """Create bins equally spaced in log(1 + z).

    Parameters
    ----------
    z_min : `float`
        Minimum redshift.
    z_max : `float`
        Maximum redshift
    n_bins : `int`
        Number of redshift bins to create

    Returns
    -------
    bin_edges : `numpy.ndarray`, (n_bins + 1,)
        Redshift bin edges.
    """
    log_min = np.log(1 + z_min)
    log_max = np.log(1 + z_max)
    log_edges = np.linspace(log_min, log_max, n_bins + 1)
    return np.exp(log_edges) - 1


def comoving_bins(z_min, z_max, n_bins):
    """Create bins equally spaced in comoving distance.

    Assumes a Planck2015 cosmology.

    Parameters
    ----------
    z_min : `float`
        Minimum redshift.
    z_max : `float`
        Maximum redshift
    n_bins : `int`
        Number of redshift bins to create

    Returns
    -------
    bin_edges : `numpy.ndarray`, (n_bins + 1,)
        Redshift bin edges.
    """
    cov_min = Planck15.comoving_distance(z_min).value
    cov_max = Planck15.comoving_distance(z_max).value
    cov_edges = np.linspace(cov_min, cov_max, n_bins + 1)

    tmp_edges = []
    for cov_edge in cov_edges:
        tmp_edges.append(z_at_value(Planck15.comoving_distance,
                         cov_edge * u.Mpc))
    return np.array(tmp_edges)


class PDFMaker:
    """Class to estimate the raw, clustering redshift over-desnity from the
    `pandas.DataFrame` output of pair_maker or pair_collapser.

    Parameters
    ----------
    z_min : `float`
        Minimum redshift for binning. Overwritten if custom bins are
        specified.
    z_max : `float`
        Maximum redshift for binning. Overwritten if custom bins are
        specified.
    bins : `int` or `numpy.ndarray`
        Either the number of redshift bins to create or a custom set of bin
        edges. If bin edges are specified: input ``z_min`` and ``z_max`` are
        overridden and then derived from the input array.
    binning_type : `str`
        Specify way to bin in redshift. Options are:

        - ``linear``: Equally spaced bins in redshift.
        - ``log``: Equally spaced bins in log(1 + z).
        - ``comoving``: Space bins linearly in comoving distance as a function
          of redshift. Assumes a Planck15 cosmology.
    scale_name : `str`
        Name scale bin to compute the estimate for e.g.' 'Mpc1.00t10.00'.
    """
    def __init__(self,
                 z_min=0.01,
                 z_max=5.00,
                 bins=10,
                 binning_type='linear',
                 scale_name="Mpc1.00t10.00"):
        self.scale_name = scale_name
        self.z_min = z_min
        self.z_max = z_max
        self.bins = bins
        self.binning_type = binning_type
        if isinstance(bins, np.ndarray):
            self.z_min = np.min(bins)
            self.z_max = np.max(bins)
            self.bin_edges = bins
            self.bins = len(self.bin_edges) - 1
            self.binning_type = "custom"
        elif isinstance(bins, int):
            self._create_bin_edges()
        else:
            raise TypeError("Input bins not valid type. Use int or ndarray.")

    def _create_bin_edges(self):
        """Compute bin edges in redshift using different prescriptions.
        """
        if self.binning_type == "linear":
            self.bin_edges = linear_bins(self.z_min, self.z_max, self.bins)
        elif self.binning_type == "log":
            self.bin_edges = log_bins(self.z_min, self.z_max, self.bins)
        elif self.binning_type == "comoving":
            self.bin_edges = comoving_bins(self.z_min, self.z_max, self.bins)
        else:
            raise TypeError("Requested binning type is invalid. Use either "
                            "'linear', 'log', 'comoving'. Custom bins can be "
                            "give by inputing a `numpy.ndarray` for ``bins``.")

    def run(self,
            ref_unkn,
            ref_rand,
            ref_weights=None,
            weight_rand=False,
            ref_ref=None,
            ref_ref_rand=None,
            ref_ref_rand_weights=None):
        """Combine pairs of reference against unknown with reference against
        random to create the output over-densities.

        Optionally, can apply weights per-reference object in the correlation.
        Additionally can provide an reference sample "auto-correlation" run
        though pair_maker or pair_collapser that allows for the production of
        an estimator that produces a quantity closer to n(z).
        (n(z) * b_u / b_r assuming delta function reference bins/redshifts.)

        Parameters
        ----------
        ref_unkn : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            containing the counts of objects with unknown redshifts around
            reference objects with known redshifts.
        ref_rand : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            containing the counts of randoms around reference objects with
            known redshifts.
        weight_rand : `bool`
            If randoms correlated against the the reference objects did not
            already have weights, this flag uses the average weight of the
            unknown sample times the random counts to compute a weight.
        ref_weights : `numpy.ndarray` or `None`
            Optional weights to apply to each reference object in the
            calculated correlations.
        ref_ref : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            counting the number of reference objects around themselves.
            Optional for producing outputs with the dark-matter clustering
            mitigated.
        ref_ref_rand : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            counting the number of reference objects around randoms. This can
            be the same data as ref_rand or a specific set of randoms sampling
            the reference objects. Optional for producing outputs with the
            dark-matter clustering mitigated.
        ref_ref_rand_weights : `numpy.ndarray` or `None`
            Optional weights to apply to each reference object in the
            calculated correlations. This can be the same as ref_weights when
            using a DD / DR estimator or, if using randoms sampling the
            reference objects, a set of weights for those randoms.

        Returns
        -------
        output_dataframe : `pandas.DataFrame`
            Output clustering redshift calculations as a function of redshift.
            Contains raw columns for additional jackknife/bootstrapping using
            independent samples on the sky.
        """
        ref_unkn_binned = self.bin_data(ref_unkn, ref_weights)
        ref_rand_binned = self.bin_data(ref_rand, ref_weights)

        ref_unkn_count_corr, ref_unkn_weight_corr = self.compute_correlation(
            ref_unkn_binned, ref_rand_binned, weight_rand)

        ref_unkn_binned["corr"] = ref_unkn_count_corr
        ref_unkn_binned["weighted_corr"] = ref_unkn_weight_corr
        ref_unkn_binned["corr_err"] = 1 / np.sqrt(ref_unkn_binned["counts"])
        ref_unkn_binned["weighted_corr_err"] = 1 / np.sqrt(
            ref_unkn_binned["weights"])

        ref_unkn_binned["rand_counts"] = ref_rand_binned["counts"]
        ref_unkn_binned["rand_weights"] = ref_rand_binned["weights"]
        if weight_rand:
            ref_unkn_binned["rand_weights"] *= ref_unkn_binned[
                "ave_unkn_weight"]
        ref_unkn_binned["tot_sample_rand"] = \
            ref_rand_binned["tot_sample"]
        ref_unkn_binned["n_ref_rand"] = ref_rand_binned["n_ref"]

        n_z_r = ref_unkn_binned["n_ref"] / ref_unkn_binned["dz"]
        n_z_r /= np.sum(ref_unkn_binned["n_ref"])
        ref_unkn_binned["n_z_ref"] = n_z_r

        if ref_ref is not None and ref_ref_rand is not None:
            ref_ref_binned = self.bin_data(ref_ref, ref_weights)
            ref_ref_rand_binned = self.bin_data(ref_ref_rand,
                                                ref_ref_rand_weights)
            ref_ref_count_corr, ref_ref_w_corr = \
                self.compute_correlation(
                    ref_ref_binned, ref_ref_rand_binned, weight_rand)

            ref_unkn_binned["n_z_bu_br"] = (
                n_z_r * ref_unkn_weight_corr / ref_ref_w_corr)
            ref_unkn_binned["n_z_bu_br_err"] = n_z_r * np.sqrt(
                (ref_unkn_binned["weighted_corr_err"] / ref_ref_w_corr) ** 2 +
                (ref_unkn_weight_corr /
                 (np.sqrt(ref_ref_binned["weights"]) *
                  ref_ref_w_corr ** 2)) ** 2)

            ref_unkn_binned["ref_ref_counts"] = ref_ref_binned["counts"]
            ref_unkn_binned["n_ref_ref_rand"] = ref_ref_rand_binned["n_ref"]
            ref_unkn_binned["ref_ref_rand_counts"] = \
                ref_ref_rand_binned["counts"]
            ref_unkn_binned["ref_ref_weights"] = ref_ref_binned["weights"]
            ref_unkn_binned["ref_ref_rand_weights"] = ref_ref_rand_binned[
                "weights"]
            if weight_rand:
                ref_unkn_binned["ref_ref_rand_weights"] *= ref_ref_binned[
                    "ave_unkn_weight"]
            ref_unkn_binned["tot_sample_ref"] = ref_ref_binned[
                "tot_sample"]
            ref_unkn_binned["tot_sample_ref_ref_rand"] = ref_ref_rand_binned[
                "tot_sample"]

        return ref_unkn_binned

    def run_region_bootstraped(self,
                               ref_unkn,
                               ref_rand,
                               ref_weights=None,
                               weight_rand=False,
                               ref_ref=None,
                               ref_ref_rand=None,
                               ref_ref_rand_weights=None,
                               bootstraps=1000,
                               output_bootstraps=None):
        """Combine pairs of reference against unknown as in the run method and
        additionally bootstrap over spatial regions to estimate errors.

        Parameters
        ----------
        ref_unkn : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            containing the counts of objects with unknown redshifts around
            reference objects with known redshifts.
        ref_rand : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            containing the counts of randoms around reference objects with
            known redshifts.
        weight_rand : `bool`
            If randoms correlated against the the reference objects did not
            already have weights, this flag uses the average weight of the
            unknown sample times the random counts to compute a weight.
        ref_weights : `numpy.ndarray` or `None`
            Optional weights to apply to each reference object in the
            calculated correlations.
        ref_ref : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            counting the number of reference objects around themselves.
            Optional for producing outputs with the dark-matter clustering
            mitigated.
        ref_ref_rand : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            counting the number of reference objects around randoms. This can
            be the same data as ref_rand or a specific set of randoms sampling
            the reference objects. Optional for producing outputs with the
            dark-matter clustering mitigated.
        ref_ref_rand_weights : `numpy.ndarray` or `None`
            Optional weights to apply to each reference object in the
            calculated correlations. This can be the same as ref_weights when
            using a DD / DR estimator or, if using randoms sampling the
            reference objects, a set of weights for those randoms.
        bootstraps : `int` or `numpy.ndarray`
            Number of bootstrap realizations to create or
        output_bootstraps : `str`
            Name output pickle file to write raw bootstraps to.

        Returns
        -------
        output_dataframe : `pandas.DataFrame`
            Output clustering redshift calculations as a function of redshift
            with errorbars derived from spatial bootstrapping.
        """
        if ref_unkn.index.name != "region":
            ref_unkn = ref_unkn.set_index(["region"])
        if ref_rand.index.name != "region":
            ref_rand = ref_rand.set_index(["region"])
        n_regions = len(ref_unkn.index.unique())
        if ref_ref is not None and ref_ref_rand is not None:
            if ref_ref.index.name != "region":
                ref_ref = ref_ref.set_index(["region"])
            if ref_ref_rand.index.name != "region":
                ref_ref_rand = ref_ref_rand.set_index(["region"])

        if isinstance(bootstraps, int):
            bootstraps = np.random.randint(n_regions,
                                           size=(bootstraps, n_regions))

        ref_unkn_regions = np.empty((n_regions, self.bins))
        ref_rand_regions = np.empty((n_regions, self.bins))
        ref_ref_regions = np.empty((n_regions, self.bins))
        ref_ref_rand_regions = np.empty((n_regions, self.bins))

        n_ref_regions = np.empty((n_regions, self.bins))
        n_ref_rand_regions = np.empty((n_regions, self.bins))
        n_ref_ref_rand_regions = np.empty((n_regions, self.bins))

        tot_ref_regions = np.empty((n_regions, self.bins))
        tot_unkn_regions = np.empty((n_regions, self.bins))
        tot_rand_regions = np.empty((n_regions, self.bins))
        tot_ref_ref_rand_regions = np.empty((n_regions, self.bins))

        mean_redshifts = np.zeros(self.bins)
        delta_z = np.empty(self.bins)

        for region in ref_unkn.index.unique():
            region_ref_unkn = ref_unkn.loc[region]
            region_ref_rand = ref_rand.loc[region]
            if ref_ref is None:
                region_ref_ref = None
            else:
                region_ref_ref = ref_ref.loc[region]
            if ref_ref_rand is None:
                region_ref_ref_rand = None
            else:
                region_ref_ref_rand = ref_ref_rand.loc[region]
            if ref_weights is None or ref_ref_rand_weights is None:
                region_ref_weights = None
                region_ref_ref_rand_weights = None
            else:
                region_ref_weights = ref_weights.loc[region]
                region_ref_ref_rand_weights = ref_ref_rand_weights.loc[region]
            data = self.run(region_ref_unkn,
                            region_ref_rand,
                            region_ref_weights,
                            weight_rand,
                            region_ref_ref,
                            region_ref_ref_rand,
                            region_ref_ref_rand_weights)

            ref_unkn_regions[region, :] = data["weights"].to_numpy()
            ref_rand_regions[region, :] = data["rand_weights"].to_numpy()

            n_ref_regions[region, :] = data["n_ref"].to_numpy()
            n_ref_rand_regions[region, :] = data["n_ref_rand"].to_numpy()

            tot_unkn_regions[region, :] = data["tot_sample"].to_numpy()
            tot_rand_regions[region, :] = \
                data["tot_sample_rand"].to_numpy()

            if ref_ref is not None and ref_ref_rand is not None:
                n_ref_ref_rand_regions[region, :] = \
                    data["n_ref_ref_rand"].to_numpy()

                tot_ref_regions[region, :] = data["tot_sample_ref"].to_numpy()
                tot_ref_ref_rand_regions[region, :] = data[
                    "tot_sample_ref_ref_rand"].to_numpy()

                ref_ref_regions[region, :] = data["ref_ref_weights"].to_numpy()
                ref_ref_rand_regions[region, :] = data[
                    "ref_ref_rand_weights"].to_numpy()

            mean_redshifts += (data["mean_redshift"].to_numpy() *
                               data["n_ref"].to_numpy())
            delta_z = data["dz"].to_numpy()

        mean_redshifts /= n_ref_regions.sum(axis=0)

        boot_ref_unkn = ref_unkn_regions[bootstraps, :].sum(axis=1)
        boot_ref_rand = ref_rand_regions[bootstraps, :].sum(axis=1)
        boot_ref_ref = ref_ref_regions[bootstraps, :].sum(axis=1)
        boot_ref_ref_rand = ref_ref_rand_regions[bootstraps, :].sum(axis=1)

        boot_n_ref = n_ref_regions[bootstraps, :].sum(axis=1)
        boot_n_ref_rand = n_ref_rand_regions[bootstraps, :].sum(axis=1)
        boot_n_ref_ref_rand = n_ref_ref_rand_regions[bootstraps, :].sum(axis=1)

        boot_tot_ref = tot_ref_regions[bootstraps, :].sum(axis=1)
        boot_tot_unkn = tot_unkn_regions[bootstraps, :].sum(axis=1)
        boot_tot_rand = tot_rand_regions[bootstraps, :].sum(axis=1)
        boot_tot_ref_ref_rand = \
            tot_ref_ref_rand_regions[bootstraps, :].sum(axis=1)

        boot_corr = ((boot_ref_unkn / boot_ref_rand) *
                     ((boot_n_ref_rand * boot_tot_rand) /
                      (boot_n_ref * boot_tot_unkn))) - 1
        boot_ref_corr = (
            (boot_ref_ref / boot_ref_ref_rand) *
            ((boot_n_ref_ref_rand * boot_tot_ref_ref_rand) /
             (boot_n_ref * boot_tot_ref))) - 1

        boot_ratio = boot_corr / boot_ref_corr
        boot_n_z_r = (boot_n_ref / boot_tot_ref) / delta_z
        boot_n_z_bu_br = boot_ratio * boot_n_z_r
        boot_normed_n_z = np.empty_like(boot_n_z_bu_br)
        boot_z_mean = np.empty(len(bootstraps))
        boot_z_var = np.empty(len(bootstraps))
        for idx in range(len(bootstraps)):
            norm = np.sum(boot_n_z_bu_br[idx, :] * delta_z)
            boot_normed_n_z[idx, :] = boot_n_z_bu_br[idx, :] / norm
            boot_z_mean[idx] = np.sum(boot_normed_n_z[idx, :] * delta_z * mean_redshifts)
            boot_z_var[idx] = np.sum(boot_normed_n_z[idx, :] *
                                     delta_z *
                                     (mean_redshifts - boot_z_mean[idx]) ** 2)

        if output_bootstraps is not None:
            with open(output_bootstraps, 'wb') as pkl_file:
                pickle.dump({"mean_redshift": mean_redshifts,
                             "dz": delta_z,
                             "unkn_corr": boot_corr,
                             "ref_corr": boot_ref_corr,
                             "n_z_bu_br": boot_n_z_bu_br},
                            pkl_file)

        low_corr, median_corr, hi_corr = np.percentile(
            boot_corr, [50 - 34.1, 50, 50 + 34.1], axis=0)
        low_ref_corr, median_ref_corr, hi_ref_corr = np.percentile(
            boot_ref_corr, [50 - 34.1, 50, 50 + 34.1], axis=0)
        low_nz_bu_br, median_n_z_bu_br, hi_nz_bu_br = np.percentile(
            boot_n_z_bu_br, [50 - 34.1, 50, 50 + 34.1], axis=0)
        low_norm_nz, median_norm_nz, hi_norm_nz, = np.percentile(
            boot_normed_n_z, [50 - 34.1, 50, 50 + 34.1], axis=0)
        z_mean_low, z_mean_med, z_mean_hi = np.percentile(
            boot_z_mean, [50 - 34.1, 50, 50 + 34.1])
        z_var_low, z_var_med, z_var_hi = np.percentile(
            boot_z_var, [50 - 34.1, 50, 50 + 34.1])

        return pd.DataFrame(
            data={"mean_redshift": mean_redshifts,
                  "dz": delta_z,
                  "weighted_corr": np.nanmean(boot_corr, axis=0),
                  "weighted_corr_err": np.nanstd(boot_corr, ddof=1, axis=0),
                  "weighted_corr_low": low_corr,
                  "weighted_corr_hi": hi_corr,
                  "ref_corr": np.nanmean(boot_ref_corr, axis=0),
                  "ref_corr_err": np.nanstd(boot_ref_corr, ddof=1, axis=0),
                  "ref_corr_low": low_ref_corr,
                  "ref_corr_hi": hi_ref_corr,
                  "n_z_bu_br": np.mean(boot_n_z_bu_br, axis=0),
                  "n_z_bu_br_err": np.nanstd(boot_n_z_bu_br, ddof=1, axis=0),
                  "n_z_bu_br_med": median_n_z_bu_br,
                  "n_z_bu_br_low": low_nz_bu_br,
                  "n_z_bu_br_hi": hi_nz_bu_br,
                  "n_z_normed": np.mean(boot_normed_n_z),
                  "n_z_normed_err": np.nanstd(boot_normed_n_z, ddof=1, axis=0),
                  "n_z_normed_med": median_norm_nz,
                  "n_z_normed_low": low_norm_nz,
                  "n_z_normed_hi": hi_norm_nz,
                  "n_z_r": np.nanmean(boot_n_z_r, axis=0),
                  "z_mean": np.full(self.bins, np.nanmean(boot_z_mean)),
                  "z_mean_err": np.full(self.bins,
                                        np.nanstd(boot_z_mean,
                                                  ddof=1)),
                  "z_mean_med": np.full(self.bins, z_mean_med),
                  "z_mean_low": np.full(self.bins, z_mean_low),
                  "z_mean_hi": np.full(self.bins, z_mean_hi),
                  "z_var": np.full(self.bins, np.nanmean(boot_z_var)),
                  "z_var_err": np.full(self.bins,
                                       np.nanstd(boot_z_var,
                                                 ddof=1)),
                  "z_var_med": np.full(self.bins, z_var_med),
                  "z_var_low": np.full(self.bins, z_var_low),
                  "z_var_hi": np.full(self.bins, z_var_hi)})

    def bin_data(self, data, ref_weights=None):
        """Bin the reference data in the requested redshift bins.

        Parameters
        ----------
        data : `pandas.DataFrame`
            DataFrame containing the raw point from pair_maker or
            pair_collapser.
        ref_weights : `numpy.ndarray`
            Optional weights to apply to each object. Used in computing the
            total weighted pairs and the average redshift of the bin.

        Returns
        -------
        output_data : `pandas.DataFrame`
            Binned pair counts as a function of redshift.
        """
        if ref_weights is None:
            ref_weights = np.ones(len(data))

        z_mask = np.logical_and(data["redshift"] > self.z_min,
                                data["redshift"] < self.z_max)
        tmp_data = data[z_mask]
        tmp_weights = ref_weights[z_mask]
        bin_number = np.digitize(tmp_data["redshift"], self.bin_edges)

        total_sample = data.iloc[0]["tot_sample"]
        ave_weight = data.iloc[0]["ave_unkn_weight"]

        output_data = []
        for z_bin in range(self.bins):
            dz = self.bin_edges[z_bin + 1] - self.bin_edges[z_bin]
            bin_mask = bin_number == z_bin + 1
            bin_data = tmp_data[bin_mask]
            bin_weights = tmp_weights[bin_mask]
            row_dict = {"mean_redshift": (np.sum(bin_data["redshift"] *
                                                 bin_weights) /
                                          np.sum(bin_weights)),
                        "z_min": self.bin_edges[z_bin],
                        "z_max": self.bin_edges[z_bin + 1],
                        "dz": dz,
                        "counts": np.sum(bin_data["%s_count" %
                                                  self.scale_name]),
                        "weights": np.sum(bin_data["%s_weight" %
                                                   self.scale_name] *
                                          bin_weights),
                        "n_ref": len(bin_data),
                        "ref_weights": tmp_weights[bin_mask].sum(),
                        "tot_sample": total_sample,
                        "ave_unkn_weight": ave_weight}
            output_data.append(row_dict)
        return pd.DataFrame(output_data)

    def compute_correlation(self, data, random, weight_rand):
        """Compute the correlation function for the input data pairs using
        the w_2 estimator from Landy & Szalay 93.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Data binned in reference redshifts to create correlation functions.
        random : `pandas.DataFrame`
            Randoms binned in reference redshifts to create correlation
            functions.
        weight_rand : `bool`
            Use average weight on randoms.

        Returns
        -------
        count_corr, weight_corr : (`pandas.Series`, 'pandas.Series')
            Simple correlation estimators (DD / DR - 1) for pure counts and
            weighted counts.
        """
        rand_ratio = ((random["n_ref"] * random["tot_sample"]) /
                      (data["n_ref"] * data["tot_sample"]))
        count_corr = (
            data["counts"] * rand_ratio / random["counts"] - 1)
        if weight_rand:
            rand_ratio /= data["ave_unkn_weight"]
        weight_corr = (
            data["weights"] * rand_ratio / random["weights"] - 1)
        return count_corr, weight_corr
