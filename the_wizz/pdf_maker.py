
from astropy.cosmology import Planck15, z_at_value
import astropy.units as u
import numpy as np
import pandas as pd


class PDFMaker(object):
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
            self.bin_edges = np.linspace(self.z_min, self.z_max, self.bins + 1)
        elif self.binning_type == "log":
            log_min = np.log(1 + self.z_min)
            log_max = np.log(1 + self.z_max)
            log_edges = np.linspace(log_min, log_max, self.bins + 1)
            self.bin_edges = np.exp(log_edges) - 1
        elif self.binning_type == "comoving":
            cov_min = Planck15.comoving_distance(self.z_min).value
            cov_max = Planck15.comoving_distance(self.z_max).value
            cov_edges = np.linspace(cov_min, cov_max, self.bins + 1)

            tmp_edges = []
            for cov_edge in cov_edges:
                tmp_edges.append(z_at_value(Planck15.comoving_distance,
                                            cov_edge * u.Mpc))
            self.bin_edges = np.array(tmp_edges)
        else:
            raise TypeError("Requested binning type is invalid. Use either "
                            "'linear', 'log', 'comoving'. Custom bins can be "
                            "give by inputing a `numpy.ndarray` for ``bins``.")

    def run(self,
            ref_unkn,
            ref_rand,
            ref_ref=None,
            ref_weights=None):
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
        ref_ref : `pandas.DataFrame`
            DataFrame output from pair_maker or pair_collapser run methods
            counting the number of reference objects around themselves.
            Optional for producing outputs with the dark-matter clustering
            mitigated.
        ref_weights : `numpy.ndarray` or `None`
            Optional weights to apply to each reference object in the
            calculated correlations.

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
            ref_unkn_binned, ref_rand_binned)

        ref_unkn_binned["corr"] = ref_unkn_count_corr
        ref_unkn_binned["weighted_corr"] = ref_unkn_weight_corr
        ref_unkn_binned["corr_err"] = 1 / np.sqrt(ref_unkn_binned["counts"])
        ref_unkn_binned["weighted_corr_err"] = 1 / np.sqrt(
            ref_unkn_binned["weights"])

        ref_unkn_binned["rand_counts"] = ref_rand_binned["counts"]
        ref_unkn_binned["rand_weights"] = ref_rand_binned["weights"]
        ref_unkn_binned["tot_sample_rand"] = ref_rand_binned["tot_sample"]

        if ref_ref is not None:
            ref_ref_binned = self.bin_data(ref_ref, ref_weights)
            ref_ref_count_corr, ref_ref_w_corr = self.compute_correlation(
                ref_ref_binned, ref_rand_binned)

            n_z_s = ref_unkn_binned["n_ref"] / ref_unkn_binned["dz"]
            n_z_s /= ref_ref_binned["tot_sample"]
            ref_unkn_binned["n_z_bu_bs"] = (
                n_z_s * ref_unkn_weight_corr / ref_ref_w_corr)
            ref_unkn_binned["n_z_bu_bs_err"] = n_z_s * np.sqrt(
                (ref_unkn_binned["weighted_corr_err"] / ref_ref_w_corr) ** 2 +
                (ref_unkn_weight_corr /
                 (np.sqrt(ref_ref_binned["weights"]) *
                  ref_ref_w_corr ** 2)) ** 2)

            ref_unkn_binned["ref_counts"] = ref_ref_binned["counts"]
            ref_unkn_binned["ref_weights"] = ref_ref_binned["weights"]
            ref_unkn_binned["tot_sample_ref"] = ref_ref_binned["tot_sample"]

        return ref_unkn_binned

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
                        "counts": np.sum(bin_data["%s_counts" %
                                                  self.scale_name]),
                        "weights": np.sum(bin_data["%s_weights" %
                                                   self.scale_name] *
                                          bin_weights),
                        "n_ref": len(bin_data),
                        "tot_sample": total_sample}
            output_data.append(row_dict)
        return pd.DataFrame(output_data)

    def compute_correlation(self, data, random):
        """Compute the correlation function for the input data pairs using
        the w_2 estimator from Landy & Szalay 93.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Data binned in reference redshifts to create correlation functions.
        random : `pandas.DataFrame`
            Randoms binned in reference redshifts to create correlation
            functions.

        Returns
        -------
        count_corr, weight_corr : (`pandas.Series`, 'pandas.Series')
            Simple correlation estimators (DD / DR - 1) for pure counts and
            weighted counts.
        """
        rand_ratio = random["tot_sample"] / data["tot_sample"]
        count_corr = (
            data["counts"] * rand_ratio / random["counts"] - 1)
        weight_corr = (
            data["weights"] * rand_ratio / random["weights"] - 1)
        return count_corr, weight_corr
