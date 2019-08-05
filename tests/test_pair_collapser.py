
import os
import numpy as np
import pandas as pd
import unittest

from the_wizz import pair_maker


class TestPairCollapser(unittest.TestCase):

    def setUp(self):

        # Seed all random numbers for reproducibility.
        np.random.seed(1234)

        # Create a random catalog centered at the pole with a redshift
        # distribution that looks kind of like a mag limited sample.
        self.n_objects = 1000
        decs = np.degrees(
            np.pi / 2 - np.arccos(np.random.uniform(np.cos(np.radians(1.0)),
                                                    np.cos(0),
                                                    size=self.n_objects)))
        ras = np.random.uniform(0, 360, size=self.n_objects)
        redshifts = np.random.lognormal(mean=-1,
                                        sigma=0.5,
                                        size=self.n_objects)
        ids = np.arange(self.n_objects)
        self.catalog = {"id": ids,
                        "ra": ras,
                        "dec": decs,
                        "redshift": redshifts}

        self.z_min = 0.05
        self.z_max = 3.0

        self.r_mins = [0.1, 1]
        self.r_maxes = [1, 10]
        self.r_min = np.min(self.r_mins)
        self.r_max = np.max(self.r_maxes)

        self.expected_columns = ["ref_id",
                                 "redshift"]
        for r_min, r_max in zip(self.r_mins, self.r_maxes):
            self.expected_columns.append("Mpc%.2ft%.2f_counts" %
                                         (r_min, r_max))
            self.expected_columns.append("Mpc%.2ft%.2f_weights" %
                                         (r_min, r_max))
        self.output_path = tempfile.mkdtemp(
            dir=os.path.dirname(__file__))
