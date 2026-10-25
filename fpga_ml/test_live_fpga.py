"""Focused checks for causal features and completed-bar selection."""
import unittest,contextlib,io
import numpy as np
import pandas as pd
from live_fpga import ROOT,completed_bars,features_for_inference,clean_bars
from features.feature_engineering import build_features
from config.config import FEATURES

class LiveFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(ROOT/'data/raw/nifty_1m.csv',index_col=0,parse_dates=True)
    def test_existing_training_features_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):training=build_features(self.raw)
        saved=pd.read_csv(ROOT/'data/processed/features.csv',index_col=0,parse_dates=True)
        np.testing.assert_allclose(training[FEATURES['feature_cols']],saved[FEATURES['feature_cols']],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(training.label,saved.label)
    def test_live_keeps_latest_bar_and_has_no_labels(self):
        live=features_for_inference(self.raw)
        self.assertNotIn('label',live)
        self.assertEqual(live.index[-1],self.raw.index[-1])
        prefix=features_for_inference(self.raw.iloc[:-10])
        np.testing.assert_allclose(prefix[FEATURES['feature_cols']],live.loc[prefix.index,FEATURES['feature_cols']],rtol=0,atol=0)
    def test_only_completed_bars_with_grace(self):
        frame=clean_bars(self.raw.tail(3))
        end=frame.index[-1]
        self.assertNotIn(end,completed_bars(frame,end+pd.Timedelta(minutes=5,seconds=9)).index)
        self.assertIn(end,completed_bars(frame,end+pd.Timedelta(minutes=5,seconds=10)).index)

if __name__=='__main__':unittest.main()
