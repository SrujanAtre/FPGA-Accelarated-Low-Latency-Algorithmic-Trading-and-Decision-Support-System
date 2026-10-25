import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import unittest
import pandas as pd
import numpy as np
from fpga_ml.provisional_live import ProvisionalHistory
from live_fpga import features_for_inference
from config.config import FEATURES

class ProvisionalTests(unittest.TestCase):
    def setUp(self):
        self.now=pd.Timestamp('2026-10-15 12:32:10',tz='Asia/Kolkata').tz_convert('UTC')
        index=pd.date_range('2026-10-14 09:15',periods=75,freq='5min',tz='Asia/Kolkata').tz_convert('UTC').append(pd.date_range('2026-10-15 09:15','2026-10-15 12:30',freq='5min',tz='Asia/Kolkata').tz_convert('UTC'))
        close=22000+np.arange(len(index))*.2+np.sin(np.arange(len(index)))
        self.raw=pd.DataFrame({'open':close-.1,'high':close+.2,'low':close-.2,'close':close,'volume':0.},index=index)
        self.state=ProvisionalHistory();self.state.refresh(self.raw,self.now)
    def add(self,price,stamp=None):
        stamp=self.now if stamp is None else stamp
        return self.state.add({'id':'^NSEI','time':str(stamp.value//1000000),'price':price},self.now)
    def test_history_initializes_without_new_candles(self):
        self.assertTrue(self.add(22100))
        frame=self.state.feature_frame(self.now)
        self.assertEqual(len(frame),len(self.raw))
        self.assertEqual(frame.index[-1],self.now.floor('5min'))
        self.assertEqual(frame.iloc[-1]['close'],22100)
        features=features_for_inference(frame)
        self.assertTrue(np.isfinite(features.iloc[-1][FEATURES['feature_cols']].to_numpy(dtype=float)).all())
    def test_updates_replace_partial_not_append_thirty_second_rows(self):
        self.add(22100);before=self.state.history.copy();size=len(self.state.feature_frame(self.now))
        self.now+=pd.Timedelta(seconds=30);self.add(22101)
        frame=self.state.feature_frame(self.now)
        self.assertEqual(len(frame),size)
        self.assertEqual(frame.iloc[-1]['close'],22101)
        pd.testing.assert_frame_equal(before,self.state.history)
        self.assertEqual(frame.index[-1].second,0)
    def test_duplicate_stale_and_missing_previous_bar(self):
        self.assertTrue(self.add(22100));self.assertFalse(self.add(22100))
        self.assertFalse(self.add(22101,self.now-pd.Timedelta(seconds=30)))
        self.state.history=self.state.history.drop(self.now.floor('5min')-pd.Timedelta(minutes=5))
        with self.assertRaisesRegex(ValueError,'preceding completed'):self.state.feature_frame(self.now)
    def test_unfinished_seed_never_committed_as_completed(self):
        self.assertNotIn(self.now.floor('5min'),self.state.history.index)
        self.add(22100)
        with self.assertRaisesRegex(ValueError,'fresh'):self.state.feature_frame(self.now+pd.Timedelta(seconds=16))

if __name__=='__main__':unittest.main()
