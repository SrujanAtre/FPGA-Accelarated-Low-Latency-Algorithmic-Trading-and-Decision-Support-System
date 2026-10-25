import unittest
import pandas as pd
import numpy as np
from streaming_30s import CandleBuilder
from live_fpga import features_for_inference
from config.config import FEATURES

class CandleTests(unittest.TestCase):
    def setUp(self):
        self.t=pd.Timestamp('2026-10-15 10:00:00',tz='Asia/Kolkata').tz_convert('UTC')
        self.b=CandleBuilder(self.t+pd.Timedelta(seconds=5))
    def tick(self,seconds,price):
        t=self.t+pd.Timedelta(seconds=seconds)
        return self.b.add({'id':'^NSEI','time':str(t.value//1000000),'price':price},t)
    def test_partial_duplicate_and_out_of_order(self):
        self.assertFalse(self.tick(20,100))
        self.assertTrue(self.tick(40,103));self.assertTrue(self.tick(35,101));self.assertTrue(self.tick(50,99))
        self.assertFalse(self.tick(50,99))
        self.assertEqual(self.b.close(self.t+pd.Timedelta(seconds=61)),[])
        bar=self.b.close(self.t+pd.Timedelta(seconds=62))[0][1]
        self.assertEqual([bar[k] for k in ['open','high','low','close','tick_count']],[101,103,99,99,3])
        self.assertEqual(self.b.close(self.t+pd.Timedelta(seconds=100)),[])
        self.assertFalse(self.tick(45,110))
    def test_no_empty_candles_and_stale_ticks(self):
        self.assertEqual(self.b.close(self.t+pd.Timedelta(seconds=100)),[])
        self.assertFalse(self.b.add({'id':'^NSEI','time':str(self.t.value//1000000),'price':100},self.t+pd.Timedelta(minutes=10)))
    def test_fifty_bar_causal_warmup(self):
        index=pd.date_range(self.t,periods=51,freq='30s');close=22000+np.arange(51)*.3+np.sin(np.arange(51))
        frame=pd.DataFrame({'open':close-.1,'high':close+.2,'low':close-.2,'close':close,'volume':0.},index=index)
        features=features_for_inference(frame)
        self.assertEqual(features.index[0],index[49])
        self.assertTrue(np.isfinite(features[FEATURES['feature_cols']]).all().all())
        prefix=features_for_inference(frame.iloc[:50])
        np.testing.assert_array_equal(prefix[FEATURES['feature_cols']],features.iloc[:1][FEATURES['feature_cols']])

if __name__=='__main__':unittest.main()
