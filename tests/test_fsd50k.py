"""Numerical and interface tests, using explicitly constructed data only."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from src.fsd50k import (Statistics, coefficient, predict, APOrder, MultilabelAP,
                       group_components, bootstrap, ClipPool, boundaries, mapped_boundaries)


class FSDNumerics(unittest.TestCase):
    def test_streamed_ridge_matches_direct_solve_with_constant_and_duplicate_columns(self):
        rng = np.random.default_rng(1701)
        x = rng.normal(size=(83, 8)); x[:, 0] = 7; x[:, 1] = x[:, 2]
        y = (rng.random((83, 5)) < .3).astype(float)
        stats = Statistics(8, 5)
        for j in range(0, len(x), 13): stats.add(x[j:j+13], y[j:j+13])
        stats.remove(x[2], y[2]); keep = np.arange(len(x)) != 2
        f = stats.factors(); z = (x[keep] - f['mean']) / f['std']; target = y[keep] - f['target_mean']
        np.testing.assert_allclose(f['mean'], x[keep].mean(0), atol=1e-14)
        self.assertEqual(f['std'][0], 1.)
        for lam in (.0001, .1, 1.):
            direct = np.linalg.solve(z.T @ z + len(z)*lam*np.eye(8), z.T @ target)
            np.testing.assert_allclose(coefficient(f, lam), direct, atol=1e-10)

    def test_ap_ties_weighted_and_undefined_classes(self):
        y = np.array([1, 0, 1, 0]); score = np.array([3., 2., 2., 0.]); weights = np.array([2., 1., 3., 0.])
        for w in (None, weights):
            self.assertAlmostEqual(APOrder(y, score).value(w), average_precision_score(y, score, sample_weight=w))
        self.assertTrue(np.isnan(APOrder(np.zeros(4), score).value()))
        self.assertTrue(np.isnan(APOrder(np.ones(4), score).value()))
        yy = np.column_stack([y, np.ones(4)]); ss = np.column_stack([score, score])
        report = MultilabelAP(yy, ss).report(ss)
        self.assertIsNone(report['macro_200_ap']); self.assertEqual(report['defined_classes'], 1)

    def test_groups_are_transitively_merged_and_bootstrap_is_paired(self):
        rows = [{'uploader':'a','content_hash_group':'x'}, {'uploader':'a','content_hash_group':'y'},
                {'uploader':'b','content_hash_group':'y'}, {'uploader':'c','content_hash_group':'z'}]
        np.testing.assert_array_equal(group_components(rows), [0,0,0,1])
        y = np.array([[1],[0],[1],[0]], dtype=np.uint8); scores=np.array([[4.],[3.],[2.],[1.]])
        values, missing = bootstrap(y, {'a':scores,'b':scores.copy()}, np.array([0,1,2,3]), draws=19)
        np.testing.assert_allclose(values['a'], values['b'], equal_nan=True)
        self.assertEqual(missing.shape, (19,))

    def test_irregular_time_support_and_balanced_partition(self):
        b=boundaries(13*44100);self.assertEqual(len(b)-1,3);self.assertEqual(np.diff(b).sum(),13*44100)
        mapped=mapped_boundaries(b,44100,16000,13*16000);self.assertEqual(np.diff(mapped).sum(),208000)
        pool=ClipPool(1,4.)
        pool.add(np.array([[2.,6.]]),np.array([.5,2.5]),0.,4.)
        g,t4,record=pool.result()
        # Cells [0,1.5], [1.5,4]; weighted mean 4.5, not unweighted mean 4.
        np.testing.assert_allclose(g[0],4.5)
        np.testing.assert_allclose(t4[[0,2,4,6]],[2.,4.,6.,6.])
        np.testing.assert_allclose(record['bin_weights'],np.ones(4))


class ReadoutInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/fsd50k_readout.py'
        spec=importlib.util.spec_from_file_location('portable_readout',path)
        cls.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.cli)

    def test_fit_evaluate_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);rng=np.random.default_rng(4);x=rng.normal(size=(24,6));x[:,0]=7
            y=np.tile((np.arange(24)%2)[:,None],(1,200)).astype(np.uint8)
            rows=[{'index':i,'id':str(i),'official_split':'train' if i<12 else ('val' if i<18 else 'eval'),
                   'dedup_train_keep':i!=2} for i in range(24)]
            np.save(r/'features.npy',x);np.save(r/'labels.npy',y)
            (r/'manifest.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in rows))
            common=dict(features=r/'features.npy',labels=r/'labels.npy',manifest=r/'manifest.jsonl')
            fitargs=argparse.Namespace(**common,out=r/'fit',arm='dedup',batch_rows=5)
            self.cli.fit(fitargs)
            selection=json.loads((r/'fit/selection.json').read_text())
            self.assertFalse(selection['evaluation_scores_computed']);self.assertFalse(selection['official_split_counts_match'])
            self.assertEqual(selection['training_rows_used'],11)
            with np.load(r/'fit/selected_model.npz') as model:
                np.testing.assert_allclose(model['mean'],np.delete(x[:12],2,axis=0).mean(0),atol=1e-14)
            evargs=argparse.Namespace(**common,out=r/'evaluation',model_dir=r/'fit')
            self.cli.evaluate(evargs)
            self.assertEqual(json.loads((r/'evaluation/metrics.json').read_text())['recordings'],6)
            with self.assertRaises(ValueError):self.cli.fit(fitargs)
            x[-1]*=2;np.save(r/'features.npy',x)
            evargs.out=r/'changed'
            with self.assertRaisesRegex(ValueError,'identity differs'):self.cli.evaluate(evargs)

    def test_validation_tie_chooses_larger_lambda(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);x=np.ones((12,2));y=np.tile((np.arange(12)%2)[:,None],(1,200)).astype(np.uint8)
            rows=[{'index':i,'id':str(i),'official_split':['train','val','eval'][i//4]} for i in range(12)]
            np.save(r/'x.npy',x);np.save(r/'y.npy',y);(r/'rows.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in rows))
            args=argparse.Namespace(features=r/'x.npy',labels=r/'y.npy',manifest=r/'rows.jsonl',out=r/'fit',arm='full',batch_rows=4)
            self.cli.fit(args)
            self.assertEqual(json.loads((r/'fit/selection.json').read_text())['lambda_'],1.)


if __name__=='__main__':unittest.main()
