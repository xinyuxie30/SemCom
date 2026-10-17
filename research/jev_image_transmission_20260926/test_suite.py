"""Focused correctness tests: budgets, label isolation, group split and CRN identity."""
import inspect
import unittest
import numpy as np
from suite_common import ROOT, ACTIONS, read, resources, feedback_ok, cases, validate_official
from suite_learning import features

class Contracts(unittest.TestCase):
    def test_budget_and_stop(self):
        cfg=read(ROOT/'suite_config.json')
        for a in ACTIONS:
            r=resources(a,2,2,cfg,a!='stop')
            self.assertLessEqual(r['total_symbols'],cfg['budget_symbols'])
            self.assertEqual(r['total_symbols'],sum(r[k] for k in ['payload_symbols','feedback_symbols','header_symbols','pilot_symbols']))
        self.assertEqual(resources('stop',1,0,cfg,False)['feedback_symbols'],0)

    def test_age_ablation(self):
        state={'snr_db':6,'last_known_csi_magnitude':1,'csi_age_at_last_base_symbol_ms':.2,
            'region_features':{r:{k:.1 for k in ['mean','std','gradient','laplacian_variance','entropy_bits','clipped_fraction']} for r in ACTIONS[:4]}}
        decision={'descriptions':{r:{'distribution':dict(zip(['low','noticeable','severe','uncertain'],[.1,.2,.3,.4]))} for r in ACTIONS[:4]},
            'official_jev':{'probabilities':{a:1/6 for a in ACTIONS}}}
        before=features(state,decision,'official_no_age')
        state['csi_age_at_last_base_symbol_ms']=99
        np.testing.assert_array_equal(before,features(state,decision,'official_no_age'))
        self.assertEqual(features(state,decision,'official')[26],99)

    def test_feedback_pairing(self):
        self.assertFalse(feedback_ok('a',1))
        self.assertTrue(feedback_ok('a',0))
        self.assertEqual(feedback_ok('a',.2),feedback_ok('a',.2))

    def test_split(self):
        manifest=read(ROOT/'suite_data.json')
        sets={}
        for item in manifest['images']:
            sets.setdefault(item['split'],set()).add(item['scene_id'])
        keys=list(sets)
        for i,a in enumerate(keys):
            for b in keys[i+1:]: self.assertFalse(sets[a]&sets[b])
        self.assertEqual(len(list(cases(read(ROOT/'suite_config.json'),manifest['images']))),16224)

    def test_blind_stage_file_contract(self):
        source=(ROOT/'suite_decisions.py').read_text()
        self.assertNotIn("/'labels'",source)
        self.assertNotIn("['path']",source)
        self.assertNotIn("state['csi_age",source)

    def test_noise_overlap_is_absolute_time(self):
        from suite_codec import Codec
        codec=Codec.__new__(Codec)
        codec.active_case,codec.blocks=None,{}
        a=codec.noise('unit',250,64).numpy()
        b=codec.noise('unit',270,20).numpy()
        np.testing.assert_array_equal(a[20:40],b)
        other=codec.noise('different',270,20).numpy()
        self.assertFalse(np.array_equal(b,other))
        np.testing.assert_array_equal(b,codec.noise('unit',270,20).numpy())

    def test_pilot_refresh_and_count(self):
        from suite_codec import Codec
        codec=Codec.__new__(Codec)
        codec.channels={'fake':{'AgingTime':np.tile(np.arange(128)*.03125,4)[None,:]}}
        self.assertEqual(codec.refresh_after('fake',130),256)
        self.assertEqual(codec.refresh_after('fake',128),128)
        self.assertEqual(codec.pilots('fake',100,64),2)
        self.assertEqual(codec.pilots('fake',128,32),1)

    def test_official_rounding_and_tie(self):
        body={'model':'jev-1.13.0','answers':{'action':{'type':'choice','choice':ACTIONS[1],
            'probabilities':dict(zip(ACTIONS,[.33,.33,.09,.08,.08,.08])),'confidence':.2}}}
        for a in ACTIONS[:-1]: body['answers']['benefit_'+a]={'type':'noul','noul':.5}
        validated=validate_official(body,'jev-1.13.0')
        self.assertEqual(validated['action'],ACTIONS[1])
        self.assertAlmostEqual(sum(validated['probabilities'].values()),1)
        body['answers']['action']['choice']=ACTIONS[-1]
        with self.assertRaisesRegex(ValueError,'choice_not_maximum'): validate_official(body,'jev-1.13.0')
        body['answers']['action']['probabilities'][ACTIONS[0]]=-.1
        with self.assertRaisesRegex(ValueError,'probability_range'): validate_official(body,'jev-1.13.0')

if __name__=='__main__': unittest.main()
