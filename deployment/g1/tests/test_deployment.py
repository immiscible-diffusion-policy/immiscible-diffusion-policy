import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from idp_g1.infer import AsyncPolicy, G1DPPolicy
from idp_g1.robot_interface import MockRobot, SafetyLimiter, urdf_arm_limits
from idp_g1.run_policy import run_session
from test_infer_logic import FakePolicy


class DeploymentTests(unittest.TestCase):
    def test_joint_limits_step_limits_and_invalid_values(self):
        limiter = SafetyLimiter(np.tile([-1., 1.], (14, 1)), max_step_rad=.08)
        limiter.reset(np.zeros(16))
        a = limiter(np.full(16, 2.))
        np.testing.assert_allclose(a[:14], .08)
        np.testing.assert_array_equal(a[14:], [1., 1.])
        for _ in range(20):
            a = limiter(np.full(16, 2.))
        self.assertLessEqual(a[:14].max(), .980001)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                limiter(np.full(16, value))
        with self.assertRaises(ValueError):
            limiter.reset(np.full(16, 5.))

    def test_urdf_uses_explicit_joint_order(self):
        names = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
        joints = [f'<joint name="{side}_{name}_joint"><limit lower="{-i-1}" upper="{i+1}"/></joint>'
                  for i, (side, name) in enumerate((s, n) for s in ['left', 'right'] for n in names)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'robot.urdf'
            path.write_text('<robot>' + ''.join(reversed(joints)) + '</robot>')
            actual = urdf_arm_limits(str(path))
        np.testing.assert_array_equal(actual[:, 1], np.arange(1, 15))

    def test_worker_failure_propagates(self):
        p = FakePolicy()
        def fail(obs):
            raise ValueError('inference failure')
        p.predict = fail
        ap = AsyncPolicy(p)
        ap.tick({'qpos': np.zeros(16)}, 0)
        ap.join()
        with self.assertRaisesRegex(RuntimeError, 'Background policy'):
            ap.tick({'qpos': np.zeros(16)}, 1)

    def test_release_does_not_publish_hold_after_hand_off(self):
        from idp_g1.g1_real import UnitreeG1Robot
        from unittest.mock import Mock
        robot = UnitreeG1Robot.__new__(UnitreeG1Robot)
        robot._released = True
        robot.arm = Mock()
        robot.hand = None
        robot.feed = Mock()
        robot.stop = Mock()
        with patch('idp_g1.g1_real._clear_last_target'):
            robot.close()
        robot.stop.assert_not_called()
        robot.feed.close.assert_called_once()

    def test_checkpoint_preprocessing_and_mock_episode(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        shape = {'obs': {'ego_cam': {'shape': [3, 32, 32], 'type': 'rgb'},
                         'wide_cam': {'shape': [3, 32, 32], 'type': 'rgb'},
                         'agent_pos': {'shape': [16], 'type': 'low_dim'}},
                 'action': {'shape': [16]}}
        cfg = OmegaConf.create({
            'shape_meta': shape, 'n_obs_steps': 2, 'n_latency_steps': 0,
            'training': {'use_ema': True},
            'policy': {
                '_target_': 'diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy',
                'shape_meta': '${shape_meta}', 'horizon': 8, 'n_action_steps': 4, 'n_obs_steps': 2,
                'num_inference_steps': 2, 'diffusion_step_embed_dim': 16, 'down_dims': [16, 32],
                'n_groups': 8, 'use_immiscible': True, 'immiscible_start_epoch': 1,
                'noise_scheduler': {'_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                                    'num_train_timesteps': 100, 'prediction_type': 'epsilon'},
                'obs_encoder': {
                    '_target_': 'diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder',
                    'shape_meta': '${shape_meta}', 'crop_shape': [28, 28], 'random_crop': True,
                    'use_group_norm': True, 'imagenet_norm': True,
                    'rgb_model': {'_target_': 'diffusion_policy.model.vision.model_getter.get_resnet',
                                  'name': 'resnet18', 'weights': None}}}})
        model = hydra.utils.instantiate(cfg.policy)
        normalizer = LinearNormalizer()
        values = np.stack([np.full(16, -2.), np.full(16, 2.)]).astype(np.float32)
        normalizer.fit({'agent_pos': values, 'action': values}, last_n_dims=1)
        for key in ('ego_cam', 'wide_cam'):
            normalizer[key] = get_image_range_normalizer()
        model.set_normalizer(normalizer)
        batch = {'obs': {'ego_cam': torch.rand(2, 2, 3, 32, 32),
                         'wide_cam': torch.rand(2, 2, 3, 32, 32), 'agent_pos': torch.zeros(2, 2, 16)},
                 'action': torch.rand(2, 8, 16)}
        from immiscible_diffusion_policy import match_noise
        with patch('diffusion_policy.policy.diffusion_unet_image_policy.match_noise', wraps=match_noise) as matching:
            loss = model.compute_loss(batch, epoch=0)
            self.assertEqual(matching.call_count, 0)
            loss = model.compute_loss(batch, epoch=1)
            self.assertEqual(matching.call_count, 1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            ckpt = path / 'checkpoint.ckpt'
            torch.save({'cfg': cfg, 'state_dicts': {'ema_model': model.state_dict()}},
                       ckpt, pickle_module=dill)
            config = {'fps': 30, 'spec': {'binary_action_dims': [14, 15]},
                      'init_pose': [0.] * 16, 'blend_steps': 4, 'max_step_rad': .08}
            meta = path / 'metadata.json'
            meta.write_text(json.dumps(config))
            p = G1DPPolicy(str(ckpt), device='cpu', metadata_path=str(meta))
            bgr = np.zeros((48, 48, 3), dtype=np.uint8)
            bgr[:, :, 0] = 255
            obs = {'ego_cam': bgr, 'wide_cam': bgr, 'qpos': np.zeros(16, np.float32)}
            p.last_grasp[:] = [1., 0.]
            prepped = p.prep_obs(obs)
            self.assertEqual(prepped['ego_cam'].shape, (3, 32, 32))
            self.assertEqual(prepped['ego_cam'][0].mean(), 1.)
            self.assertEqual(prepped['ego_cam'][2].mean(), 0.)
            np.testing.assert_array_equal(prepped['agent_pos'][14:], [1., 0.])
            p.reset()
            actions = p.step(obs)
            self.assertEqual(actions.shape, (4, 16))
            self.assertTrue(np.isfinite(actions).all())
            self.assertTrue(np.isin(actions[:, 14:], [0, 1]).all())
            # The serialized model and reloaded model have identical inference with a fixed RNG.
            model.eval()
            p.reset()
            prepped = p.prep_obs(obs)
            tensors = {k: torch.from_numpy(np.stack([v, v]))[None] for k, v in prepped.items()}
            with torch.no_grad():
                torch.manual_seed(10)
                expected = model.predict_action(tensors)['action_pred'][0].numpy()
                torch.manual_seed(10)
                actual = p.predict({k: np.stack([v, v]) for k, v in prepped.items()})
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            robot = MockRobot({'ego_cam': bgr[None], 'wide_cam': bgr[None]}, np.zeros(16))
            output = path / 'episode'
            run_session(robot, p, config, np.tile([-3., 3.], (14, 1)), output,
                        seconds=.6, mock=True, video=False)
            log = np.load(output / 'ep001/log.npz')
            self.assertTrue(np.isfinite(log['cmd']).any())
            self.assertTrue((output / 'ep001/summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
