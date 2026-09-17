"""CPU contracts for identity/role deterministic observation sampling."""
import dataclasses
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .oracle_packet import FMapZeroContextPacket, _derive_frontend_state, extract_frontend_packet
from .observation_sampling import observation_rng_scope, observation_seed, validate_scientific_seed
from .test_h2 import frame

def cpu_patchify(value, centroids, radius):
    """Integer-coordinate indexing substitutes only the native CUDA primitive."""
    return torch.stack([torch.stack([
        value[n, :, int(y) - radius:int(y) + radius + 1, int(x) - radius:int(x) + radius + 1]
        for x, y in points]) for n, points in enumerate(centroids)])


def cpu_coords(disps, device):
    b, n, h, w = disps.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack((x.expand(b, n, h, w), y.expand(b, n, h, w), disps), dim=2)
    return grid, None


def draw_observation(identity, scientific_seed=1234):
    fmap = torch.arange(2 * 32 * 40, dtype=torch.float32).reshape(1, 1, 2, 32, 40)
    modules = {"dpvo": SimpleNamespace(altcorr=SimpleNamespace(patchify=cpu_patchify)),
               "dpvo.utils": SimpleNamespace(coords_grid_with_index=cpu_coords)}
    with patch.dict(sys.modules, modules):
        packet, stats = _derive_frontend_state(FMapZeroContextPacket(fmap), identity, scientific_seed,
            patches_per_image=32, patch_size=3, context_dim=4)
    # Same native random-depth operation and shape; DPVO itself is not imported.
    with observation_rng_scope(identity, scientific_seed):
        depth = torch.rand_like(packet.patch_xy[:, :, 0, 0, 0, None, None])
        python_value, numpy_value = random.random(), np.random.rand(3)
    return stats["centroids"], packet.patch_xy, packet.gmap, depth, torch.tensor(python_value), torch.from_numpy(numpy_value)


class IdentitySamplingTests(unittest.TestCase):
    def test_frozen_identity_role_seed_values_and_validation(self):
        expected = {0: 1199367982294886610, 1: 5040992563004887393,
                    12: 7093586990363723977, 17: 1653243464083768270}
        for index, value in expected.items():
            self.assertEqual(observation_seed(frame(index), 1234, "dpvo_track_packet"), value)
            self.assertNotEqual(observation_seed(frame(index), 1234, "other_role"), value)
        for invalid in (True, -1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                validate_scientific_seed(invalid)

    def assertDrawEqual(self, first, second):
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_same_identity_constructs_identical_actual_hidden_frontend(self):
        a = draw_observation(frame(17))
        torch.rand(711)
        np.random.rand(31)
        random.random()
        self.assertDrawEqual(a, draw_observation(frame(17)))
        centroids, xy, gmap, depth, *_ = a
        self.assertEqual(xy.shape, (1, 32, 2, 3, 3))
        self.assertEqual(gmap.shape, (1, 32, 2, 3, 3))
        self.assertTrue(((centroids[..., 0] >= 1) & (centroids[..., 0] < 39)).all())
        self.assertTrue(((centroids[..., 1] >= 1) & (centroids[..., 1] < 31)).all())
        self.assertTrue(((depth >= 0) & (depth < 1)).all())

    def test_distinct_identity_seed_sequence_and_role_have_distinct_streams(self):
        identity = frame(17)
        a = draw_observation(identity)
        for other in (frame(18), dataclasses.replace(identity, sequence="MH_02_easy")):
            b = draw_observation(other)
            self.assertFalse(torch.equal(a[0], b[0]))
            self.assertFalse(torch.equal(a[3], b[3]))
        self.assertFalse(torch.equal(a[0], draw_observation(identity, 42)[0]))
        self.assertNotEqual(observation_seed(identity, 1234, "depth"), observation_seed(identity, 1234, "query"))

    def test_insertion_order_and_discard_subset_do_not_advance_other_streams(self):
        identities = [frame(i) for i in range(6)]
        baseline = {i.key: draw_observation(i) for i in identities}
        for order in (identities[::-1], identities[::2], identities[1:] + identities[:1]):
            for identity in order:
                torch.rand(73)
                random.random()
                np.random.rand(19)
                self.assertDrawEqual(baseline[identity.key], draw_observation(identity))

    def test_scopes_restore_python_numpy_torch_on_exception(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with observation_rng_scope(frame(9), 234):
                random.random(), np.random.rand(4), torch.rand(4)
                raise RuntimeError("test")
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_anchor_frontend_native_candidate_sampling_is_identity_scoped(self):
        def patchify(image, patches_per_image, centroid_sel_strat, return_color):
            self.assertEqual((patches_per_image, centroid_sel_strat, return_color), (32, "RANDOM", True))
            x, y = torch.randint(1, 39, (1, 32)), torch.randint(1, 31, (1, 32))
            xy = torch.stack((x, y), dim=2).float()[..., None, None].expand(1, 32, 2, 3, 3)
            return torch.zeros(1, 1, 2, 32, 40), torch.zeros(1, 32, 2, 3, 3), torch.zeros(1, 32, 4, 1, 1), xy, None, torch.zeros(1, 32, 3)
        slam = SimpleNamespace(M=32, cfg=SimpleNamespace(MIXED_PRECISION=False, CENTROID_SEL_STRAT="RANDOM"),
                               network=SimpleNamespace(patchify=patchify))
        image = torch.zeros(3, 128, 160)
        first = extract_frontend_packet(slam, image, frame(4), 55)
        extract_frontend_packet(slam, image, frame(5), 55)
        torch.rand(100)
        second = extract_frontend_packet(slam, image, frame(4), 55)
        self.assertTrue(torch.equal(first.patch_xy, second.patch_xy))

    def test_seed_is_stable_in_fresh_process(self):
        code = ("from research.src.protocol import FrameIdentity; "
                "from research.src.observation_sampling import observation_seed; "
                f"print(observation_seed(FrameIdentity(**{dataclasses.asdict(frame(12))!r}), 77, 'dpvo_track_packet'))")
        value = subprocess.check_output([sys.executable, "-B", "-c", code], text=True)
        self.assertEqual(int(value.strip()), observation_seed(frame(12), 77, "dpvo_track_packet"))

