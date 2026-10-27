import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ovr.cloud_backend import (access, download_transport, snapshot_with_fallback,
                               verify_source, PUBLIC_MIRROR, OFFICIAL_ENDPOINT)


class MirrorTests(unittest.TestCase):
    def test_download_workers_can_be_reduced_without_changing_revision(self):
        with patch.dict(os.environ, {"OVR_DOWNLOAD_WORKERS": "1"}), patch("huggingface_hub.snapshot_download") as mock:
            snapshot_with_fallback({"repo": "owner/model", "revision": "pinned"}, Path("unused"), OFFICIAL_ENDPOINT)
        self.assertEqual(mock.call_args.kwargs["max_workers"], 1)
        self.assertEqual(mock.call_args.kwargs["revision"], "pinned")
        with patch.dict(os.environ, {"OVR_DOWNLOAD_WORKERS": "20"}), patch("huggingface_hub.snapshot_download") as mock:
            with self.assertRaisesRegex(ValueError, "WORKERS"):
                snapshot_with_fallback({"repo": "owner/model", "revision": "pinned"}, Path("unused"), OFFICIAL_ENDPOINT)
            mock.assert_not_called()

    def test_only_public_coldstart_uses_mirror(self):
        with patch.dict(os.environ, {"OVR_PUBLIC_MODEL_ENDPOINT": PUBLIC_MIRROR}):
            self.assertEqual(download_transport("coldstart", {"gated": False}), PUBLIC_MIRROR)
            self.assertEqual(download_transport("coldstart", {"gated": True}), OFFICIAL_ENDPOINT)
            self.assertEqual(download_transport("rl", {"gated": False}), OFFICIAL_ENDPOINT)
            self.assertEqual(download_transport("coldstart", {}), OFFICIAL_ENDPOINT)

    def test_official_override_and_unknown_endpoint(self):
        with patch.dict(os.environ, {"OVR_PUBLIC_MODEL_ENDPOINT": OFFICIAL_ENDPOINT}):
            self.assertEqual(download_transport("coldstart", {"gated": False}), OFFICIAL_ENDPOINT)
        with patch.dict(os.environ, {"OVR_PUBLIC_MODEL_ENDPOINT": "https://example.invalid"}):
            with self.assertRaises(ValueError):
                download_transport("coldstart", {"gated": False})

    def test_mirror_never_gets_implicit_token_and_falls_back_once(self):
        with patch("huggingface_hub.snapshot_download", side_effect=[ConnectionError(), "done"]) as mock:
            result = snapshot_with_fallback({"repo": "owner/model", "revision": "pinned"}, Path("unused"), PUBLIC_MIRROR)
        self.assertEqual(mock.call_count, 2)
        first, second = [c.kwargs for c in mock.call_args_list]
        self.assertIs(first["token"], False)
        self.assertEqual(first["endpoint"], PUBLIC_MIRROR)
        self.assertEqual(second["endpoint"], OFFICIAL_ENDPOINT)
        self.assertIsNone(second["token"])
        self.assertEqual(result["fallback_from"], PUBLIC_MIRROR)
        self.assertEqual(first["revision"], second["revision"])

    def test_official_failure_is_not_retried_as_a_mirror(self):
        with patch("huggingface_hub.snapshot_download", side_effect=ConnectionError()) as mock:
            with self.assertRaises(ConnectionError):
                snapshot_with_fallback({"repo": "owner/model", "revision": "pin"}, Path("unused"), OFFICIAL_ENDPOINT)
        self.assertEqual(mock.call_count, 1)

    def test_mirror_config_and_weight_bytes_must_match_official_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = b'{"model_type":"qwen2_5_vl"}'
            weight = b"test fixture, not real model weights"
            (root / "config.json").write_bytes(config)
            (root / "model.safetensors").write_bytes(weight)
            files = [{"name": "config.json", "size": len(config),
                      "git_blob_sha1": hashlib.sha1(b"blob " + str(len(config)).encode() + b"\0" + config).hexdigest()},
                     {"name": "model.safetensors", "size": len(weight), "sha256": hashlib.sha256(weight).hexdigest()}]
            self.assertEqual(len(verify_source(root, files)), 2)
            (root / "config.json").write_bytes(config.replace(b"qwen", b"fake"))
            with self.assertRaisesRegex(ValueError, "Git hash"):
                verify_source(root, files)
            (root / "config.json").write_bytes(config)
            (root / "model.safetensors").write_bytes(b"X" * len(weight))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_source(root, files)

    def test_access_authentication_stays_official_despite_global_mirror(self):
        config = {"models": {"coldstart": {"repo": "owner/public", "revision": "pin"}}}
        sibling = SimpleNamespace(rfilename="model.safetensors", size=5, lfs={"sha256": "a" * 64}, blob_id=None)
        info = SimpleNamespace(sha="pin", gated=False, siblings=[sibling])
        with patch.dict(os.environ, {"HF_ENDPOINT": PUBLIC_MIRROR}), patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.get_hf_file_metadata") as head:
            api.return_value.model_info.return_value = info
            result = access(config)
        self.assertTrue(result["complete"])
        self.assertEqual(api.call_args.kwargs["endpoint"], OFFICIAL_ENDPOINT)
        self.assertTrue(all(c.args[0].startswith(OFFICIAL_ENDPOINT + "/") for c in head.call_args_list))
