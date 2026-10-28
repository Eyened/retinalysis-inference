import pytest
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from rtnls_inference.ensembles import base


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
@pytest.mark.parametrize("missing_config", [False, True])
def test_hf_download_requests_root_config_without_reading_it(
    monkeypatch, tmp_path, suffix, missing_config
):
    calls = []
    artifact = tmp_path / ("model" + suffix)
    config = tmp_path / "config.yaml"
    config.write_text("not valid yaml: [")

    def download(*, repo_id, filename):
        calls.append((repo_id, filename))
        if filename == "config.yaml":
            if missing_config:
                raise EntryNotFoundError("not found")
            return str(config)
        return str(artifact)

    loaded = []

    def load(cls, path, **kwargs):
        loaded.append((path, kwargs))
        return "loaded"

    monkeypatch.setattr(base, "hf_hub_download", download)
    monkeypatch.setattr(base.Ensemble, "from_torchscript", classmethod(load))
    monkeypatch.setattr(base.Ensemble, "from_onnx", classmethod(load))
    assert (
        base.Ensemble.from_huggingface(f"org/repo:nested/model{suffix}", tta=False)
        == "loaded"
    )
    assert calls == [("org/repo", f"nested/model{suffix}"), ("org/repo", "config.yaml")]
    assert loaded == [(str(artifact), {"tta": False})]


def test_hf_config_only_ignores_missing_file(monkeypatch):
    def download(*, repo_id, filename):
        if filename == "config.yaml":
            raise HfHubHTTPError("access denied")
        return "/tmp/model.pt"

    monkeypatch.setattr(base, "hf_hub_download", download)
    with pytest.raises(HfHubHTTPError, match="access denied"):
        base.Ensemble.from_huggingface("org/repo:model.pt")
