"""Guards for the single algorithm source (algorithm/). Offline, no images."""
import hashlib
import json
import shutil
from pathlib import Path

ALG = Path(__file__).resolve().parents[1] / 'algorithm'
REPO = ALG.parents[2]
PRODUCTION = {'baseline_peaks', 'chromatic_ink', 'faint_ink', 'uneven_photo', 'grayscale_contrast', 'same_appearance'}


def _lf_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def test_frozen_peaks34_matches_manifest():
    root = ALG / 'research_methods' / 'frozen_peaks34'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['commit'].startswith('323b4e6')
    assert {f: _lf_sha(root / f) for f in manifest['files']} == manifest['files']


def test_peaks34_reference_is_pinned():
    # Historical control: changing it must be a deliberate new method, not an edit.
    assert _lf_sha(ALG / 'research_methods' / 'peaks34_reference.py') == \
        '52600791d564bc153266cf3d15fe2ee0504bdb1c8f25595591a63c8a2e36d25b'


def test_production_registry_is_the_six_methods():
    from research_methods.registry import METHODS
    assert set(METHODS) == PRODUCTION


def test_research_registry_reuses_production_modules():
    from research_methods import registry, research_registry
    from research_methods.submitted_adapter import SUBMITTED_METHODS
    assert all(research_registry.METHODS[n] is m for n, m in registry.METHODS.items())
    assert set(research_registry.METHODS) - PRODUCTION == {'local_ink_graph', 'peaks34_reference'}
    assert len(SUBMITTED_METHODS) == 8


def test_imports_resolve_to_algorithm_dir():
    import ecg, research_methods, review, selector.score as score
    for module in (ecg, research_methods, review, score):
        assert Path(module.__file__).resolve().is_relative_to(ALG), module.__file__


def test_digest_ignores_line_endings_and_caches(tmp_path):
    import source_info
    copy = tmp_path / 'algorithm'
    shutil.copytree(ALG, copy, ignore=shutil.ignore_patterns('__pycache__'))
    for p in copy.rglob('*.py'):
        p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    (copy / 'ecg' / '__pycache__').mkdir(exist_ok=True)
    (copy / 'ecg' / '__pycache__' / 'x.pyc').write_bytes(b'x')
    assert source_info.digest(copy) == source_info.digest(ALG)
    (copy / 'ecg' / 'config.py').write_text('changed')
    assert source_info.digest(copy) != source_info.digest(ALG)


def test_no_second_copy_of_the_algorithm_in_this_repo():
    skip = {'.git', 'build', 'node_modules', '.terraform'}
    found = [p for name in ('research_methods', 'frozen_peaks34') for p in REPO.rglob(name)
             if p.is_dir() and not skip & set(p.relative_to(REPO).parts)]
    allowed = {ALG / 'research_methods', ALG / 'research_methods' / 'frozen_peaks34'}
    assert set(found) <= allowed, found
