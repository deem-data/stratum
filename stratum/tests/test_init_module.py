import stratum


def test_versions_contains_strings():
    versions = stratum.versions()
    assert set(versions.keys()) == {"stratum", "skrub"}
    assert all(isinstance(v, str) and v for v in versions.values())
    module_dir = stratum.__dir__()


