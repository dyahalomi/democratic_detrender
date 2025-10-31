
def test_import_helper_functions_module():
    import democratic_detrender.helper_functions as hf  # noqa: F401


def test_helper_functions_has_expected_symbols():
    import democratic_detrender.helper_functions as hf

    for name in [
        "durbin_watson",
        "get_detrended_lc",
        "determine_cadence",
        "find_nearest",
        "ensemble_step",
    ]:
        assert hasattr(hf, name), f"helper_functions.{name} should exist"


def test_package_top_level_import():
    # If __init__.py pulls heavy deps and you don't want this,
    # delete this test.
    import democratic_detrender  # noqa: F401
