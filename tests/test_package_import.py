from importlib.metadata import version

import graphnamlss


def test_package_import_exposes_installed_version():
    assert graphnamlss.__version__ == version("graphnamlss")
