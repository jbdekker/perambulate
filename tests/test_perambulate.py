import toml

from perambulate import __version__


def test_version():
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    assert __version__ == pyproject["tool"]["poetry"]["version"]
