#### Resources for creating a new Python project:
- [Understanding modules, packages, and imports](https://dagster.io/blog/python-packages-primer-1)
- [Understanding Python dependencies and virtual environments](https://dagster.io/blog/python-packages-primer-2)
- [Configuring setuptools using pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

#### Managing dependencies using pyproject.toml
Need to include the following in your `pyproject.toml` file if you want to use setuptools:
```
[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"
```

#### Installing dependencies
Navigate to the folder containing your pyproject.toml file and then perform what is known as editable pip install:
`pip install -e .`