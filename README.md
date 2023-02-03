# Installation

This project requires the sourcec code of isaacgym inside the folder
thirdparties. Download it from https://developer.nvidia.com/isaac-gym and place
it there. Then you can proceed with the installation descibed below.

Use [poetry](https://python-poetry.org/docs/) to install the package with:
```bash
poetry install
```

Access the virtual environment using
```bash
poetry shell
```

You can also install using pip:
```bash
pip install .
```

# Developers

## Installation

You can install useful developer software using.
```bash
poetry install --with dev
```

## Testing

Test changes using pytest
```bash
poetry run pytest examples
```
