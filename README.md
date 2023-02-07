# Installation

This project requires the sourcec code of isaacgym inside the folder
thirdparties. Download it from https://developer.nvidia.com/isaac-gym and place
it there. Then you can proceed with the installation descibed below.

## Developers

For the time being, you can install useful developer software using in a poetry virtual environment:
```bash
poetry install --with dev
```
Bare in mind that the installation might take several minutes the first time. But it's worth it.

Later on, you should also be able to just use [poetry](https://python-poetry.org/docs/) to install the package with:
```bash
poetry install
```

Access the virtual environment using
```bash
poetry shell
```

Alternatively, you can also install at the system level using pip:
```bash
pip install .
```

## Testing

Test changes using pytest
```bash
cd examples
poetry run pytest
```

## Troubleshooting
If you have an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched