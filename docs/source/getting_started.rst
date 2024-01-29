Getting started
===============

This section will guide you through the installation and basic usage of our package.

Pre-requisites
--------------

- python >=3.8,<3.9
- pip3 or poetry
- NVIDIA graphics card for GPU pipeline (optional)


Installation
---------------------------

Clone the repository and navigate to the project root:

.. code-block:: bash

    git clone https://github.com/tud-airlab/mppi-isaac.git
    cd mppi-isaac

*NOTE: To use the GPU pipeline (default) you need an NVIDIA graphics card. If you do not have one, you can use the CPU pipeline, with fewer samples.*

This project requires the source code of IsaacGym inside the folder
`thirdparty`. Download it from https://developer.nvidia.com/isaac-gym, extract it, and place
it in `mppi-isaac/thirdparty`. Then you can proceed with the installation described below. 


Virtual environment (advised)
---------------------------

You can install the necessary dependencies using `poetry <https://python-poetry.org/docs/>` virtual environment. After installing poetry, move into ``mppi-isaac`` and run

.. code-block:: bash
    
    poetry install --with dev

Bear in mind that the installation might take several minutes the first time. But it's worth it.

Access the virtual environment using::

.. code-block:: bash

    poetry shell

Virtual environment - Pycharm
------------------------------

If you are using Pycharm (professional), first add a new Python interpreter via 'Add New Interpreter', 'Add local interpreter" and create a virtual environment within the mppi-isaac folder called 'venv'. Then install the necessary dependencies using `poetry <https://python-poetry.org/docs/>`_ in the virtual environment

.. code-block:: bash

    poetry install --with dev

Test the installation
---------------------

Test that everything is properly set up, use pytest

.. code-block:: bash

    cd examples
    poetry run pytest

System-level installation
-------------------------

Alternatively, you can also install at the system level using pip, even though we advise using the virtual environment

.. code-block:: bash

    pip install .

Troubleshooting
---------------

If you have an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched for every terminal

Running the examples
--------------------

Access the virtual environment if installed with poetry (with ``poetry shell``). You can run two types of examples, either the ones using IsaacGym or the ones using Pybullet. In the ``examples`` folder, you find all the scripts. The ones related to IsaacGym contain either "server" or "client" in the name.

IsaacGym examples
~~~~~~~~~~~~~~~~~

To run the examples with IsaacGym (for example for non-prehensile pushing and whole-body control), you need two terminals because it is required to run a "server" and a "client" script. In the first one run the server

.. code-block:: bash

    python3 heijn_push_server.py

In the second one the client

.. code-block:: bash

    python3 heijn_push_client.py

Pybullet examples
~~~~~~~~~~~~~~~~~

Simply run one of the example scripts which use Pybullet, for instance for the panda robot

.. code-block:: bash

    python3 panda_robot_with_obstacles.py
