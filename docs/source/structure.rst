Structure
===============

The ``mppiisaac`` package is built using the modular mppi implementation found in the ``mppi-torch`` package. 
The goal of the ``mppiisaac`` package is to provide a simple interface to the ``mppi-torch`` package for use with the isaacgym simulator as the dynamical model.

The ``mppiisaac`` package is structured as follows:

isaacgym_wrapper
----------------

The ``isaacgym_wrapper`` module contains the ``IsaacGymWrapper`` class which is a wrapper around the isaacgym simulator.
It provides a set of easy to use methods for interacting with the isaacgym simulator. See the api documentation for more details.

mppi_isaac
----------

The ``mppi_isaac`` module contains the ``MPPIIsaac`` class which connects the mppi implementation of ``mppi_torch`` package and the simulator interface of ``isaacgym_wrapper``.
This class provides a simple interface for running mppi on the isaacgym simulator. See the api documentation for more details.

assets
------

The ``assets`` directory contains the assets used by the isaacgym simulator, which mostly consist of the urdf files for the robot models.

conf
----
The ``conf`` directory contains reusable configuration files for the isaacgym simulator.
See the configurations section for more details.

examples
--------
The ``examples`` directory contains example scripts for running mppi on the isaacgym simulator.
Every example is a directory containing a ``world.py`` and a ``planner.py`` file. 

**This is because we use the isaacgym simulator both to compute the action as well as to simulate the real world.**
**However, isaacgym does not support running two instances of the simulator from a single script, therefore we use have two scripts and communicate via the zerorpc package**