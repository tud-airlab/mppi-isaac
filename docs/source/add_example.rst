Add your own custom example
=================

The ``mppi-isaac`` package is written to make it easy to add your own custom examples. 
We will guide you through the process of adding a new example to the package.

1. Create a new directory in the ``examples`` directory. 
----------------
   The name of the directory will be the name of the example.

2. configuration
----------------

The configuration of the example is done using the `Hydra <https://hydra.cc/docs/intro>`_ configuration system.
This enables you to easily change the configuration of the example from the command line, and reuse parts of the configuration in other examples in a modular way.
The reusable configuration is stored in the ``conf`` directory. We divide configuration settings into 3 parts: ``mppi``, ``isaacgym`` and ``actors``.
Using your examples top level configuration file, you can specify which configuration settings you want to use for each part.

For example, the following configuration file will use the ``boxer_reach`` configuration for the ``mppi`` part, and the ``normal`` configuration for the ``isaacgym`` part.

*Note that we need to let hydra know where the reusable configuration is stored. This is done using the ``searchpath`` setting.*

.. code-block:: yaml

    defaults:
    - mppi: boxer_reach
    - isaacgym: normal

    render: true
    n_steps: 5
    nx: 4

    actors: ['boxer', 'wall', 'goal']
    initial_actor_positions: [[0.0, 0.0, 0.05]]

    hydra:
        searchpath:
            - pkg://conf

3. Create the world file:
-------------------------

Create a new python file in your example directory called ``world.py``. 
The world file represents your 'real' world and should call the planner each step to compute the action to apply to the robot.

.. code-block:: python

    @hydra.main(version_base=None, config_path=".", config_name="config_boxer_reach")
    def run(cfg):
        sim = IsaacGymWrapper(
            cfg.isaacgym,
            actors=cfg.actors,
            ...
        )

        planner = zerorpc.Client()
        planner.connect("tcp://127.0.0.1:4242")

        for _ in range(cfg.n_steps):
            # Compute action
            action = bytes_to_torch(
                planner.compute_action_tensor(
                    torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state)
                )
            )

            # Apply action
            sim.apply_robot_cmd(action.unsqueeze(0))

            # Step simulator
            sim.step()

    if __name__ == "__main__":
        res = run()

4. Create the planner file:
---------------------------

Create a new python file in your example directory called ``planner.py``.
This file initializes the planner and defines the objective function.
Since you have access to a handle of sim (which is an IsaacGymWrapper class object), you can use the simulator to easily define complex cost functions

.. code-block:: python

    class Objective(object):
        ...
        def compute_cost(self, sim):
            ...
            return total_cost


    @hydra.main(version_base=None, config_path=".", config_name="config_albert")
    def run(cfg: ExampleConfig):
        objective = Objective(cfg)
        planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
        planner.bind("tcp://0.0.0.0:4242")
        planner.run()


    if __name__ == "__main__":
        run()

5. Run the example
------------------

To run your example you need two terminals.
One for the planner and one for the world.
Simply executing the python scripts in each terminal will run the example.

*Note make sure you are in the poetry virtual environment*

.. code-block:: bash

    # Terminal 1
    python planner.py

    # Terminal 2
    python world.py