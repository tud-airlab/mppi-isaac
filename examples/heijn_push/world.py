from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
import hydra
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.transport import torch_to_bytes, bytes_to_torch
import time


@hydra.main(version_base=None, config_path=".", config_name="config_heijn_push")
def run_heijn_robot(cfg: ExampleConfig):

    sim = IsaacGymWrapper(
        cfg.isaacgym,
        actors=cfg.actors,
        init_positions=cfg.initial_actor_positions,
        num_envs=1,
        viewer=True,
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")

    t = time.time()
    while True:
        # Compute action
        action = bytes_to_torch(
            planner.compute_action_tensor(
                torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state)
            )
        )

        # Apply action
        sim.apply_robot_cmd_velocity(action)

        # Step simulator
        sim.step()

        # Visualize samples
        rollouts = bytes_to_torch(planner.get_rollouts())
        sim._gym.clear_lines(sim.viewer)
        sim.draw_lines(rollouts)

        # Timekeeping
        actual_dt = time.time() - t
        rt = cfg.isaacgym.dt / actual_dt
        if rt > 1.0:
            time.sleep(cfg.isaacgym.dt - actual_dt)
            actual_dt = time.time() - t
            rt = cfg.isaacgym.dt / actual_dt
        print(f"FPS: {1/actual_dt}, RT={rt}")
        t = time.time()


if __name__ == "__main__":
    res = run_heijn_robot()
