from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from isaacgym import gymapi
import hydra
import optuna
import zerorpc
import torch
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.transport import torch_to_bytes, bytes_to_torch
import time
import numpy as np


class Tuning:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.viewer = True
        self.sim = IsaacGymWrapper(
            cfg.isaacgym,
            actors=cfg.actors,
            init_positions=cfg.initial_actor_positions,
            num_envs=1,
            viewer=self.viewer,
        )

        self.planner = zerorpc.Client()
        self.planner.connect("tcp://127.0.0.1:4242")
        print("Mppi server found!")

        if self.viewer: 
            self.sim._gym.viewer_camera_look_at(
                self.sim.viewer,
                None,
                gymapi.Vec3(1.0, 6.5, 4),
                gymapi.Vec3(1.0, 0, 0),  # CAMERA LOCATION, CAMERA POINT OF INTEREST
            )

    def tune(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=20)
        study.best_params
    
    def objective(self, trial):
        weights = {
            "robot_to_block": trial.suggest_float("robot_to_block", 0.0, 100.0),
            "block_to_goal": trial.suggest_float("block_to_goal", 0.0, 20.0),
            "collision": trial.suggest_float("collision", 0.0, 100.0),
            "robot_ori": trial.suggest_float("robot_ori", 0.0, 5.0)
        }
        noise_sigma = trial.suggest_float("noise_sigma", 0.01, 2.0)
        mppi_params = {
            "noise_sigma": (np.eye(self.cfg.nx//2)*noise_sigma).tolist()
        }

        # TODO: incorporate latency into this
        self.planner.update_weights(weights)
        self.planner.update_mppi_params(mppi_params)
        cost = self.run()
        self.reset()
        return cost

    def reset(self):
        # self.sim.stop_sim()
        # self.sim.start_sim()
        self.sim.reset_to_initial_poses()

        if self.viewer: 
            self.sim._gym.viewer_camera_look_at(
                self.sim.viewer,
                None,
                gymapi.Vec3(1.0, 6.5, 4),
                gymapi.Vec3(1.0, 0, 0),  # CAMERA LOCATION, CAMERA POINT OF INTEREST
            )

    def run(self):
        obj = 0
        t = time.time()
        for _ in range(200):
            # Compute action
            action = bytes_to_torch(
                self.planner.compute_action_tensor(
                    torch_to_bytes(self.sim._dof_state), torch_to_bytes(self.sim._root_state)
                )
            )

            # Apply action
            self.sim.set_dof_velocity_target_tensor(action)

            # Step simulator
            self.sim.step()

            # Visualize samples
            if self.viewer:
                rollouts = bytes_to_torch(self.planner.get_rollouts())
                self.sim._gym.clear_lines(self.sim.viewer)
                self.sim.draw_lines(rollouts)

            # Timekeeping
            actual_dt = time.time() - t
            rt = self.cfg.isaacgym.dt / actual_dt
            if rt > 1.0 and self.viewer:
                time.sleep(self.cfg.isaacgym.dt - actual_dt)
                actual_dt = time.time() - t
                rt = self.cfg.isaacgym.dt / actual_dt
            if self.viewer:
                print(f"FPS: {1/actual_dt}, RT={rt}")
            t = time.time()

            block_pos = self.sim.get_actor_position_by_name("panda_pick_block")
            goal_pos = self.sim.get_actor_position_by_name("goal")
            block_to_goal = block_pos[:, 0:3] - goal_pos[:, 0:3]
            obj += torch.linalg.norm(block_to_goal, axis=1)[0]

        return obj

@hydra.main(version_base=None, config_path="../../conf", config_name="panda_pick")
def main(cfg: ExampleConfig):
    t = Tuning(cfg)
    t.tune()


if __name__ == "__main__":
    main()
