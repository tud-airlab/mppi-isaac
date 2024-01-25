from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from isaacgym import gymapi
import hydra
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.transport import torch_to_bytes, bytes_to_torch
import time
from pynput import mouse, keyboard

MODE = "auto"  # "step"
CONTINUE = True


def on_press(key):
    global MODE, CONTINUE
    if key.char == 'c':
        MODE = "auto"
    elif key.char == 's':
        MODE = "step"
        CONTINUE = True



@hydra.main(version_base=None, config_path="../../conf", config_name="config_anymal")
def run_heijn_robot(cfg: ExampleConfig):
    global MODE, CONTINUE
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cfg.isaacgym.dt = 0.1
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

    sim._gym.viewer_camera_look_at(
        sim.viewer,
        None,
        gymapi.Vec3(1.0, 6.5, 4),
        gymapi.Vec3(1.0, 0, 0),  # CAMERA LOCATION, CAMERA POINT OF INTEREST
    )

    t = time.time()
    for _ in range(cfg.n_steps):
        # Compute action
        action = bytes_to_torch(
            planner.compute_action_tensor(
                torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state)
            )
        )

        # Apply action
        sim.set_dof_velocity_target_tensor(action)

        # Step simulator
        sim.step()

        # Visualize samples
        # rollouts = bytes_to_torch(planner.get_rollouts())
        # sim._gym.clear_lines(sim.viewer)
        # sim.draw_lines(rollouts)

        # Timekeeping
        if MODE == "auto":
            actual_dt = time.time() - t
            rt = cfg.isaacgym.dt / actual_dt
            if rt > 1.0:
                time.sleep(cfg.isaacgym.dt - actual_dt)
                actual_dt = time.time() - t
                rt = cfg.isaacgym.dt / actual_dt
            print(f"FPS: {1/actual_dt}, RT={rt}")
            t = time.time()
        elif MODE == "step":
            while not CONTINUE:
                time.sleep(0.05)
            CONTINUE = False


if __name__ == "__main__":
    res = run_heijn_robot()
