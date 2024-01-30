import gym
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
import yaml
from yaml import SafeLoader
import mppiisaac
import os
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import hydra
from omegaconf import OmegaConf
import os
import torch
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from isaacgym import gymapi
import time
from examples.omnipanda_isaacgym_client import Objective
import sys

import io

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

def set_viewer(sim):
    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(-3., 2, 2), gymapi.Vec3(0.5, 0, 0.5)        # CAMERA LOCATION, CAMERA POINT OF INTEREST
    )

def reset_trial(sim, init_pos, init_vel):
    sim.stop_sim()
    sim.start_sim()
    set_viewer(sim)
    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2],
                                           init_pos[3], init_vel[3], init_pos[4], init_vel[4], init_pos[5], init_vel[5],
                                           init_pos[6], init_vel[6], init_pos[7], init_vel[7], init_pos[8], init_vel[8],
                                           init_pos[9], init_vel[9], init_pos[10], init_vel[10], init_pos[11], init_vel[11],
                                           init_pos[12], init_vel[12]], device=sim.device))
        
@hydra.main(version_base=None, config_path="../conf", config_name="config_dingo")
def run_omnipanda_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    actors=[]
    for actor_name in cfg.actors:
        with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml') as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))

    sim = IsaacGymWrapper(
        cfg.isaacgym,
        init_positions=cfg.initial_actor_positions,
        actors=actors,
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
    )

    # Manually add table + block and restart isaacgym
    obj_index = 0

                #  l     w     h    mu     m    x    y
    obj_set =  [[0.05, 0.05, 0.05, 0.90, 0.05, 0.37, 0.],    # Cube
                [0.03, 0.03, 0.04, 0.90, 0.2, 0.37, 0.]]    # Other
    
    obj_ = obj_set[obj_index][:]
    table_dim = [0.8, 1.0, 0.30]
    table_pos = [0.5, 0., 0.6]
    goal_pos = [-1., -1., 0.6]

    additions = [
        {
            "type": "box",
            "name": "obj_to_push",
            "size": [obj_[0], obj_[1], obj_[2]],
            "init_pos": [obj_[5], obj_[6], table_dim[-1] + obj_[2] / 2],
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 1.0],
            "friction": obj_[3],
            "noise_sigma_size": [0.001, 0.001, 0.0],
            "noise_percentage_friction": 0.3,
            "noise_percentage_mass": 0.3,
        },
        {
            "type": "box",
            "name": "table",
            "size": table_dim,
            "init_pos": table_pos,
            "color": [255 / 255, 120 / 255, 57 / 255],
            "fixed": True,
            "handle": None,
        },
        # Add goal, 
        {
            "type": "box",
            "name": "goal",
            "size": [obj_[0], obj_[1], obj_[2]],
            "init_pos": [goal_pos[0], goal_pos[1], goal_pos[2]],
            "fixed": True,
            "color": [119 / 255, 221 / 255, 119 / 255],
            "handle": None,
            "collision": False,
        }
    ]

    sim.add_to_envs(additions)

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")

    planner.add_to_env(additions)
    
    set_viewer(sim)
    
    # Select starting pose
    init_pos1 = [-1.0, -1.0, 0.0, 1.5, -0.94, 0., -2.0, 0., 1.8675, 0., 0.04, 0.03, 0.0]
    # init_pos2 = [-1.0, 1.0, 0.0, 0.0, -0.94, 0., -2.0, 0., 1.8675, 0., 0.04, 0.03]

    init_pos = init_pos1
    init_vel = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2],
                                           init_pos[3], init_vel[3], init_pos[4], init_vel[4], init_pos[5], init_vel[5],
                                           init_pos[6], init_vel[6], init_pos[7], init_vel[7], init_pos[8], init_vel[8],
                                           init_pos[9], init_vel[9], init_pos[10], init_vel[10], init_pos[11], init_vel[11],
                                           init_pos[12], init_vel[12],], device=sim.device))

    # Helpers
    count = 0
    client_helper = Objective(cfg, cfg.mppi.device)
    init_time = time.time()
    block_index = 1
    data_time = []
    data_err = []
    n_trials = 0 
    timeout = 60
    rt_factor_seq = []
    data_rt = []

    while n_trials < cfg.n_steps:
        t = time.time()
        # Reset state
        planner.reset_rollout_sim(
            torch_to_bytes(sim.dof_state[0]),
            torch_to_bytes(sim.root_state[0]),
            torch_to_bytes(sim.rigid_body_state[0]),
        )
        sim.gym.clear_lines(sim.viewer)
        
        # Compute action
        action = bytes_to_torch(planner.command())
        if torch.any(torch.isnan(action)):
            print("nan action")
            action = torch.zeros_like(action)

        # Apply action
        sim.set_dof_velocity_target_tensor(action)

        # Step simulator
        sim.step()

        # Monitoring
        # Evaluation metrics 
        # ------------------------------------------------------------------------
        if count > 10:
            block_pos = sim.root_state[:, block_index, :3]
            Ex, Ey, Ez = client_helper.compute_metrics(block_pos)
            metric = Ex+Ey
            # print("Metric Baxter", metric_1)
            print("Ex", Ex)
            print("Ey", Ey)
            print("Angle", Ez)
          
            if Ex < 0.02 and Ey < 0.02 and Ez < 0.02: 
                print("Success")
                final_time = time.time()
                time_taken = final_time - init_time
                print("Time to completion", time_taken)
                reset_trial(sim, init_pos, init_vel)                
                
                init_time = time.time()
                count = 0
                data_rt.append(np.sum(rt_factor_seq) / len(rt_factor_seq))
                data_time.append(time_taken)
                data_err.append(np.float64(metric))
                n_trials += 1

            rt_factor_seq.append(cfg.isaacgym.dt/(time.time() - t))
            print(f"FPS: {1/(time.time() - t)} RT-factor: {cfg.isaacgym.dt/(time.time() - t)}")
            
            count = 0
        else:
            count +=1

        if time.time() - init_time >= timeout:
            reset_trial(sim, init_pos, init_vel)
            init_time = time.time()
            count = 0
            # data_time.append(-1)
            # data_err.append(-1)
            # data_rt.append(-1)
            n_trials += 1

        # Visualize samples
        # rollouts = bytes_to_torch(planner.get_rollouts())
        # sim.draw_lines(rollouts)
    
    # Post processing

    # Print for data collection
    # original_stdout = sys.stdout # Save a reference to the original standard output

    # with open('databaselines.txt', 'a') as f:
    #     sys.stdout = f # Change the standard output to the file we created.
    #     print('Benchmark object {} in baseline {} for pose "{}"'.format(obj_index-2, baseline, baseline_pose))
    #     print('Time taken:', np.around(data_time, decimals=3))
    #     print('Placement error:', np.around(data_err, decimals=3))
    #     print('RT factor:', np.around(data_rt, decimals=3))
    #     print('\n')
    #     sys.stdout = original_stdout # Reset the standard output to its original value

    if len(data_time) > 0: 
        print("Num. trials", n_trials)
        print("Success rate", len(data_time)/n_trials*100)
        print("Avg. Time", np.mean(np.array(data_time)*np.array(data_rt)))    
        print("Std. Time", np.std(np.array(data_time)*np.array(data_rt)))
    else:
        print("Seccess rate is 0")
    return {}


if __name__ == "__main__":
    res = run_omnipanda_robot()