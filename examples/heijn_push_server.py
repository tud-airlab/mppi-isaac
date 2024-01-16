from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
import yaml
from yaml import SafeLoader
import mppiisaac
import numpy as np
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from isaacgym import gymapi
import time
from examples.heijn_push_client import Objective
import sys
import os

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
        sim.viewer, None, gymapi.Vec3(1., 5.5, 3), gymapi.Vec3(1., 0, 0)        # CAMERA LOCATION, CAMERA POINT OF INTEREST
    )

def reset_trial(sim, init_pos, init_vel):
    sim.stop_sim()
    sim.start_sim()
    set_viewer(sim)
    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2]], device=sim.device))
        
@hydra.main(version_base=None, config_path="../conf", config_name="config_heijn_push")
def run_heijn_robot(cfg: ExampleConfig):
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
    
    # Experiment setup
    # ----------------------------------------------
    # Select initial position
    init_pos1 = [0.0, 2., 0.]       
    init_pos2 = [0.0, 0., 3.14/2]

    init_pos = init_pos1
    init_vel = [0., 0., 0.]

    # Select object 
    obj_index = 0       # 0 = Crate, 1 = Sphere
    #------------------------------------------------

                #  l      w      h      mu      m     x      y
    obj_set =  [[0.300, 0.500, 0.300, 0.300, 1.00, 2.00, 2.00],   # Crate
                [0.200, 0.200, 0.200, 0.300, 1.00, 2.00, 2.00]]   # Sphere
    obj_ = obj_set[obj_index][:]

    obst_1_dim = [0.6, 0.8, 0.108]
    obst_1_pos = [1, 1, obst_1_dim[-1]/2]

    obst_2_dim = [0.6, 0.8, 0.108]
    obst_2_pos = [-0.15, 1, obst_2_dim[-1]/2]
    
    if obj_index == 0:
        goal_pos_ghost = [0., 0.]
        object_dict = [{
            "type": "box",
            "name": "obj_to_push",
            "size": [obj_[0], obj_[1], obj_[2]],
            "init_pos": [obj_[5], obj_[6], obj_[2] / 2],
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 1.0],
            "friction": obj_[3],
            "noise_sigma_size": [0.005, 0.005, 0.0],
            "noise_percentage_friction": 0.3,
            "noise_percentage_mass": 0.3,
        }]
    else:
        goal_pos_ghost = [0.42, 1.]
        object_dict = [{
            "type": "sphere",
            "name": "obj_to_push",
            "size": [obj_[0]],
            "init_pos": [obj_[5], obj_[6], obj_[2] / 2],
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [4 / 255, 160 / 255, 218 / 255],
            "friction": obj_[3],
            "noise_sigma_size": [0.005],
            "noise_percentage_friction": 0.3,
            "noise_percentage_mass": 0.3,
        }]

    additions = [
        object_dict[0], 
        {
            "type": "box",
            "name": "obst_1",
            "size": obst_1_dim,
            "init_pos": obst_1_pos,
            "fixed": True,
            "color": [255 / 255, 120 / 255, 57 / 255],
            "handle": None,
        },
        {
            "type": "box",
            "name": "obst_2",
            "size": obst_2_dim,
            "init_pos": obst_2_pos,
            "fixed": True,
            "color": [255 / 255, 120 / 255, 57 / 255],
            "handle": None,
        },
        # Add goal, 
        {
            "type": "box",
            "name": "goal",
            "size": [obj_[0], obj_[1], obj_[2]],
            "init_pos": [goal_pos_ghost[0], goal_pos_ghost[1], -obj_[2]/2 + 0.005],
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

    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2]], device=sim.device))

    # Helpers
    count = 0
    client_helper = Objective(cfg, cfg.mppi.device)
    init_time = time.time()
    block_index = 1
    data_time = []
    n_trials = 0 
    timeout = 45
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
        # sim.gym.clear_lines(sim.viewer)
        
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
            block_ort = sim.root_state[:, block_index, 3:7]

            Ex, Ey, Etheta = client_helper.compute_metrics(block_pos, block_ort)
            print(Ex, Ey, Etheta)

            # Neglect sphere orientation error for tests
            if obj_index == 1:  
                Etheta_max = 10
                Ex_max = obj_[0]
                Ey_max = obj_[0]
            else: 
                Ex_max = 0.05
                Ey_max = 0.05
                Etheta_max = 0.17

            if Ex < Ex_max and Ey < Ey_max and Etheta < Etheta_max: 
                print("Success")
                final_time = time.time()
                time_taken = final_time - init_time
                print("Time to completion", time_taken)
                reset_trial(sim, init_pos, init_vel)
                
                init_time = time.time()
                count = 0
                data_rt.append(np.sum(rt_factor_seq) / len(rt_factor_seq))
                data_time.append(time_taken)
                n_trials += 1

            # print(f"FPS: {1/(time.time() - t)} RT-factor: {cfg.isaacgym.dt/(time.time() - t)}")
            
            count = 0
        else:
            count +=1

        if time.time() - init_time >= timeout:
            reset_trial(sim, init_pos, init_vel)
            init_time = time.time()
            count = 0
            n_trials += 1

        # Visualize samples
        # rollouts = bytes_to_torch(planner.get_rollouts())
        # sim.draw_lines(rollouts)
        
        rt_factor_seq.append(cfg.isaacgym.dt/(time.time() - t))

    if len(data_time) > 0: 
        print("Num. trials", n_trials)
        print("Success rate", len(data_time)/n_trials*100)
        print("Avg. Time", np.mean(np.array(data_time)*np.array(data_rt)))    
        print("Std. Time", np.std(np.array(data_time)*np.array(data_rt)))
    else:
        print("Seccess rate is 0")
    return {}

if __name__ == "__main__":
    res = run_heijn_robot()
