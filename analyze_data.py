import rlbench.utils
import rlbench.action_modes.action_mode
import rlbench.action_modes.arm_action_modes
import rlbench.action_modes.gripper_action_modes
import rlbench.observation_config
import rlbench.tasks.reach_target
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R


@hydra.main(version_base=None, config_path='./configs/visualize', config_name='base')
def main(cfg: DictConfig) -> None:
    env: rlbench.Environment = hydra.utils.instantiate(cfg.environment)
    task = env.get_task(hydra.utils.get_class(cfg.task))
    demos = task.get_demos(**cfg.demos)
    
    xyz_deltas = []
    axangle_deltas = []
    for demo in demos:
        prev_obs = None
        for obs in demo:
            if prev_obs is None:
                prev_obs = obs
                continue
            xyz = obs.gripper_pose[:3]
            axangle = R.from_quat(obs.gripper_pose[3:]).as_rotvec()
            prev_xyz = prev_obs.gripper_pose[:3]
            prev_axangle = R.from_quat(prev_obs.gripper_pose[3:]).as_rotvec()
            
            xyz_deltas.append(xyz - prev_xyz)
            axangle_deltas.append(axangle - prev_axangle)
            print(axangle - prev_axangle)
    print(np.mean(xyz_deltas, axis=0), np.std(xyz_deltas, axis=0))
    print(np.mean(axangle_deltas, axis=0), np.std(axangle_deltas, axis=0))

    env.shutdown()

if __name__ == '__main__':
    main()