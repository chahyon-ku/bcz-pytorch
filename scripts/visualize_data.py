import rlbench.utils
import rlbench.action_modes.action_mode
import rlbench.action_modes.arm_action_modes
import rlbench.action_modes.gripper_action_modes
import rlbench.observation_config
import rlbench.tasks.reach_target
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='./configs/visualize', config_name='config')
def main(cfg: DictConfig) -> None:
    env: rlbench.Environment = hydra.utils.instantiate(cfg.environment)
    task = env.get_task(hydra.utils.get_class(cfg.task))
    demos = task.get_demos(**cfg.demos)
    
    for demo in demos:
        task.reset_to_demo(demo)
        for obs in demo:
            action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            task.step(action)

    env.shutdown()

if __name__ == '__main__':
    main()