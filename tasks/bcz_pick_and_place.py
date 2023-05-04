import random
from typing import List, Tuple
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.spawn_boundary import SpawnBoundary
import rlbench.backend.exceptions
from pyrep.objects.dummy import Dummy
import numpy as np

SOURCE_NAMES = ['lime', 'strawberry', 'eraser', 'sponge']
TARGET_NAMES = ['ceramic_plate', 'paper_bowl']

class BczPickAndPlace(Task):

    def init_task(self) -> None:
        self._shapes = [Shape(ob) for ob in SOURCE_NAMES] + [Shape(ob) for ob in TARGET_NAMES]
        self._pair_names = [(src, tgt) for tgt in TARGET_NAMES for src in SOURCE_NAMES]
        self._pairs = [(Shape(src), Shape(tgt)) for tgt in TARGET_NAMES for src in SOURCE_NAMES]
        self.task_waypoints = [[Dummy(f'{src}_pregrasp'), Dummy(f'{src}_grasp'), Dummy(f'{src}_pregrasp'), Dummy(f'{tgt}_release')] for tgt in TARGET_NAMES for src in SOURCE_NAMES]
        self.waypoints = [Dummy(f'waypoint{i}') for i in range(4)]
        self._success_sensors = [ProximitySensor(f'{tgt}_success') for tgt in TARGET_NAMES for src in SOURCE_NAMES]

        sources = [Shape(src) for src in SOURCE_NAMES]
        self.register_graspable_objects(sources)
        self.boundary = SpawnBoundary([Shape('plane')])

    def init_episode(self, index: int) -> List[str]:
        # move robot to initial position
        j = np.array([-19.98446757,  -5.13051495,  19.61395883, -93.61771465, 1.72630822, 85.01589052, 44.61888011])*np.pi/180
        self.robot.arm.set_joint_positions(j, disable_dynamics=True)
        
        n_distractor, i_pair = index // len(self._pair_names), index % len(self._pair_names)
        src, tgt = self._pairs[i_pair]
        src_name, tgt_name = self._pair_names[i_pair]
        src_name = src_name.replace('_', ' ')
        tgt_name = tgt_name.replace('_', ' ')
        success_sensor = self._success_sensors[i_pair]

        for i_sample in range(100):
            self.boundary.clear()
            try:
                for ob in self._shapes:
                    self.boundary.sample(ob, min_distance=0.15, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
            except rlbench.backend.exceptions.BoundaryError:
                continue
            break

        ob_not_in_scene = [ob for ob in self._shapes if ob != src and ob != tgt]
        ob_not_in_scene = random.sample(ob_not_in_scene, len(ob_not_in_scene) - n_distractor)
        for ob in ob_not_in_scene:
            pos = ob.get_position()
            pos[2] = 0
            ob.set_position(pos)

        for i_waypoint, waypoint in enumerate(self.task_waypoints[i_pair]):
            self.waypoints[i_waypoint].set_pose(waypoint.get_pose())

        self.register_success_conditions([
            DetectedCondition(src, success_sensor)
        ])
        # move robot to initial position
        # j = np.array([-19.98446757,  -5.13051495,  19.61395883, -93.61771465, 1.72630822, 85.01589052, 44.61888011])*np.pi/180
        # self.robot.arm.set_joint_positions(j, disable_dynamics=True)

        # spawn objects in the workspace
        
        return [f'place the {src_name} in the {tgt_name}']

    def variation_count(self) -> int:
        return len(SOURCE_NAMES) * len(TARGET_NAMES) * 5
    
    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return ((0, 0, 0), (0, 0, 0))
    
    def is_static_workspace(self) -> bool:
        return True