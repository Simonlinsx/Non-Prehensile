"""Record RGB frames rendered by the running IsaacLab environment."""
import hashlib
import json
from pathlib import Path
import subprocess


class IsaacOnlineVideo:
    def __init__(self, base, path, fps, goal_ghost_opacity=.35):
        import cv2
        import numpy as np
        self.cv2 = cv2
        self.base, self.path, self.fps = base, Path(path), fps
        self.frames = []
        self.writer = None
        from dapl.contact_planner.isaac_visualization import _create_goal_object_ghost
        from pxr import Usd, UsdPhysics
        self.goal_ghost = _create_goal_object_ghost(base, goal_ghost_opacity)
        goal = base._c3_fixed_goal_pose_wxyz
        self.goal_ghost.set_world_poses(positions=goal[:, :3] + base.scene.env_origins,
                                        orientations=goal[:, 3:7], usd=True)
        ghost_path = '/Visuals/ContactPlannerM1/GoalObjectGhost'
        disabled_properties = []
        for prim in Usd.PrimRange(base.sim.stage.GetPrimAtPath(ghost_path)):
            for api, attribute in ((UsdPhysics.CollisionAPI, 'GetCollisionEnabledAttr'),
                                   (UsdPhysics.RigidBodyAPI, 'GetRigidBodyEnabledAttr')):
                if prim.HasAPI(api):
                    enabled = getattr(api(prim), attribute)().Get()
                    if enabled is not False:
                        raise RuntimeError(f'Goal visualization has active physics: {prim.GetPath()}')
                    disabled_properties.append(dict(path=str(prim.GetPath()), api=api.__name__, enabled=False))
        self.goal_visualization = dict(prim_path=ghost_path, opacity=goal_ghost_opacity,
                                       goal_pose_wxyz=goal[0].detach().cpu().tolist(),
                                       disabled_physics_properties=disabled_properties)
        # Initialize the actual Isaac render product, without advancing physics.
        base.render()
        for _ in range(30):
            base.sim.render()
            rgb = base.render(recompute=True)
            if rgb is not None and rgb.size and np.std(rgb) > .5:
                break
        else:
            raise RuntimeError('Isaac RGB renderer did not produce a nonempty frame')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.path.with_suffix('.raw.mp4')
        height, width = rgb.shape[:2]
        self.writer = cv2.VideoWriter(str(self.raw_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError('Could not open Isaac simulation video writer')

    def capture(self, step, simulation_time_s):
        if self.frames and self.frames[-1]['completed_physics_steps'] == step:
            return
        self.base.sim.render()
        rgb = self.base.render(recompute=True)
        frame = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self.cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (30, 30, 30), -1)
        self.cv2.putText(frame, 'IsaacLab physical simulation | FR3 closed stock gripper | native C3+ / OSC',
                         (16, 23), self.cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, self.cv2.LINE_AA)
        self.cv2.putText(frame, f'Simulation time: {simulation_time_s:.3f} s',
                         (16, 47), self.cv2.FONT_HERSHEY_SIMPLEX, .52, (220, 235, 245), 1, self.cv2.LINE_AA)
        self.cv2.putText(frame, 'Transparent cyan object: fixed goal pose',
                         (16, frame.shape[0]-18), self.cv2.FONT_HERSHEY_SIMPLEX, .52, (90, 65, 25), 1, self.cv2.LINE_AA)
        self.writer.write(frame)
        if not self.frames:
            self.cv2.imwrite(str(self.path.with_suffix('.initial.png')), frame)
        self.frames.append(dict(frame=len(self.frames), completed_physics_steps=step, simulation_time_s=simulation_time_s))

    def close(self):
        self.writer.release()
        frame_path = self.path.with_suffix('.frames.jsonl')
        frame_path.write_text(''.join(json.dumps(row)+'\n' for row in self.frames))
        # Keep the raw encoded video if optional H.264 conversion fails.
        converted = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(self.raw_path),
                                    '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                                    '-movflags', '+faststart', str(self.path)], capture_output=True, text=True)
        path = self.path if converted.returncode == 0 else self.raw_path
        if converted.returncode == 0:
            self.raw_path.unlink()
        return dict(path=str(path), frames=len(self.frames), fps=self.fps,
                    frame_timestamps_path=str(frame_path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source='RGB render product from the live IsaacLab physics execution; no pose replay',
                    goal_visualization=self.goal_visualization,
                    encoder_error=None if converted.returncode == 0 else converted.stderr[-2000:])
