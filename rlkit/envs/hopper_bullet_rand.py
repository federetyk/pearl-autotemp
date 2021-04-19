import numpy as np

from pybullet_envs.gym_locomotion_envs import HopperBulletEnv


from . import register_env


LOG_SCALE_LIMIT = 3.0

@register_env('hopper-bullet-rand')
class HopperBulletRandEnv(HopperBulletEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        super(HopperBulletRandEnv, self).__init__()
        self.reset()
        self.initial_parameters = self.save_parameters()
        self.current_parameters = self.initial_parameters
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def sample_tasks(self, num_tasks):
        """
        Based on https://github.com/dennisl88/rand_param_envs
        """
        tasks = []
        for _ in range(num_tasks):
            body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
            body_lateral_frict_multiplier = np.array(1.5) ** np.random.uniform(-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
            body_params = {
                'mass': self.initial_parameters['body']['mass'] * body_mass_multiplyers,
                'lateralFriction': self.initial_parameters['body']['lateralFriction'] * body_lateral_frict_multiplier
            }

            legs_mass_multiplyers = np.array(1.5) ** np.random.uniform(-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
            legs_lateral_frict_multiplier = np.array(1.5) ** np.random.uniform(-LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)
            legs_params = {}
            for part_name, part in self.robot.parts.items():
                if part_name not in self.robot.foot_list: continue
                current_leg_mass = self.initial_parameters['legs'][part_name]['mass']
                current_leg_lateral_friction = self.initial_parameters['legs'][part_name]['lateralFriction']
                legs_params[part_name] = {
                    'mass': current_leg_mass * legs_mass_multiplyers,
                    'lateralFriction': current_leg_lateral_friction * legs_lateral_frict_multiplier
                }

            new_task = {
                'body': body_params,
                'legs': legs_params
            }
            tasks.append(new_task)
        return tasks

    def save_parameters(self):
        body_dyn_info = self._p.getDynamicsInfo(self.robot_body.bodies[0], -1)
        body_params = {
            'mass': body_dyn_info[0],
            'lateralFriction': body_dyn_info[1]
        }

        legs_params = {}
        for part_name, part in self.robot.parts.items():
            if part_name not in self.robot.foot_list: continue
            leg_dyn_info = self._p.getDynamicsInfo(self.robot_body.bodies[0], part.bodyPartIndex)
            legs_params[part_name] = {
                'mass': leg_dyn_info[0],
                'lateralFriction': leg_dyn_info[1]
            }

        initial_parameters = {
            'body': body_params,
            'legs': legs_params
        }

        return initial_parameters

    def set_task(self, task):
        # Update body dynamics
        body_mass = task['body']['mass']
        body_lateral_frict = task['body']['lateralFriction']
        self._p.changeDynamics(self.robot_body.bodies[0], -1, mass=body_mass, lateralFriction=body_lateral_frict)

        # Update legs dynamics
        for part_name, part in self.robot.parts.items():
            if part_name not in self.robot.foot_list: continue
            mass = task['legs'][part_name]['mass']
            lateral_frict = task['legs'][part_name]['lateralFriction']
            self._p.changeDynamics(self.robot_body.bodies[0], part.bodyPartIndex, mass=mass, lateralFriction=lateral_frict)

        self.current_parameters = task

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()
        # also calculates self.joints_at_limit

        self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        # state[0] is body height above ground, body_rpy[1] is pitch

        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):
            # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                #see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())
        # let's assume we have DC motor with controller, and reverse current braking

        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]

        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}
