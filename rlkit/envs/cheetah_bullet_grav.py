import numpy as np

from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv


from . import register_env


@register_env('cheetah-bullet-grav')
class CheetahBulletGravEnv(HalfCheetahBulletEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        super(CheetahBulletGravEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._grav = self._task['gravity']
        # self._task = self.tasks[idx]
        self._goal = self._grav # assume parameterization of task by single vector
        self.reset()

    def sample_tasks(self, num_tasks):
        forces = np.random.uniform(5.8, 13.8, size=(num_tasks,))
        tasks = [{'gravity': g} for g in forces]
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        # directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        # tasks = [{'goal': d} for d in directions]
        return tasks

    def step(self, a):
        # Corresponding task within the task distribution
        self._p.setGravity(0, 0, -self._grav)

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
