import math
import torch

from controller import Supervisor
import numpy as np


from gym import spaces

def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):

    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax


def cal_dis(x,y):
    dis = np.linalg.norm([x,y])
    return dis




class EpuckSupervisor:
    def __init__(self, num_robots=10):
        super().__init__()
        self.num_robots = num_robots
        self.num_cols = 2
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.steps = 0
        self.steps_limit = 1000
        self.communication = self.initialize_comms()

        share_obs_dim=0

        self.observation_space=[]
        self.action_space = []
        for i in range(self.num_robots):
            self.observation_space.append(spaces.Box(low = -np.ones(7),high = np.ones(7),dtype=np.float64))
            self.action_space.append(spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))
            share_obs_dim+=7

        self.share_observation_space=[]
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.num_robots
        self.n_actions = self.action_space[0].shape[0]

        self.obs_history = [ ]
        self.dis_epu_history = [ ]
        self.selfless = [True for i in range(self.num_robots)]
        self.robot = [self.supervisor.getFromDef("e-puck" + str(i)) for i in range(self.num_robots)]
        self.col = [self.supervisor.getFromDef("col"+str(i)) for i in range(self.num_cols)]


        self.messageReceived = None
        self.episode_score = 0
        self.episode_score_list = []
        self.is_solved = False
        self.targetx = [1.00,-1.0,1.00,0.00,-1.00,1.00,-1.00,0.00,-0.5,0.50]
        self.targety = [-1.00,-1.0,0.00,-1.00,0.00,1.00,1.00,1.00,-1.00,1.00]
        self.evaluate_reward_history = []

    def is_done(self):
        if self.steps >= self.steps_limit:
            return np.array([True]*self.num_robots).reshape(self.num_robots)
        else:
            return np.array([False]*self.num_robots).reshape(self.num_robots)

    def initialize_comms(self):
        communication = []
        for i in range(self.num_robots):
            emitter = self.supervisor.getDevice(f'emitter{i}')
            receiver = self.supervisor.getDevice(f'receiver{i}')

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            communication.append({
                'emitter': emitter,
                'receiver': receiver,
            })


        return communication


    def step(self, action):

        self.handle_emitter(np.clip(np.array(action).reshape(10,2),-1.0,1.0))
        if self.supervisor.step(self.timestep) == -1:
            exit()
        self.steps +=1

        return (
            self.get_observations(),
            self.get_state(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
            None
        )



    def handle_emitter(self, actions):
        for i, action in enumerate(actions):
            message = (",".join(map(str, action))).encode("utf-8")
            self.communication[i]['emitter'].send(message)

    def handle_receiver(self):
        messages = []
        for com in self.communication:
            receiver = com['receiver']
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getData().decode("utf8"))
                receiver.nextPacket()
            else:
                messages.append(None)
        # print(messages)
        return messages

    def get_observations(self):
        self.positions_x = np.array([normalizeToRange(self.robot[i].getPosition()[0], -1.97, 1.97, -2.0, 2.0)
                                     for i in range(self.num_robots)])
        # 限制位置到-0.97
        self.positions_y = np.array([normalizeToRange(self.robot[i].getPosition()[1], -1.97, 1.97, -2.0, 2.0)
                                     for i in range(self.num_robots)])
        self.rot = np.array([self.robot[i].getField("rotation").getSFRotation()[3] % (2 * np.pi)
                                              if self.robot[i].getField("rotation").getSFRotation()[2]
                                                 > 0 else (-self.robot[i].getField("rotation").getSFRotation()[
                                                  3]) % (2 * np.pi) for i in range(self.num_robots)])


        self.col_x = np.array(
            [normalizeToRange(self.col[i].getField("translation").getSFVec3f()[0], -1.97, 1.97, -2.0, 2.0)
             for i in range(self.num_cols)])
        self.col_y = np.array(
            [normalizeToRange(self.col[i].getField("translation").getSFVec3f()[1], -1.97, 1.97, -2.0, 2.0)
             for i in range(self.num_cols)])
        self.messageReceived = self.handle_receiver()

        self.observations = np.empty((self.num_robots, self.observation_space[0].shape[0]), float)
        self.dis_goal = [cal_dis(self.positions_x[i] - self.targetx[i], self.positions_y[i] - self.targety[i]) for i
                         in range(self.num_robots)]
        self.rot_goal = [math.atan2(self.targety[i] - self.positions_y[i], self.targetx[i] - self.positions_x[i])
                         for i in range(self.num_robots)]
        self.dis_epu = np.empty((self.num_robots, self.num_robots), float)
        self.rot_epu = np.empty((self.num_robots, self.num_robots), float)
        self.dis_col = np.empty((self.num_robots,self.num_cols),float)
        for i in range(self.num_robots):
            a = self.positions_x - self.positions_x[i]
            b = self.positions_y - self.positions_y[i]
            c = self.positions_x[i] - self.col_x
            d = self.positions_y[i] - self.col_y
            for k in range(self.num_robots):
                self.dis_epu[i][k] = cal_dis(a[k], b[k])
                self.rot_epu[i][k] = math.atan2(b[k], a[k])
            for j in range(self.num_cols):
                self.dis_col[i][j] = cal_dis(c[j],d[j])
            dis_col_temp = np.copy(self.dis_col[i])
            index_col = np.argwhere(dis_col_temp  == dis_col_temp .min())[0][0]
            dis_epu_temp = np.copy(self.dis_epu[i])
            dis_epu_temp[np.where(self.dis_epu[i] == 0)] = 999
            index = np.argwhere(dis_epu_temp == dis_epu_temp.min())[0][0]

            delta_x = self.positions_x[i] - self.targetx[i]
            delta_y = self.positions_y[i] - self.targety[i]
            if self.dis_col[i].min() > 0.4:
                c_temp = 0.4 #c[index_col]
                d_temp = 0.4 #d[index_col]
            else:
                c_temp = c[index_col]
                d_temp = d[index_col]
            if dis_epu_temp.min() > 0.4:
                a_temp = 0.4
                b_temp = 0.4
                self.selfless[i] = False
            else:
                a_temp_list = [ ]
                b_temp_list = [ ]
                t_temp = []
                for t in range(self.num_robots):
                    if 0.0<self.dis_epu[i][t] <=0.4:
                        a_temp_list.append(a[t])
                        b_temp_list.append(b[t])
                        t_temp.append(t)
                    else:
                        a_temp_list = a_temp_list
                        b_temp_list = b_temp_list
                a_temp = a[index]
                b_temp = b[index]



            self.observations[i] = np.hstack([a_temp, b_temp, delta_x, delta_y,c_temp,d_temp,
                                              self.robot[i].getField("rotation").getSFRotation()[3] % (2 * np.pi)
                                              if self.robot[i].getField("rotation").getSFRotation()[2]
                                                 > 0 else (-self.robot[i].getField("rotation").getSFRotation()[
                                                  3]) % (2 * np.pi)
                                              ])

        return self.observations.reshape(self.num_robots,7)

    def get_state(self):
        state=[]
        for i in range(self.num_robots):
            state.append(self.get_observations())

        self.obs_history.append(self.observations)
        del self.obs_history[:-2]
        self.dis_epu_history.append(self.dis_epu)
        del self.dis_epu_history[:-2]
        out=np.array(state).reshape(10,70)

        return out

    def get_reward(self, action=None):

        rewards = np.empty((self.num_robots, 1), float)
        for i in range(self.num_robots):
            rewards[i] = - np.linalg.norm([self.positions_x[i]-self.targetx[i],self.positions_y[i]-self.targety[i]])
            if self.dis_goal[i]<0.03:
                rewards[i] += 0.05


        return rewards.reshape(self.num_robots,1)

    def get_info(self):
        pass


    def reset(self):
        self.steps = 0
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))

        for i in range(self.num_robots):
            self.communication[i]['receiver'].disable()
            self.communication[i]['receiver'].enable(self.timestep)
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        return self.get_observations(),self.get_state(),None






if __name__ =='__main__':
    env_name= 'nav'
    env=EpuckSupervisor()
    env_evaluate = env
    number = 5
    seed = 41
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    max_action = float(env.action_space[0].high[0])
    max_episode_steps = env.steps_limit
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))


    print('ttteeesst')

    print("observation_space=", env.observation_space)

    print("action_space=", env.action_space)



    s = env.reset()
    print(env.share_observation_space)


    a = np.empty((env.num_robots, 2), float)

    print(a.shape)
    for n in range(env.num_robots):
        a[n] = env.action_space[0].sample()

    s_,state, r, done, _,__ = env.step(a)
    print(r)
    print(state.shape)
    print(r.shape)
    print(done.shape)


