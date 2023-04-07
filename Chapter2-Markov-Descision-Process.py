import numpy as np


class MDP_example:
    def __init__(self):
        self.gamma = 0.5
        self.S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
        self.A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
        # 状态转移函数
        self.P = {
            "s1-保持s1-s1": 1.0,
            "s1-前往s2-s2": 1.0,
            "s2-前往s1-s1": 1.0,
            "s2-前往s3-s3": 1.0,
            "s3-前往s4-s4": 1.0,
            "s3-前往s5-s5": 1.0,
            "s4-前往s5-s5": 1.0,
            "s4-概率前往-s2": 0.2,
            "s4-概率前往-s3": 0.4,
            "s4-概率前往-s4": 0.4,
        }
        # 奖励函数
        self.R = {
            "s1-保持s1": -1,
            "s1-前往s2": 0,
            "s2-前往s1": -1,
            "s2-前往s3": -2,
            "s3-前往s4": -2,
            "s3-前往s5": 0,
            "s4-前往s5": 10,
            "s4-概率前往": 1,
        }
        # 策略1,随机策略
        self.Pi_1 = {
            "s1-保持s1": 0.5,
            "s1-前往s2": 0.5,
            "s2-前往s1": 0.5,
            "s2-前往s3": 0.5,
            "s3-前往s4": 0.5,
            "s3-前往s5": 0.5,
            "s4-前往s5": 0.5,
            "s4-概率前往": 0.5,
        }
        # 策略2
        self.Pi_2 = {
            "s1-保持s1": 0.6,
            "s1-前往s2": 0.4,
            "s2-前往s1": 0.3,
            "s2-前往s3": 0.7,
            "s3-前往s4": 0.5,
            "s3-前往s5": 0.5,
            "s4-前往s5": 0.1,
            "s4-概率前往": 0.9,
        }
        self.MDP = (self.S, self.A, self.P, self.R, self.gamma)

    # 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
    def join(self, str1, str2):
        return str1 + '-' + str2

    def compute_value(self, P, rewards):
        ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
        rewards = np.array(rewards).reshape((-1, 1))  # 将rewards写成列向量形式
        value = np.dot(np.linalg.inv(np.eye(np.size(self.S), np.size(self.S)) - self.gamma * P), rewards)
        return value

    def sample(self, Pi, timestep_max, sample_num):
        ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
        episodes = []
        for _ in range(sample_num):
            episode = []
            timestep = 0
            s = self.S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
            # 当前状态为终止状态或者时间步太长时,一次采样结束
            while s != "s5" and timestep <= timestep_max:
                timestep += 1
                rand, temp = np.random.rand(), 0
                # 在状态s下根据策略选择动作
                for a_opt in self.A:
                    temp += Pi.get(self.join(s, a_opt), 0)
                    if temp > rand:
                        a = a_opt
                        r = self.R.get(self.join(s, a), 0)
                        break
                rand, temp = np.random.rand(), 0
                # 根据状态转移概率得到下一个状态s_next
                for s_opt in self.S:
                    temp += self.P.get(self.join(self.join(s, a), s_opt), 0)
                    if temp > rand:
                        s_next = s_opt
                        break
                episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
                s = s_next  # s_next变成当前状态,开始接下来的循环
            episodes.append(episode)
        return episodes

    # 对所有采样序列计算所有状态的价值
    def MC(self, episodes, V, N):
        for episode in episodes:
            G = 0
            for i in range(len(episode) - 1, -1, -1):  # 一个序列从后往前计算
                (s, a, r, s_next) = episode[i]
                G = r + self.gamma * G
                N[s] = N[s] + 1
                V[s] = V[s] + (G - V[s]) / N[s]

    def occupancy(self, episodes, s, a, timestep_max):
        ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
        rho = 0
        total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
        occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
        for episode in episodes:
            for i in range(len(episode)):
                (s_opt, a_opt, r, s_next) = episode[i]
                total_times[i] += 1
                if s == s_opt and a == a_opt:
                    occur_times[i] += 1
        for i in reversed(range(timestep_max)):
            if total_times[i]:
                rho += self.gamma ** i * occur_times[i] / total_times[i]
        return (1 - self.gamma) * rho


if __name__ == "__main__":
    np.random.seed(0)
    mdp = MDP_example()

    '''1.使用状态价值函数解析解计算状态价值'''
    # 转化后的MRP的状态转移矩阵
    P_from_mdp_to_mrp = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.1, 0.2, 0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

    V = mdp.compute_value(P_from_mdp_to_mrp, R_from_mdp_to_mrp)
    print("MDP中每个状态价值分别为\n", V)

    '''2.使用蒙特卡罗方法计算状态价值'''
    timestep_max = 20
    sample_num = 1000
    # 采样1000次,可以自行修改
    episodes = mdp.sample(mdp.Pi_1, timestep_max, sample_num)
    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    mdp.MC(episodes, V, N)
    print("使用蒙特卡洛方法计算MDP的状态计数器为\n", N)
    print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)

    '''3.计算策略的占用度量'''
    timestep_max = 1000
    sample_num = 1000
    episodes_1 = mdp.sample(mdp.Pi_1, timestep_max, sample_num)
    episodes_2 = mdp.sample(mdp.Pi_2, timestep_max, sample_num)
    rho_1 = mdp.occupancy(episodes_1, "s4", "概率前往", timestep_max)
    rho_2 = mdp.occupancy(episodes_2, "s4", "概率前往", timestep_max)
    print(rho_1, rho_2)  # 可见不同策略对于同一个状态动作对的占用度量是不一样
