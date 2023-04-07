class CliffWalkingEnv:
    """ 悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,起点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]

        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for cur_y in range(self.nrow):
            for cur_x in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    # prob=1,next_state=state,reward = 0,done=True
                    if cur_y == self.nrow - 1 and cur_x > 0:
                        P[cur_y * self.ncol + cur_x][a] = [(1, cur_y * self.ncol + cur_x, 0,
                                                            True)]
                        continue
                    # 其他位置
                    # 被限制在边界内，如果受限，则不动
                    next_x = min(self.ncol - 1, max(0, cur_x + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, cur_y + change[a][1]))
                    next_state = next_y * self.ncol + next_x  # 序号
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True  # Episode 结束
                        if next_x != self.ncol - 1:  # 下一个位置不在终点，即在悬崖
                            reward = -100
                    P[cur_y * self.ncol +
                      cur_x][a] = [(1, next_state, reward, done)]
        return P
