import numpy as np
import maze_env

s = (4, 4)
env = np.tile(1, s)
env[0, 0] = 0
env[3, 3] = 0
# env[2, 2] = 0
env

# reward
r = -1
# action
p0 = np.tile(r, s)*env
p1 = p0.copy()
p2 = p0.copy()
p3 = p0.copy()
p4 = p0.copy()

value = np.zeros(s)
# value = env
for i in range(10):
    if i==0:
        print(i)
        print(value)
        p1 = p1[(0, 0, 1, 2), :]*env
        p2 = p2[(1, 2, 3, 3), :]*env
        p3 = p3[:, (0, 0, 1, 2)]*env
        p4 = p4[:, (1, 2, 3, 3)]*env
    else:
        p1 = value[(0, 0, 1, 2), :]*env
        p2 = value[(1, 2, 3, 3), :]*env
        p3 = value[:, (0, 0, 1, 2)]*env
        p4 = value[:, (1, 2, 3, 3)]*env
    value = p0+0.25*(p1+p2+p3+p4)
    print(i+1, value)










