import numpy as np
import matplotlib.pyplot as plt


a=[[0.003     , 0.003     ],
 [0.003     , 0.103     ],
 [0.003     , 0.20300001]]

b=[[-0.015     ,  0.003     ],
 [-0.015     ,  0.271     ],
 [-0.015     ,  0.53900003]]

a=[[0.003, 0.003],
 [0.003, 0.043],
 [0.003, 0.083]]
b=[[-0.015     ,  0.003     ],
 [-0.015     ,  0.071     ],
 [-0.015     ,  0.13900001]]

xs = [x[0] for x in a]
ys = [x[1] for x in a]
plt.scatter(xs, ys, s=5)


xs1 = [x[0] for x in b]
ys1 = [x[1] for x in b]
plt.scatter(xs1, ys1, s=5)

plt.show()