import numpy as np
import matplotlib.pyplot as plt

x=np.load("single.npy")
x=np.delete(x,1,1)
y=np.load("single.npy")
y=np.delete(y,0,1)
asdf = np.ones(len(x))[np.newaxis]
asdf=np.transpose(asdf)
x1=np.hstack([x,asdf])
m,c = np.linalg.lstsq(x1,y,rcond=None)[0]
print(m,c)
print(np.linalg.lstsq(x1,y,rcond=None)[1])
#
# _ = plt.plot(x, y, 'o', label='Original data', markersize=10)
# _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
# _ = plt.legend()
# plt.show()