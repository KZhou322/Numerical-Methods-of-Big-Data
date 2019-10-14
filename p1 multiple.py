import numpy as np

x=np.load("multiple.npy")
x=np.delete(x,np.s_[1:6],1)
y=np.load("multiple.npy")
y=np.delete(y,0,1)

m,c = np.linalg.lstsq(x,y,rcond=None)[0]
print(m,c)
print(np.linalg.lstsq(x,y,rcond=None)[1])