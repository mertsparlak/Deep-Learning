import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(5.0,4.00)
plt.rcParams["image.interpolation"]="nearest"
plt.rcParams["image.cmap"]="gray"

np.random.seed(1)

def zero_pad(X,pad):
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),"constant")
    return X_pad

np.random.seed(1)
x=np.random.randn(4,3,3,2)
x_pad=zero_pad(x,2)

print("x.shape=",x.shape)
print("x_pad.shape=",x_pad.shape)
print("x[1,1]=",x[1,1])
print("x_pad[1,1]=",x_pad[1,1])

fig,axarr=plt.subplots(1,2)
axarr[0].set_title("x")
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title("x_pad")
axarr[1].imshow(x_pad[0,:,:,0])

# a_slice_prev giriş matrisi,W ağırlık matrisi,b bayes değerleri
# çıkış yani z matrisi: giriş matrisi.W+bayes değerleri
def conv_single_step(a_slice_prev,W,b): 
    s=np.multiply(a_slice_prev,W)
    Z=np.sum(s)
    Z=float(b)+Z
    
    return Z
np.random.seed(1)
a_slice_prev=np.random.randn(4,4,3)
W=np.random.randn(4,4,3)
b=np.random.randn(1,1,1)
z=conv_single_step(a_slice_prev, W, b)
print("Z=",z)




