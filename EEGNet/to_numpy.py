import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

file_dict = "../raw_data/"
file_path = []
for i in range(35):
    data = np.transpose(io.loadmat(file_dict + "S%d.mat" % (i + 1))["data"])
    data = data.transpose(0, 1, 3, 2)
    data = data[:,:,:,500:1000]
    np.save("./data/S%d.npy" % (i + 1), data)

print(data.shape)

plt.plot(data[0, 0, 0, :])
plt.show()
