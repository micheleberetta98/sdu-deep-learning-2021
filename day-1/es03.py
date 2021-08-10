import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])
b = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])

plt.plot(a, label='training accuracy')
plt.plot(b, label='validation accuracy')
plt.legend()
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
