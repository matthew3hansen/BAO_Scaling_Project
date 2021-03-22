import helper
import Coding_Alpha
import numpy as np
from matplotlib import pyplot as plt

list_of_d_alpha_positive = np.zeros(30)
list_of_d_alpha_negative = np.zeros(30)
alphas = [0.998, 1., 1.001]
predicted_alphas_positive = np.zeros(len(alphas))
predicted_alphas_negative = np.zeros(len(alphas))

for i in range(len(alphas)):
	print(i)
	for r in range(0, 28):
		value = Coding_Alpha.run(alphas[i], r, r+1)
		list_of_d_alpha_positive[r] = value[0]
		list_of_d_alpha_negative[r] = value[1]
	
	predicted_alphas_positive[i] = 1 - np.mean(list_of_d_alpha_positive)
	predicted_alphas_negative[i] = 1 - np.mean(list_of_d_alpha_negative)

plt.figure()
plt.plot(alphas, predicted_alphas_positive, label='+ solution')
plt.plot(alphas, predicted_alphas_negative, label='- solution')
plt.plot(alphas, alphas, label='Linear')
plt.xlabel('Alpha Value')
plt.ylabel('Predicted Alpha Value')
plt.legend()
plt.show()
