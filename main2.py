import helper
import coding_alpha2
import numpy as np
from matplotlib import pyplot as plt


alphas = np.linspace(0.5, 1.5, 25)
predicted_alphas_positive = np.zeros(len(alphas))
predicted_alphas_negative = np.zeros(len(alphas))

for i in range(len(alphas)):
	print(i)
	value = coding_alpha2.run(alphas[i])	
	predicted_alphas_positive[i] = 1 - value[0]
	predicted_alphas_negative[i] = 1 - value[1]
	print('alpha = ', alphas[i], ' predicted = ', predicted_alphas_negative[i])

plt.figure()
plt.plot(alphas, predicted_alphas_positive, label='+ solution')
plt.plot(alphas, predicted_alphas_negative, label='- solution')
plt.plot(alphas, alphas, label='Linear')
plt.xlabel('Alpha Value')
plt.ylabel('Predicted Alpha Value')
plt.legend()
plt.show()