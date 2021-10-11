'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import BAO_scale_fitting
import numpy as np
from matplotlib import pyplot as plt


alphas = np.linspace(0.9, 1.1, 30)
#log_likelihood = coding_alpha2.run(1)

predicted = np.zeros(len(alphas))

for i in range(len(alphas)):
	predicted[i] = BAO_scale_fitting.run(alphas[i]) + 1
	print('predicted alpha: ', predicted[i])
	print()


difference = predicted - alphas

difference_error = difference / alphas

#np.savetxt('output_delta_alpha.txt', predicted)
#np.savetxt('Marginal_Alphas_3.5.txt', predicted)

plt.figure()
plt.plot(alphas, predicted, label=r'Predicted $\alpha$')
plt.plot(alphas, alphas, label=r'True $\alpha$')
plt.xlabel(r'True $\alpha$')
plt.ylabel(r'Predicted $\alpha$')
plt.title(r'Constant $b_1$ Predicted $\alpha$')
plt.legend(fontsize=15)
plt.ylim(0.85, 1.2)
plt.grid()
plt.show()

'''
plt.figure()
plt.plot(alphas, difference, label='Difference')
plt.xlabel('Inputted Alpha')
plt.ylabel('Difference')
plt.legend()
plt.show()

plt.figure()
plt.plot(alphas, difference_error, label='Difference Error')
plt.xlabel('Inputted Alpha')
plt.ylabel('Difference Error')
plt.legend()
plt.show()
'''