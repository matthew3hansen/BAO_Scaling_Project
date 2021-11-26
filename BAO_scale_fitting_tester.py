'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import BAO_scale_fitting
import numpy as np
from matplotlib import pyplot as plt


alphas = np.linspace(0.9, 1.1, 100)

predicted = np.zeros(len(alphas))

running = True

for i in range(len(alphas)):
    if running:
        predicted[i] = BAO_scale_fitting.marginal_linear_bias(alphas[i]) + 1
    else:
        predicted[i] = BAO_scale_fitting.fixed_linear_bias(alphas[i]) + 1
        
    print('predicted alpha: ', predicted[i])
    print()


difference = predicted - alphas

difference_error = difference / alphas

#np.savetxt('output_delta_alpha.txt', predicted)
#np.savetxt('Marginalized_Predicted_100_iterations.txt', predicted)

plt.figure()
plt.plot(alphas, predicted, label=r'Predicted $\alpha$')
plt.plot(alphas, alphas, label=r'True $\alpha$', linestyle='dashed')
plt.xlabel(r'True $\alpha$')
plt.ylabel(r'Predicted $\alpha$')

if running:
    plt.title(r'Marginalized $\mathcal{B}$')
else:
    plt.title(r'Fixed $\mathcal{B}$')
    
plt.legend(fontsize=15)
plt.ylim(0.89, 1.1)
plt.xlim(0.9, 1.1)
plt.grid()
plt.savefig('Marginal_B_Plot_11-24-21.pdf')
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