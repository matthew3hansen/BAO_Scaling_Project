import helper
import coding_alpha3
import numpy as np
from matplotlib import pyplot as plt


alphas = np.linspace(0.5, 1.5, 25)
predicted_alphas_positive = np.zeros(len(alphas))
predicted_alphas_negative = np.zeros(len(alphas))

coding_alpha3.run(0.99)