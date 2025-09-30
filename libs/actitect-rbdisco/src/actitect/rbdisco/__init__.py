import warnings
from scipy.optimize._linesearch import LineSearchWarning   # private module is correct

warnings.filterwarnings('ignore', category=LineSearchWarning)  # mute noisy bayes HP opt
