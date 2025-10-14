import warnings
from scipy.optimize._linesearch import LineSearchWarning  # private module is correct
from .api import predict

warnings.filterwarnings('ignore', category=LineSearchWarning)  # mute noisy bayes HP opt

__all__ = ['predict']
