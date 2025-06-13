import warnings
from scipy.optimize._linesearch import LineSearchWarning   # private module is correct
from importlib.metadata import version

__version__ = version(__name__)
del version

warnings.filterwarnings('ignore', category=LineSearchWarning)  # mute noisy bayes HP opt
