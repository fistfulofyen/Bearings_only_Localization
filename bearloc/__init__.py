"""bearloc -- bearings-only target localization (BEAR / DBEAR project).

Implements the project of Section 3.8.3 of Bar-Shalom, Li & Kirubarajan,
*Estimation with Applications to Tracking and Navigation* for the problem of
Section 3.7: estimating the initial position and velocity of a
constant-velocity target from noisy bearing measurements taken by a moving
platform.

Modules
-------
model       kinematics, bearing measurement, Jacobian, Fisher information
scenario    inputs A1-A7 and the BEAR.DEF defaults file
estimation  maximum likelihood estimation (Gauss-Newton/LM and BFGS)
stats       observability test, chi-square intervals, NEES
analysis    the B1-B8 pipeline and Monte Carlo runs; JSON results
report      text output of BEAR
ellipse     g-sigma ellipse of a 2x2 covariance matrix
plotting    DBEAR graphics and the scenario geometry plot
samples     ready-made sample scenarios
gui         Tkinter windows (packaged as BEAR.exe and DBEAR.exe)
"""

from .analysis import AnalysisResult, run_analysis
from .ellipse import sigma_ellipse
from .model import Maneuver
from .scenario import Scenario, load_defaults, save_defaults

__all__ = [
    "AnalysisResult",
    "Maneuver",
    "Scenario",
    "load_defaults",
    "run_analysis",
    "save_defaults",
    "sigma_ellipse",
]

__version__ = "1.0.0"
