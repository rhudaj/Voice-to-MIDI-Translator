import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib import gridspec
from AudioUtil.DataStructures.AudioSignal import AudioSignal
import numpy as np

id = 0

class CustomFig:
  def __init__(self, x=np.ndarray|None, y=np.ndarray|None) -> None:
    """
      one column and multiple rows.
      subplots are added to the bottom.
    """
    self.fig = plt.figure(num=id)
    self.row = 0
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
      self.row = 1            # Start with one subplot
      ax = self.fig.add_subplot(self.row, 1, 1)
      ax.plot(x, y)

  def plot_bottom(self,
    x: np.ndarray,
    y: np.ndarray,
    label_x: str|None=None,
    label_y: str|None=None,
    x_min: int = None,
    x_max: int = None,
    y_min: int = None,
    y_max: int = None
  ) -> Axes:
    """
      Plots the data to a new subplot at the bottom.
    """
    self.row += 1
    gs = gridspec.GridSpec(self.row, 1)

    # Reposition existing subplots
    for i, ax in enumerate(self.fig.axes):
      ax.set_position(gs[i].get_position(self.fig))
      ax.set_subplotspec(gs[i])

    # Add new subplot
    new_ax = self.fig.add_subplot(gs[self.row-1])
    new_ax.plot(x, y)

    new_ax.set_xlim(left = x_min, right = x_max)
    new_ax.set_ylim(bottom=y_min, top=y_max)
    if(label_x): new_ax.set_xlabel(label_x)
    if(label_y): new_ax.set_ylabel(label_y)

    return new_ax

  def show(self) -> None:
      plt.show()

  def addPoint(self, x: float, y: float):
    plt.plot(x, y, 'ro')  # 'o' can be used to only draw a marker. 'r' = red

  def addLine(self, x: float):
    plt.axvline(x, color="red")
