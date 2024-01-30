######### visualizations #################################
from imported_libraries import *


def deviation_for_CI (size1, size2, p1, p2, alpha=0.05):
    ''' Returns the std det vor difference
        Will use this function in the plots'''
    assert 0 <= p1 and p1<=1, 'p1 must be a probability (TPR) '
    assert 0 <= p2 and p2<=1, 'p1 must be a probability (TPR) '
    z_critical = norm.ppf(1 - alpha / 2)
    s1=np.sqrt(p1*(1-p1)/size1) ##Since we have mean of Bernoulis
    s2=np.sqrt(size2*p2*(1-p2)/size2)
    sp=np.sqrt(   ((size1-1)*s1**2 + (size2-1)*s2**2)/  (size1+size2-2)  )
    dev= z_critical * np.sqrt(sp**2/size1 +sp**2/size2)
    return dev


def plot_values(x_values, y_values, plot_name, x_axis_name, y_axis_name):
    """
    Plot a line chart given lists of x and y values.

    Parameters:
    - x_values (list): List of x-axis values.
    - y_values (list): List of y-axis values.
    - plot_name (str): Name of the plot.
    - x_axis_name (str): Label for the x-axis.
    - y_axis_name (str): Label for the y-axis.

    Returns:
    - None (displays the plot).
    """
    # Plot the values
    plt.plot(x_values, y_values, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(plot_name)

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt

def plot_values_with_legend(x_values, y_values1, y_values2, plot_name, x_axis_name, y_axis_name):
    """
    Plot two lines with different colors and add a legend.

    Parameters:
    - x_values (list): List of x-axis values.
    - y_values1 (list): List of y-axis values for the first line.
    - y_values2 (list): List of y-axis values for the second line.
    - plot_name (str): Name of the plot.
    - x_axis_name (str): Label for the x-axis.
    - y_axis_name (str): Label for the y-axis.

    Returns:
    - None (displays the plot).
    """
    # Plot the first line with red color and label "before optimization"
    plt.plot(x_values, y_values1, color='red', linestyle='-', marker='o', label='before optimization')

    # Plot the second line with blue color and label "after optimization"
    plt.plot(x_values, y_values2, color='blue', linestyle='-', marker='o', label='after optimization')

    # Add labels, title, and legend
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(plot_name)
    plt.legend()

    # Show the plot
    plt.show()

