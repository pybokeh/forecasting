from numpy import log as ln
from numpy import random
from matplotlib.ticker import FuncFormatter
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_weibull_linreg(
    data: pd.DataFrame,
    col_failure_time: str,
    col_median_rank: str,
    title: str = "Weibull Probability Plot of Failure Times with Linear Regression Line",
    xlabel: str = r'$ln\left(ln\left(\frac{1}{1-F(x)}\right)\right)$',
    ylabel: str = "ln(x)",
):
    """
    Generates plot of failure times and the linear regression line.
    Also returns the R squared value and the Weibull parameters (shape, scale)

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing our failure times and median ranks
    col_failure_time : str
        column name containing failure times
    col_median_rank : str
        column name containing Bernard's median ranks for y-axis plotting positions
    title : str
        plot title
    xlabel : str
        x axis label
    ylabel : str
        y axis label

    Returns
    -------
    Plot of failure times along with linear regression best fit line
    """

    # Try to access the specified columns
    try:
        # First make sure we exclude rows where median_rank equals nan/None/null values
        df = data[~data[col_median_rank].isna()]
        y = ln(df[col_failure_time].values)
        median_rank = df[col_median_rank].values
        x = ln(-ln(1 - median_rank))
    except KeyError as e:
        # Handle the KeyError if the column is not found
        print(f"KeyError: {e} - The specified column does not exist in the DataFrame.")
        return None, None  # Return None if there's an error

    # Use Scipy's stats package to perform least-squares fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Since we plot failure times on the y-axis, the actual slope is inverted
    shape = 1 / slope

    # Since we're plotting failure times on the y-axis, we want the x-intercept, not the y-intercept
    # x-intercept is equal to the negative y-intercept divided by the slope/shape parameter
    # Basically, you are solving for x: 0 = mx + b, the equation of the line where y = 0
    x_intercept = -intercept / shape

    scale = math.exp(-x_intercept / slope)

    # Create the plot using the 'ax' object instead of plt object so that the plot doesn't immediately render
    fig, ax = plt.subplots(figsize=(8,6))
    ax.spines[['right', 'top']].set_visible(False)

    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, label=f"r\u00b2 value: {r_value**2:.5G}\nshape(k): {shape:.5G}\nscale(λ): {scale:.5G}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()

    return plt.show()


def plot_weibull_cdf_log_mrr(
    data: pd.DataFrame,
    col_failure_time: str,
    col_median_rank: str,
    shape: float,
    scale: float,
    title: str = "Weibull Probability Plot Log Scale",
    xlabel: str = "x (time)",
    ylabel: str = "Cumulative Distribution Function",
):
    """
    Generates Weibull cumulative distribution plot with log scale

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing our failure data
    col_failure_time : str
        column name containing our time to failure data
    col_median_rank : str
        column name containing our Bernard's median rank
    shape : float
        Weibull shape parameter
    scale : float
        Weibull scale parameter
    title : str
        title for plot
    xlabel : str
        x-axis label for plot
    ylabel : str
        y-axis label for plot

    Returns
    -------
    Plot of Weibull Median Rank Regression (MRR) CDF plot on log scale
    """

    # Try to access the specified columns
    try:
        # First make sure we exclude rows where median_rank equals nan/None/null values
        df = data[~data[col_median_rank].isna()]
        x = df[col_failure_time].values
        median_rank = df[col_median_rank].values
        y = ln(-ln(1 - median_rank))
    except KeyError as e:
        # Handle the KeyError if the column is not found
        print(f"KeyError: {e} - The specified column does not exist in the DataFrame.")
        return

    # Generate 100 plotting points following a Weibull distribution that
    # we think ideally fits our data using the shape and scale parameter
    x_ideal = scale * random.weibull(shape, size=100)
    x_ideal.sort()
    F = 1 - np.exp( -(x_ideal/scale)**shape )
    y_ideal = ln(-ln(1 - F))

    # Set up figure
    fig, ax = plt.subplots(figsize=(8,6))
    ax.spines[['right', 'top']].set_visible(False)

    # Create Weibull plot with log scale for both x and y-axis
    plt.semilogx(x, y, "bs")
    plt.semilogx(x_ideal, y_ideal, 'r-', label=f"shape(k)={shape:.5G}\nscale(λ)={scale:.5G}")
    plt.title(title, weight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')

    # Generate ticks
    def weibull_cdf(y, pos):
        return "%G %%" % (100*(1-np.exp(-np.exp(y))))

    # Format y-axis
    formatter = FuncFormatter(weibull_cdf)
    ax.yaxis.set_major_formatter(formatter)

    yt_f = np.array(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.8, 0.9, 0.95, 0.99
        ]
    )

    yt_lnf = ln(-ln(1-yt_f))
    plt.yticks(yt_lnf)
    ax.yaxis.grid()
    ax.xaxis.grid(which='both')
    plt.tight_layout()
    plt.show()


def plot_weibull_cdf_mrr(
    data: pd.DataFrame,
    col_failure_time: str,
    col_median_rank,
    shape: float,
    scale: float,
    title: str = "Weibull CDF",
    xlabel: str = "x (time)",
    ylabel: str = "Cumulative Distribution Function",
):
    """
    Generates Weibull CDF plot

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing our failure data
    col_failure_time : str
        column name containing our time to failure data
    col_median_rank : str
        column name containing our Bernard's median rank
    shape : float
        Weibull shape parameter
    scale : float
        Weibull scale parameter
    title : str
        title for plot
    xlabel : str
        x-axis label for plot
    ylabel : str
        y-axis label for plot

    Returns
    -------
    Plot of Weibull CDF
    """

    x_max = scale * 3
    x_cdf = np.arange(0,x_max)
    # Equation for 2-parameter Weibull CDF
    y_cdf = 1-np.exp(-(x_cdf/scale)**shape)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.spines[['right', 'top']].set_visible(False)

    plt.plot(x_cdf,y_cdf, label=f"shape(k)={shape:.5G}\nscale(λ)={scale:.5G}")

    # Try to access the specified columns
    try:
        # First make sure we exclude rows where median_rank equals nan/None/null values
        df = data[~data[col_median_rank].isna()]
        plt.scatter(x=df[col_failure_time], y=df[col_median_rank], c='red')
    except KeyError as e:
        # Handle the KeyError if the column is not found
        print(f"KeyError: {e} - The specified column does not exist in the DataFrame.")
        return

    plt.title("Weibull CDF",weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_weibull_cdf(
    shape: float,
    scale: float,
    title: str = "Weibull CDF",
    xlabel: str = "x (time)",
    ylabel: str = "Cumulative Distribution Function",
):
    """
    Generates Weibull CDF plot

    Parameters
    ----------
    shape : float
        Weibull shape parameter
    scale : float
        Weibull scale parameter
    title : str
        title for plot
    xlabel : str
        x-axis label for plot
    ylabel : str
        y-axis label for plot

    Returns
    -------
    Plot of Weibull CDF
    """

    x_max = scale * 3
    x_cdf = np.arange(0,x_max)
    # Equation for 2-parameter Weibull CDF
    y_cdf = 1-np.exp(-(x_cdf/scale)**shape)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.spines[['right', 'top']].set_visible(False)

    plt.plot(x_cdf,y_cdf, label=f"shape(k)={shape:.5G}\nscale(λ)={scale:.5G}")

    plt.title("Weibull CDF",weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()
