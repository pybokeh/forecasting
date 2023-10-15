from numpy import log as ln
import math
import numpy as np
import pandas as pd
import scipy.stats as stats


def add_rank(data: pd.DataFrame, col_to_sort: str):
    """
    Adds a rank column based on the column to sort on. rank will equal: 1, 2, 3, ... , length of dataframe

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe
    col_to_sort: str
        column to sort for which the rank will be based on
    """

    df = data.sort_values(by=col_to_sort)
    df = df.assign(rank=range(1, len(df)+1))
    
    return df


def add_reverse_rank(data: pd.DataFrame, col_to_sort: str):
    """
    Adds a reverse rank column based on the column to sort on.

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe
    col_to_sort : str
        column to sort for which the reverse rank will be based on
    """
    
    df = data.sort_values(by=col_to_sort)
    df = df.assign(reverse_rank=range(len(df), 0, -1))

    return df


def add_adjusted_rank(data: pd.DataFrame, col_status: str, col_rev_rank: str):
    """
    Adds adjusted rank column

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing failure data
    col_status: str
        column containing the status of the unit: failed or suspended
    col_rev_rank : str
        column containing the reverse rank
    """

    prev_adj_rank = [0]
    
    def adj_rank(series):
        if series[col_status] == "SUSPENDED":
            return "SUSPENSION"
        else:
            adjusted_rank = (series[col_rev_rank] * 1.0 * prev_adj_rank[-1] + (len(data) + 1))/(series[col_rev_rank] + 1)
            prev_adj_rank[0] = adjusted_rank
            return adjusted_rank

    df = data.assign(adjusted_rank=data.apply(adj_rank, axis=1))

    return df


def add_median_rank(data: pd.DataFrame, col_adj_rank: str):
    """
    Adds new column containing Bernard's adjusted median rank which accounts for suspended units

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe
    col_adj_rank : str
        column containing the adjusted rank
    """

    def median_ranks(series):
        if series[col_adj_rank] == "SUSPENSION":
            return None
        else:
            median_rank = (series[col_adj_rank] - 0.3)/(len(data) + 0.4)
            return median_rank

    df = data.assign(median_rank_rank=data.apply(median_ranks, axis=1))

    return df


def add_median_ranks(data: pd.DataFrame, col_failure_time: str, col_status: str):
    """
    Calculates Bernard's median ranks when accounting for suspended units.  This function will add
    rank, revere rank, adjusted rank, and median rank columns.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame
    col_failure_time: str
        Column containing failure times: days to failures, miles to failure, etc
    col_status: str
        Column containing status of each unit: Valid values are FAILED or SUSPENDED
    """

    prev_adj_rank = [0]

    def col_status_check(status: str) -> bool:
        if all(data[status].isin(['failed','suspended','FAILED', 'SUSPENDED'])):
            return True
        else:
            return False

    def adj_ranks(series):
        if series[col_status] in("suspended", "SUSPENDED"):
            return None
        else:
            adjusted_rank = (series['reverse_rank'] * 1.0 * prev_adj_rank[-1] + (len(data) + 1)) / (series['reverse_rank'] + 1)
            prev_adj_rank[0] = adjusted_rank
            return adjusted_rank

    def median_ranks(series):
        if series['adjusted_rank'] in("suspended", "SUSPENDED"):
            return None
        else:
            median_rank = (series['adjusted_rank'] - 0.3) / (len(data) + 0.4)
            return median_rank

    try:
        if col_status_check(col_status):
            df = data.sort_values(by=col_failure_time)
            df = df.assign(rank=range(1, len(df)+1))
            df = df.assign(reverse_rank=range(len(df), 0, -1))
            df = df.assign(adjusted_rank=df.apply(adj_ranks, axis=1))
            df = df.assign(median_rank=df.apply(median_ranks, axis=1))

            return df
        else:
            print('Error: Failure status column must ONLY contain "failed", "FAILED", "suspended", or "SUSPENDED"')
    except KeyError as e:
        # Handle the KeyError if the column is not found
        print(f"KeyError: {e} - The specified column does not exist in the DataFrame.")
        return


def weibull_params_linreg(
    data: pd.DataFrame,
    col_failure_time: str,
    col_median_rank: str,
):
    """
    Generates plot of failure times and the linear regression line.
    Also returns the R squared value and the Weibull parameters (shape, scale)

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing our failure data
    col_failure_time : str
        column name containing failure times
    col_median_rank : str
        column name containing Bernard's median ranks for y-axis plotting positions
    """

    # Try to access the specified columnAs
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

    params_tuple = (r_value**2, shape, scale)

    return params_tuple


def weibull_cdf_table(shape: float, scale: float) -> pd.DataFrame:
    """
    Creates dataframe containing our failures times and Weibull cumulative distribution function (CDF)

    Parameters
    ----------
    shape : float
        Weibull shape parameter
    scale : float
        Weibull scale paramter
    """

    x_max = scale * 3
    x_cdf = np.arange(0,x_max)
    # Equation for 2-parameter Weibull CDF
    y_cdf = 1-np.exp(-(x_cdf/scale)**shape)

    # Create pandas dataframe containing our failure times x and CDF
    df = pd.DataFrame({'x': x_cdf, 'cdf': y_cdf})

    return df
