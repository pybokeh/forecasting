import pandas as pd


def add_rank(df: pd.DataFrame, col_to_sort: str):
    """
    Adds a rank column based on the column to sort on. rank will equal: 1, 2, 3, ... , length of dataframe

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe
    col_to_sort: str
        column to sort for which the rank will be based on
    """

    df = df.sort_values(by=col_to_sort)
    df = df.assign(rank=range(1, len(df)+1))
    
    return df


def add_reverse_rank(df: pd.DataFrame, col_to_sort: str):
    """
    Adds a reverse rank column based on the column to sort on.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe
    col_to_sort : str
        column to sort for which the reverse rank will be based on
    """
    
    df = df.sort_values(by=col_to_sort)
    df = df.assign(reverse_rank=range(len(df), 0, -1))

    return df


def add_adjusted_rank(df: pd.DataFrame, col_status: str, col_rev_rank: str):
    """
    Adds adjusted rank column

    Parameters
    ----------
    df : pd.DataFrame
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
            adjusted_rank = (series[col_rev_rank] * 1.0 * prev_adj_rank[-1] + (len(df) + 1))/(series[col_rev_rank] + 1)
            prev_adj_rank.append(adjusted_rank)
            return adjusted_rank

    df = df.assign(adjusted_rank=df.apply(adj_rank, axis=1))

    return df


def add_median_rank(df: pd.DataFrame, col_adj_rank: str):
    """
    Adds new column containing Bernard's adjusted median rank which accounts for suspended units

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe
    col_adj_rank : str
        column containing the adjusted rank
    """

    def median_rank(series):
        if series[col_adj_rank] == "SUSPENSION":
            return None
        else:
            median_rank = (series[col_adj_rank] - 0.3)/(len(df) + 0.4)
            return median_rank

    df = df.assign(median_rank_rank=df.apply(median_rank, axis=1))

    return df


def add_median_ranks(df: pd.DataFrame, col_failure: str, col_status: str):
    """
    Calculates Bernard's median ranks when accounting for suspended units.  This function will add
    rank, revere rank, adjusted rank, and median rank columns.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe
    col_failure: str
        Column containing failure data: days to failures, miles to failure, etc
    col_status: str
        Column containing status of each unit: Valid values are FAILED or SUSPENDED
    """

    prev_adj_rank = [0]

    def adj_rank(series):
        if series[col_status] == "SUSPENDED":
            return "SUSPENSION"
        else:
            adjusted_rank = (series['reverse_rank'] * 1.0 * prev_adj_rank[-1] + (len(df) + 1)) / (series['reverse_rank'] + 1)
            prev_adj_rank.append(adjusted_rank)
            return adjusted_rank

    def median_rank(series):
        if series['adjusted_rank'] == "SUSPENSION":
            return None
        else:
            median_rank = (series['adjusted_rank'] - 0.3) / (len(df) + 0.4)
            return median_rank

    return (
        df.sort_values(by=col_failure)
        .assign(rank=range(1, len(df)+1))
        .assign(reverse_rank=range(len(df), 0, -1))
        .assign(adjusted_rank=df.apply(adj_rank, axis=1))
        .assign(median_rank=df.apply(median_rank, axis=1))
    )
