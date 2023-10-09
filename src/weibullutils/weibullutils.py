import pandas as pd


def add_rank(df: pd.DataFrame, col_to_sort: str):
    """
    Adds a rank column based on the column to sort. rank will equal: 1, 2, 3, ... , length of dataframe

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
    Adds new column containing Bernard's adjusted median rank which accounts for susppended units

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
