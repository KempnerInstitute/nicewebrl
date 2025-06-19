from typing import Callable, List, Optional

import numpy as np
import polars as pl
from flax import struct

# Type definitions
REMOVE = bool
EPISODE = struct.PyTreeNode
EPISODE_FILTER = Callable[[EPISODE], REMOVE]


class DataFrame(object):
  def __init__(
    self,
    df: pl.DataFrame,
    episodes: List[struct.PyTreeNode],
    index_key: str = "index",
    reindex: bool = True,
  ):
    """
    Initialize the DataFrame object.

    Args:
        df: A polars DataFrame.
        episodes: A list of PyTreeNode objects representing episodes.
        index_key: The name of the index column (default: 'index').
    """
    if reindex or "index" not in df.columns:
      if "index" in df.columns:
        df = df.drop("index")
      df = df.with_row_count("index")
    self._df = df
    self._episodes = episodes
    self._index_key = index_key

  def reindex(self):
    df = self._df
    if "index" in df.columns:
      df = df.drop("index")
    df = df.with_row_count("index")
    return DataFrame(df, self._episodes, self._index_key)

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    """
    Provide proxy access to regular attributes of the wrapped polars DataFrame.

    Args:
        name: The attribute name to access.

    Returns:
        The attribute value from the wrapped DataFrame, with methods wrapped to handle DataFrame outputs.
    """
    attr = getattr(self._df, name)

    if callable(attr):
      # If it's a method, wrap it to check its output
      def wrapped_method(*args, **kwargs):
        result = attr(*args, **kwargs)
        if isinstance(result, pl.DataFrame):
          idxs = np.array(result[self._index_key])
          episodes = [self._episodes[idx] for idx in idxs]
          return DataFrame(result, episodes, index_key=self._index_key)
        return result

      return wrapped_method

    return attr

  def filter(
    self,
    *args,
    episode_filter: Optional[EPISODE_FILTER] = None,
    reindex: Optional[bool] = True,
    **kwargs,
  ):
    """
    Filter the DataFrame and corresponding episodes based on given conditions.

    Args:
        **kwargs: Keyword arguments for filtering conditions.

    Returns:
        A new DataFrame object with filtered data and episodes.
    """
    if len(args) == 0 and len(kwargs) == 0:
      assert episode_filter is not None, (
        "need either episode_filter or kwargs for dataframe filter"
      )
      return self._filter_episodes(episode_filter)

    df = self._df.filter(*args, **kwargs)
    idxs = np.array(df[self._index_key])
    episodes = [self._episodes[idx] for idx in idxs]

    df = DataFrame(df=df, episodes=episodes, index_key=self._index_key, reindex=reindex)

    if episode_filter is not None:
      df = df._filter_episodes(episode_filter)
    return df

  def _filter_episodes(self, episode_filter: EPISODE_FILTER):
    """
    Filter rows and episodes based on a given filter function.

    Args:
        filter_fn: A function that takes an episode and returns a boolean.

    Returns:
        A new DataFrame object with filtered rows and episodes.
    """
    new_rows = []
    new_episodes = []

    # Define schema explicitly based on the original dataframe
    schema = {col: self._df[col].dtype for col in self._df.columns}

    for idx, row in enumerate(self._df.iter_rows(named=True)):
      if not episode_filter(self.episodes[idx]):
        new_rows.append(row)
        new_episodes.append(self._episodes[idx])

    # Use the explicit schema when creating the new dataframe
    new_df = pl.DataFrame(new_rows, schema=schema)
    return DataFrame(df=new_df, episodes=new_episodes, index_key=self._index_key)

  def filter_by_group(
    self,
    input_episode_filter: EPISODE_FILTER,
    input_settings: dict,
    output_settings: dict,
    output_episode_filter: Optional[EPISODE_FILTER] = None,
    group_key: str = "user_id",
  ):
    """
    Create a subset of the DataFrame by applying filtering steps based on a group key.

    This function performs the following steps:
    1. Splits the DataFrame into groups based on unique values in the group_key column.
    2. For each group:
       a. Applies an initial filter (input_settings) to the group.
       b. Evaluates the filtered group using input_episode_filter to decide whether to include it.
       c. If included, adds the filtered group data to the subset.
    3. Combines the included groups into a new DataFrame.

    Args:
        input_episode_filter: A function that takes a filtered DataFrame and returns a boolean.
                              If True, the group is excluded from the final subset.
        input_settings: A dictionary of filtering conditions for the initial group evaluation.
        output_settings: A dictionary of filtering conditions for the final output.
        output_episode_filter: An optional function to filter episodes in the output.
        group_key: The column name to use for splitting the DataFrame into groups (default: 'user_id').

    Returns:
        A new DataFrame object containing the filtered subset of data and episodes.
    """
    dfs = []
    episodes = []

    if group_key in output_settings:
      # if you're only getting a single value from splitting_key, then you don't need to iterate
      keys = [output_settings[group_key]]
    else:
      # Get unique values in the splitting_key column
      keys = self[group_key].unique().to_list()

    # Iterate through each unique key
    for key in keys:
      # Apply initial filter to the group
      final_input_settings = {group_key: key}
      final_input_settings.update(output_settings)
      final_input_settings.update(input_settings)

      filter_df = self.filter(**final_input_settings)

      # Apply filter_fn to determine if the group should be included
      remove = input_episode_filter(filter_df)
      if remove:
        continue

      final_output_settings = {group_key: key, **output_settings}

      ouput_df = self.filter(**final_output_settings)
      if output_episode_filter is not None:
        ouput_df = ouput_df._filter_episodes(episode_filter=output_episode_filter)

      dfs.append(ouput_df._df)
      episodes.extend(ouput_df.episodes)

    subset = (
      pl.concat(dfs, how="diagonal_relaxed")
      .with_row_count(name="row_number")
      .with_columns(pl.col("row_number").alias("index"))
      .drop("row_number")
    )
    return DataFrame(subset, episodes)

  def apply(
    self,
    fn,
    episode_filter: Optional[EPISODE_FILTER] = None,
    output_transform=lambda x: x,
    **kwargs,
  ):
    """
    Apply a function to each episode, with optional filtering and post-processing.

    Args:
        fn: The function to apply to each episode.
        episode_filter: An optional function to filter episodes.
        output_transform: An optional function to transform the results (default: identity function).
        **kwargs: Optional keyword arguments for filtering the DataFrame before applying the function.

    Returns:
        The result of applying output_transform to the list of function results.
    """
    if len(kwargs) > 0:
      eval_df = self.filter(**kwargs)
    else:
      eval_df = self
    array = []
    for e in eval_df.episodes:
      if episode_filter is not None:
        if episode_filter(e):
          continue
      val = fn(e)

      array.append(val)

    return output_transform(array)

  def apply_by_group(
    self,
    fn,
    input_episode_filter: EPISODE_FILTER,
    input_settings: dict,
    output_settings: dict,
    output_episode_filter: Optional[EPISODE_FILTER] = None,
    output_transform=lambda x: x,
    splitting_key: str = "user_id",
  ):
    """
    Split the DataFrame by a key, apply filters, and compute a function on each group.

    Args:
        fn: The function to apply to each group.
        input_episode_filter: A function to filter split groups.
        input_settings: Settings for filtering split groups.
        output_settings: Settings for filtering computation groups.
        output_episode_filter: An optional function to filter episodes in the output.
        output_transform: An optional function to transform the final output.
        splitting_key: The key to split the DataFrame by (default: 'user_id').

    Returns:
        The result of applying output_transform to the list of computed results.
    """
    keys = self[splitting_key].unique()
    array = []
    for key in keys:
      filter_df = self.filter(**input_settings, **{splitting_key: key})
      remove = input_episode_filter(filter_df)
      if remove:
        continue

      output_df = self.filter(**output_settings, **{splitting_key: key})
      val = output_df.apply(fn=fn, episode_filter=output_episode_filter)
      if len(val) > 0:
        array.append(np.array(val))

    return output_transform(array)

  @property
  def episodes(self):
    """
    Property to access the episodes list.

    Returns:
        The list of episodes.
    """
    return self._episodes

  def __len__(self):
    """
    Get the length of the DataFrame (number of episodes).

    Returns:
        The number of episodes in the DataFrame.
    """
    return len(self._episodes)

  def __getitem__(self, key):
    """
    Access columns of the wrapped polars DataFrame.

    Args:
        key: The column name or index to access.

    Returns:
        The specified column from the wrapped DataFrame.
    """
    return self._df[key]


def concat_list(*dfs: List[DataFrame]) -> DataFrame:
  """Concatenate multiple DataFrame objects into a single DataFrame."""
  _dfs = [df._df for df in dfs]  # Extract polars DataFrames

  episodes = []
  for df in dfs:  # Concatenate episode lists
    episodes.extend(df.episodes)

  return DataFrame(df=pl.concat(_dfs, how="diagonal_relaxed"), episodes=episodes)


# Module-level variables
CONCAT_DATAFRAMES = concat_list
