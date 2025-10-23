import os
import dill
import logging
import pandas as pd
import hashlib

logger = logging.getLogger(__name__)


def examine_cache(cache_path):
    # Check if the cache file exists
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found: {cache_path}")
        return None

    # Load and return the cached result
    try:
        with open(cache_path, 'rb') as f:
            cached_result = dill.load(f)
        logger.info(f"Successfully loaded cache from {cache_path}")
        return cached_result
    except Exception as e:
        logger.error(f"Error loading cache from {cache_path}: {e}")
        return None


def examine_cache_result(cache_dir: str, cache_name: str):
    """
    Examines the cache result stored in `cache_dir` for a given step.
    """
    # Generate the cache file name
    
    cache_path = os.path.join(cache_dir, str(cache_name))

    # Check if the cache file exists
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found: {cache_path}")
        return None

    # Load and return the cached result
    try:
        with open(cache_path, 'rb') as f:
            cached_result = dill.load(f)
        logger.info(f"Successfully loaded cache from {cache_path}")
        return cached_result
    except Exception as e:
        logger.error(f"Error loading cache from {cache_path}: {e}")
        return None

def display_cache_as_table(result):
    """
    Converts the cache result to a pandas DataFrame and displays it as a table.
    """
    if isinstance(result, dict):  
        # Convert dictionary result to DataFrame
        df = pd.DataFrame(list(result.items()), columns=["Key", "Value"])
    elif isinstance(result, list):  
        # Convert list result to DataFrame
        df = pd.DataFrame(result, columns=["Values"])
    elif isinstance(result, pd.DataFrame):
        # If already a DataFrame, use it directly
        df = result
    else:
        print("Cached result is not in a table-compatible format.")
        return
    
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Cache Result Table", dataframe=df)


# Example usage:
cache_directory = "/Users/llupo/dev/infogap/scratch/full_cache_gpt_en_zh/"  # Change this to your actual cache directory
cache_name = "9c6b24f5f9c3abf10d6c32902a4f82bc8bd17cb939651c5049de7504d7af0e3c"

result = examine_cache_result(cache_directory, cache_name)

print(result)

# if result is not None:
#    # Convert Polars DataFrame to Pandas DataFrame
#     df_pandas = result[0].to_pandas()

#     # Save as CSV
#     df_pandas.to_csv("english.csv", index=False)
# else:
#     print("Cache not found or failed to load.")