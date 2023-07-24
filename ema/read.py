"""
Helpers for reading data

"""
import pandas as pd

def meal_info(participant_id: str) -> pd.DataFrame:
    """
    Get smartwatch meal info from the smartwatch data

    :param participant_id: ID of participant
    :returns: dataframe

    """
    p_id = int(participant_id)

    # TODO get this from conf
    path = r"Z:\SEACO data\SEACO-CH20_Smartwatch_data\New_files\combine_csv_file.csv"

    df = pd.read_csv(path)
    return df[df["p_id"] == p_id]