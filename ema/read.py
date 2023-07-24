"""
Helpers for reading data

"""
import pandas as pd
from openmovement.load import CwaData


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


def accel_info(filepath: str) -> pd.DataFrame:
    """
    Get accelerometer data from a CWA file

    This file should be an AX6 database that includes accelerometry and
    gyroscopic data.
    These files are Big (and accessed over the network if using RDSF drive),
    which means this function might be slow

    :param filepath: path to the CWA file
    :returns: the accelerometer, gyroscope and time data

    """
    with CwaData(filepath, include_accel=True, include_gyro=True) as cwa_data:
        return cwa_data.get_samples()
