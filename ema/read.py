"""
Helpers for reading data from disk

"""
import yaml
import pathlib
from functools import cache

import pandas as pd
from openmovement.load import CwaData

from . import util, parse


def _data_dir() -> pathlib.Path:
    """
    Directory where data is stored

    :returns: path object to the directory

    """
    return pathlib.Path(__file__).parents[1] / "data"


@cache
def _userconf() -> dict:
    """
    User defined configuration

    """
    with open(
        pathlib.Path(__file__).resolve().parents[1] / "userconf.yaml", "r"
    ) as stream:
        return yaml.safe_load(stream)


@cache
def _conf() -> dict:
    """
    Hard coded stuff

    """
    with open(
        pathlib.Path(__file__).resolve().parents[1] / "config.yaml", "r"
    ) as stream:
        return yaml.safe_load(stream)


@cache
def _qnaire_df() -> pd.DataFrame:
    """
    Read the questionnaire dataframe

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["questionnaire"]

    return pd.read_csv(path)


def consented(residents_id: str) -> bool:
    """
    Whether a participant consented, based on the questionnaire answer

    """
    # Read in the file of questionnaire information
    qnaire_responses = _qnaire_df()

    r_id = int(residents_id)

    # Check that this value is in the df
    if r_id not in qnaire_responses["residents_id"].values:
        raise ValueError(f"{residents_id} not found in questionnaire responses")

    # Check that the participant consented
    # i.e. that the status is 1
    # im tired so this is bad
    return (
        qnaire_responses.loc[
            qnaire_responses["residents_id"] == r_id, "respondent_status"
        ].values
        == [1]
    ).all()


def accel_filepath(
    device_id: str, recording_id: str, participant_id: str
) -> pathlib.Path:
    """
    Absolute path to a file holding accelerometer data

    TODO this function should also check that the participant agreed to take part in the study

    :param device_id: 7-digit string holding the ID of the accelerometer
    :param recording_id: 10-digit string holding the ID of the recording
    :param participant_id: the ID of the participant

    :returns: path object to the right accelerometer file
    :raises AssertionError: if device or recording ID have the wrong length

    """
    assert len(str(device_id)) == 7
    assert len(str(recording_id)) == 10

    # Check that the participant agreed to take part in the study
    # Do this by opening "SEACO data\SEACO-CH20 qnaire data\SEACO_CH20_17082022_de_id.csv"
    # as a dataframe, finding the row with participant_id = df["residents_id"] and checking
    # that the value of "respondent_status" here == 1
    if not consented(participant_id):
        raise ValueError(f"Participant {participant_id} didn't consent")

    filename = f"{device_id}_{recording_id}-{participant_id}.cwa"
    return _data_dir() / filename


def meal_info(participant_id: str) -> pd.DataFrame:
    """
    Get smartwatch meal info from the smartwatch data

    :param participant_id: ID of participant
    :returns: dataframe

    """
    p_id = int(participant_id)

    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["meal_info"]

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
        retval = cwa_data.get_samples()

    retval.set_index("time", inplace=True, verify_integrity=False)

    # Convert from g to m/s
    for x in "xyz":
        retval[f"accel_{x}"] = retval[f"accel_{x}"] * util.GRAVITY_MS2

    return retval


def accel_filepath(
    device_id: str, recording_id: str, participant_id: str
) -> pathlib.Path:
    """
    Assumes in the data/ dir
    """
    filename = f"{device_id}_{recording_id}-{participant_id}.cwa"
    filepath = pathlib.Path(rf"data/{filename}")

    # TODO if reading straight from RDSF, add some code to look for the right files in all the "Week X" folders

    assert filepath.exists()

    return filepath


def get_participant_meal(
    device_id: str,
    recording_id: str,
    participant_id: str,
    meal_no: int,
) -> pd.DataFrame:
    """
    Get the accelerometer information from an hour preceding the provided meal for the given participant

    :param device_id: 7-digit device ID
    :param recording_id: 10-digit device ID
    :param participant_id: 5-digit participant ID
    :param meal_no: which meal to take the accelerometer information from

    :raises ValueError: if the participant did not consent
    :raises ValueError: if the participant file doesn't exist
    :returns: a dataframe holding the accelerometer information for the hour. Uses time as the index

    """
    # Get the whole dataframe
    samples = accel_info(str(accel_filepath(device_id, recording_id, participant_id)))

    # Slice the right meal
    # TODO refactor this cus its bad, i cba rn
    meal_df = meal_info(participant_id)
    allowed_meal_types = {"Snack", "Drink", "Meal", "No food/drink"}
    meal_df = parse.extract_meals(meal_df, allowed_meal_types, verbose=True)

    # Find meal times
    meal_times = meal_df["date"].map(str) + meal_df["timestamp"]

    # Find an hour slot before each meal
    ends = pd.to_datetime(meal_times, format=r"%d%b%Y%H:%M:%S")
    starts = ends - pd.Timedelta(1, "hour")

    start, end = starts.iloc[meal_no], ends.iloc[meal_no]

    # Copy so the original df doesn't have to be held in memory
    return samples[start:end].copy()


def income_data() -> pd.DataFrame:
    """
    Get household income data 

    Hold information about survey responses relating to household income; the codebook
    can be found on RDSF (and maybe I'll write a function to grab a dict of encoded answers)


    :returns: dataframe holding survey responses

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["income_info"]

    return pd.read_csv(path)

