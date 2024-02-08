"""
Helpers for reading data from disk

"""

import os
import yaml
import shutil
import pathlib
from functools import cache

import sqlite3
import pandas as pd
from openmovement.load import CwaData
from tqdm import tqdm

from . import util, parse, clean


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


def raw_meal_info() -> pd.DataFrame:
    """
    Meal info exactly as it appears in the CSV

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["meal_info"]
    return pd.read_csv(path)


def _datetime(meal_info: pd.DataFrame) -> pd.Series:
    """
    Get a series representing the timestamp

    """
    return pd.to_datetime(
        meal_info["date"].map(str) + meal_info["timestamp"], format=r"%d%b%Y%H:%M:%S"
    )


@cache
def all_meal_info(*, verbose=False) -> pd.DataFrame:
    """
    Get smartwatch meal info from the smartwatch data; sorted by entry timestamp

    :param verbose: extra print output
    :returns: dataframe where the date and timestamp are combined into a single column and set as the index

    """
    retval = raw_meal_info()

    # Find a series representing the timestamp
    retval["Datetime"] = _datetime(retval)

    # Set it as the index
    retval = retval.set_index("Datetime")

    # We likely care more about the time since the start of the study
    retval = add_timedelta(retval)

    # Remove the old date/time columns
    retval = retval.drop(["date", "timestamp"], axis=1)

    # Add catchup info
    retval = clean.flag_catchups(retval)
    retval = clean.flag_catchup_entries(retval)

    # Remove incorrect Ramadan flags
    retval = retval[[col for col in retval if "ramadanflag" not in col]]

    # Remove the start/end dates, since they're wrong
    retval = retval[[col for col in retval if col not in {"firstdate", "lastdate"}]]

    return retval


def meal_info(participant_id: str) -> pd.DataFrame:
    """
    Get smartwatch meal info for a single participant from the smartwatch data

    :param participant_id: ID of participant
    :returns: dataframe

    """
    p_id = int(participant_id)
    all_meals = all_meal_info()

    return all_meals[all_meals["p_id"] == p_id]


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


@cache
def income_data() -> pd.DataFrame:
    """
    Get household income data

    Hold information about survey responses relating to household income; the codebook
    can be found on RDSF (and maybe I'll write a function to grab a dict of encoded answers)


    :returns: dataframe holding survey responses

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["income_info"]

    return pd.read_csv(path)


def income_codebook() -> dict:
    """
    Dict of dicts for the answers to each question

    5 and 6 are in a weird order in the codebook - maybe they were swapped in the survey?

    """
    # Parsed from the income codebook
    return {
        "income_1": {
            "q": "In the last 12 months did the household have enough money to make ends meet?",
            -9: "Refused to answer",
            -8: "Don't Know",
            1: "Yes",
            2: "No",
        },
        "income_2": {
            "q": "38. In the last week did anyone in the household go hungry because there was not",
            -9: "Refused to answer",
            -8: "Don't Know",
            1: "Yes",
            2: "No",
        },
        "income_3": {
            "q": "39. In the last 12 months, did anyone in the household fail to receive needed me",
            -9: "Refused to answer",
            -8: "Don't Know",
            1: "Yes",
            2: "No",
        },
        "income_4": {
            "q": "40. In an emergency, could the household raise RM2,000 in 24 hours?",
            -9: "Refused to answer",
            -8: "Don't Know",
            1: "Yes",
            2: "No",
        },
        "income_5": {
            "q": "42. Please define how well off is your economic status.",
            -9: "Refused to answer",
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            10: 9,
            11: 10,
        },
        "income_6": {
            "q": "41. Which category best describes the household's monthly income?",
            -9: "Refused to answer",
            -8: "Don't Know",
            1: "Less than RM 1,000 per month",
            2: "RM 1,000 - RM 1,999",
            3: "RM 2,000 - RM 2,999",
            4: "RM 3,000 - RM 3,999",
            5: "RM 4,000 - RM 4,999",
            6: "RM 5,000 - RM 5,999",
            7: "RM 6,000 and above",
        },
    }


def qnaire_status_codebook() -> dict:
    """
    Lookup of response status -> response for the "respondent_status" question

    :returns: dict mapping number to string response

    """
    return {
        1: "Agree",
        2: "Disagree",
        3: "Not at Home (Uncontactable)",
        4: "Empty/ Moved",
        5: "Passed away",
        6: "Exclude",
    }


def qnaire_sex_codebook() -> dict():
    """
    Lookup of response status -> response for the "respondent_sex" question

    """
    return {1.0: "Male", 2.0: "Female"}


def qnaire_ethnicity_codebook() -> dict():
    """
    Lookup of response status -> readable ethnicity

    It's not easy to find what these are in the codebook, so I've left them as numbers

    """
    return {
        1.0: "E1",
        2.0: "E2",
        3.0: "E3",
        5.0: "E4",
    }


@cache
def smartwatch_feasibility() -> pd.DataFrame:
    """
    Get a dataframe of smartwatch feasibility data

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["feasibility_info"]
    return pd.read_stata(path)


@cache
def full_questionnaire() -> pd.DataFrame:
    """
    Get a dataframe of the full questionnaire data

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["full_questionnaire"]
    return pd.read_stata(path)


@cache
def full_codebook() -> pd.DataFrame:
    """
    Get a dataframe of the full questionnaire codebook

    """
    path = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["qnaire_codebook"]
    return pd.read_excel(path, sheet_name="Sheet1")


def ramadan22_df(dataframe: pd.DataFrame, *, keep: bool) -> pd.DataFrame:
    """
    Get a copy of a dataframe containing only/none of the dates in ramadan 2022

    :param dataframe: must be indexed by datetime
    :param keep: whether to keep (True) or remove (False) the ramadan dates

    :returns: a copy of the dataframe

    """
    in_ramadan = util.in_ramadan_2022(dataframe.index)

    return dataframe[in_ramadan].copy() if keep else dataframe[~in_ramadan].copy()


def no_collection_date(participant_ids: pd.Series) -> set:
    """
    Find whether a collection date was given for each participant

    :param participant_ids: series of participant IDs
    :returns: set of participants for whom no collection date was given

    """
    feasibility_info = smartwatch_feasibility()

    # Check none of the provided participant IDs aren't in the feasibility info
    assert participant_ids.isin(
        feasibility_info["residents_id"]
    ).all(), "Provided participant ID not in the list of possible residents IDs"

    # Slice the feasibility info to only include the participants we're interested in
    feasibility_info = feasibility_info[
        feasibility_info["residents_id"].isin(participant_ids)
    ]

    # Could check that for all the provided participants, if collection date is not provided
    # then neither watch or AX collection date are provided either

    # Check whether the collection date is null for each participant
    return set(
        feasibility_info["residents_id"][
            pd.isnull(feasibility_info["collectiondate_actual"])
        ]
    )


def add_timedelta(meal_info: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column showing the delta between watch distribution date and entry date
    to a dataframe

    Also adds Datetime, residents_id column

    """
    feasibility_info = smartwatch_feasibility()

    # We only care about ones who consented to the smartwatch study
    feasibility_info = feasibility_info[feasibility_info["smartwatchwilling"] == 1]
    feasibility_info = feasibility_info[["residents_id", "actualdateofdistribution1st"]]

    # Join dataframes
    meal_info = (
        meal_info.reset_index()
        .merge(feasibility_info, left_on="p_id", right_on="residents_id", how="left")
        .set_index(meal_info.index)
    )

    meal_info["delta"] = (
        meal_info.index.to_series() - meal_info["actualdateofdistribution1st"]
    )

    assert (meal_info["residents_id"] == meal_info["p_id"]).all()

    return meal_info.drop(
        columns=["Datetime", "residents_id", "actualdateofdistribution1st"]
    )


def n_days_in_school(*, subset: str = None) -> dict:
    """
    Find how many days each participant was in school for during the study

    Q: In the last 7 days, how many days did you attend school (except home-based teaching)?

    :param subset: whether to only include participants who consented to the smartwatch study

    """
    assert subset in {None, "smartwatch", "not smartwatch"}

    feasibility_df = _qnaire_df()

    # Only the respondents who consented
    feasibility_df = feasibility_df[feasibility_df["respondent_status"] == 1]

    if subset is None:
        keep = ...
    elif subset == "smartwatch":
        keep = feasibility_df["smartwatchwilling"] == 1
    elif subset == "not smartwatch":
        keep = feasibility_df["smartwatchwilling"] == 2
    else:
        raise ValueError

    return dict(
        zip(feasibility_df["residents_id"][keep], feasibility_df["phyactq1"][keep])
    )


def ax6_summary():
    """
    Cleaned summary data for AX6 accelerometers

    """
    return pd.read_stata(
        pathlib.Path(_userconf()["seaco_dir"]) / _conf()["ax6_summary"]
    )


def ax6_day_summary(*, part: int):
    """
    day summary data for AX6 accelerometers

    """
    assert part in {2, 5}

    return pd.read_csv(
        pathlib.Path(_userconf()["seaco_dir"]) / _conf()[f"ax6_day_summary_pt{part}"]
    )


def ax6_person_summary(*, part: int):
    """
    person summary data for AX6 accelerometers

    """
    assert part in {2, 5}

    return pd.read_csv(
        pathlib.Path(_userconf()["seaco_dir"]) / _conf()[f"ax6_person_summary_pt{part}"]
    )


def ax6_data_quality():
    """
    Data quality report

    """
    return pd.read_csv(
        pathlib.Path(_userconf()["seaco_dir"]) / _conf()["ax6_data_quality"]
    )


def copy_battery_files():
    """
    Copy the smartwatch databases containing battery level to a local directory so I
    can read them etc

    """
    # Create an output file directory
    battery_dir = (pathlib.Path(__file__).parents[1] / "data" / "battery_dbs").resolve()
    if not battery_dir.is_dir():
        battery_dir.mkdir(parents=True)

    # Find all the files
    dirname = pathlib.Path(_userconf()["seaco_dir"]) / _conf()["smartwatch_dbs_dir"]
    assert dirname.exists()

    # Recurse into all "Week X" directories, extracting all .db files
    source_files = [file for file in dirname.glob("Week*/**/*.db")]
    dest_files = [os.path.join(battery_dir, file.name) for file in source_files]

    # Check that they don't all exist
    if not all([os.path.isfile(dest) for dest in dest_files]):
        for source, dest in tqdm(
            zip(source_files, dest_files), total=len(source_files)
        ):
            assert not os.path.isfile(dest), f"Duplicate name {source}"
            shutil.copyfile(source, dest)
