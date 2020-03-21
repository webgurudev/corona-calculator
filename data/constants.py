"""
Range estimates for various epidemiology constants. Currently only default values are used.
For sources, please visit https://www.notion.so/Modelling-d650e1351bf34ceeb97c82bd24ae04cc
"""

import datetime
import pickle
from pathlib import Path

import pandas as pd

from data.preprocessing import preprocess_bed_data
from s3_utils import download_file_from_s3, S3_DISEASE_DATA_OBJ_NAME
from fetch_live_data import download_data


_DATA_DIR = Path(__file__).parent
_DEMOGRAPHICS_DATA_PATH = _DATA_DIR / "demographics.csv"
_BED_DATA_PATH = _DATA_DIR / "world_bank_bed_data.csv"
_AGE_DATA_PATH = _DATA_DIR / "age_data.csv"

DEMOGRAPHIC_DATA = pd.read_csv(_DEMOGRAPHICS_DATA_PATH, index_col="Country/Region")
BED_DATA = preprocess_bed_data(_BED_DATA_PATH)
AGE_DATA = pd.read_csv(_AGE_DATA_PATH, index_col="Age Group")


def build_country_data(demographic_data=DEMOGRAPHIC_DATA, bed_data=BED_DATA):
    # Try to download from S3, else download from JHU
    objects = download_file_from_s3(S3_DISEASE_DATA_OBJ_NAME)
    if objects is None:
        data_dict = download_data()
        last_modified = datetime.datetime.now()
    else:
        data_dict_pkl_bytes, last_modified = objects
        data_dict = pickle.loads(data_dict_pkl_bytes)

    full_disease_data = data_dict["full_table"]
    latest_disease_data = data_dict["latest_table"]

    latest_disease_data = latest_disease_data.set_index("Country/Region", drop=True)
    
    # Rename name "US" to "United States" in disease and demographics data to match bed data
    latest_disease_data = latest_disease_data.rename(index={"US": "United States"})
    demographic_data = demographic_data.rename(index={"US": "United States"})

    full_disease_data['Country/Region'] = full_disease_data['Country/Region'].\
        apply(lambda x: {"US": "United States"}.get(x, x))

    country_data = latest_disease_data.merge(demographic_data, on="Country/Region")

    # Beds are per 1000 people so we need to calculate absolute

    bed_data = bed_data.merge(demographic_data, on="Country/Region")

    bed_data["Num Hospital Beds"] = (
        bed_data["Latest Bed Estimate"] * bed_data["Population"] / 1000
    )

    country_data = country_data.merge(
        bed_data[["Num Hospital Beds"]], on="Country/Region"
    )

    # Check that all of the countries in our selectable dropdown are also present in the full data
    assert set(latest_disease_data.index.unique()).issubset(set(full_disease_data['Country/Region'].unique()))

    return country_data.to_dict(orient="index"), last_modified, full_disease_data


class Countries:
    def __init__(self, timestamp):
        self.country_data, self.last_modified, self.historical_country_data = build_country_data()
        self.countries = list(self.country_data.keys())
        self.default_selection = self.countries.index("Canada")
        self.timestamp = timestamp

    @property
    def stale(self):
        delta = datetime.datetime.utcnow() - self.timestamp
        return delta > datetime.timedelta(hours=1)


class AgeData:
    data = AGE_DATA


"""
SIR model constants
"""


class RecoveryRate:
    default = 1 / 10  # Recovery period around 10 days


class MortalityRate:
    # Take weighted average of death rate across age groups. This assumes each age group is equally likely to
    # get infected, which may not be exact, but is an assumption we need to make for further analysis,
    # notably segmenting deaths by age group.
    default = (AgeData.data.Proportion * AgeData.data.Mortality).sum()


class CriticalDeathRate:
    # Death rate of critically ill patients who don't have access to a hospital bed.
    # This is the max reported from Wuhan:
    # https://wwwnc.cdc.gov/eid/article/26/6/20-0233_article
    default = 0.122


class TransmissionRatePerContact:
    # Probability of a contact between carrier and susceptible leading to infection.
    # Found using binomial distribution in Wuhan scenario: 14 contacts per day, 10 infectious days, 2.5 average people infected.
    default = 0.018


class AverageDailyContacts:
    min = 0
    max = 50
    default = 15


"""
Health care constants
"""


class ReportingRate:
    # Proportion of true cases diagnosed
    default = 0.14


class HospitalizationRate:
    # Cases requiring hospitalization. We multiply by the ascertainment rate because our source got their estimate
    # from the reported cases, whereas we will be using it with total cases.
    default = 0.19 * ReportingRate.default
