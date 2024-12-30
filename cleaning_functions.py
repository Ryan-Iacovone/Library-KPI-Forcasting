import pandas as pd  
import datetime as dt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kurtosis, skew #correlation coefficient, skewness, and kurtosis 
import seaborn as sns #used for color palletes on graphs and boxplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller # Dicky Fuller test for stationarity
import statsmodels.api as smm 
import statsmodels.formula.api as sm  # Multiple Linear Regression
from statsmodels.graphics.tsaplots import plot_acf #autocorrelation plot 
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMA Model import
#from pylab import rcParams # Decomposition of time series

def expand_dates(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    #We use the map function along with a lambda function to apply the strftime method to each datetime object in the index. 
    # This approach is necessary because the strftime method is designed to work with individual datetime objects, not with entire Series or DataFrames

    df['dayofweekchar'] = df.index.map(lambda x: x.strftime('%A'))  # Full day name, e.g., "Monday" 
    # Define the desired order of days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Convert 'day_of_week_char' to a categorical variable with the specified order
    df['dayofweekchar'] = pd.Categorical(df['dayofweekchar'], categories=day_order, ordered=True)

    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['month_char'] = df.index.map(lambda x: x.strftime('%b'))  # Full month name, e.g., "September"

    # Define the custom order for the month names
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Convert 'month_char' to a categorical data type with the custom order
    df['month_char'] = pd.Categorical(df['month_char'], categories=month_order, ordered=True)
     # Sort the DataFrame based on the custom order

    df['year'] = df.index.year

    #turns it not to numeric for whatever reasosn
    df['dayofyear'] = df.index.dayofyear
    df['dayofyear'] = pd.to_numeric(df['dayofyear'])

    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


# Reading in new daily circulation data (covers from 2016-2023) 
def read_clean():

    circ_day = pd.read_excel(r"data\new_daily_checkouts.xlsx").sort_values(by = "date")

    circ_day.set_index('date', inplace=True) # Setting the date variable as the index
    egr_day = circ_day[(circ_day["branch"] == "EGR")] # Filtering to only the EGR branch

    # Creating a new df that has all dates listed in the range of our df (kept as circ_day for bigger range)
    start_date = min(circ_day.index)
    end_date = max(circ_day.index)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D') # Filling out our df
    df_dates = pd.DataFrame({'date': date_range}).set_index('date') 

    egr_circ = pd.merge(df_dates, egr_day, on= ['date'], how='left') # Adding the extra dates to our df 
    egr_circ = egr_circ.drop(columns= ["open_hours", "branch"]) # Dropping the open hours and branch variables because irrelevant 

    # getting in a whole bunch of date variations from function abaove 
    egr_circ = expand_dates(egr_circ)

    # Filling in all new transaction data with a zero
    egr_circ['transactions'].fillna(0, inplace=True)

    # Changing transactions from float64 to int variable
    egr_circ["transactions"] = egr_circ["transactions"].astype(int)

    return egr_circ
