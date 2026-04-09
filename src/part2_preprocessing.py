'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> df_arrests['num_fel_arrests_last_year'].mean()
- Print df_arrests.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np

# Your code here

def join_add():
    # load csv's into df's
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')
    
    # merge the two df's
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # set to standard datetime format
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])

    # create new column that = 1 if felony arrest within one year of arrest date
    df_arrests = df_arrests.sort_values(['person_id', 'arrest_date_event'])

    df_arrests['y'] = df_arrests.groupby('person_id').apply(
        lambda x: (
            (x['arrest_date_event'].shift(-1) - x['arrest_date_event'] <= pd.Timedelta(days=365)) & 
            (x['charge_degree'].shift(-1) == 'felony')
        )
    ).reset_index(level=0, drop=True).astype(int)

    # sort by index
    df_arrests.sort_index(inplace= True)

    # check and print how many people were rearrested for a felony within a year
    felony_reoffs = df_arrests[df_arrests['y'] == 1]['person_id'].nunique()
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?: {felony_reoffs}")

    # predictive feature that identifies if current charge was a felony
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'felony').astype(int)
    
    # print and answer question What share of current charges are felonies?
    print(f"What share of current charges are felonies?: {df_arrests['current_charge_felony'].sum()}")

    # predictive feature that identifies how many felony arrests in one year before current charge
    df_arrests['num_fel_arrests_last_year'] = (
        df_arrests.groupby('person_id', group_keys=False)
        .apply(lambda x: x.sort_values('arrest_date_event') # Force monotonic order per group
                        .rolling('365D', on='arrest_date_event', closed='left')['current_charge_felony']
                        .sum())
        .fillna(0)
    ).astype(int)

    # answer and print question
    print(f"What is the average number of felony arrests in the last year?: {df_arrests['num_fel_arrests_last_year'].mean()}")

    print(df_arrests.head())

    df_arrests.to_csv('data/df_arrests.csv', index=False)



    



