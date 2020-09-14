import pandas as pd
import numpy as np

from datetime import datetime


def store_relational_JH_data_confirmed():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path='data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')

    pd_relational_model.to_csv('data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))

def store_relational_JH_data_deaths():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path='data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'deaths'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')

    pd_relational_model.to_csv('data/processed/COVID_relational_deaths.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))

def store_relational_JH_data_recovered():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path='data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'recovered'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')

    pd_relational_model.to_csv('data/processed/COVID_relational_recovered.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))


def store_confimed_data_for_sir():
    data_path = 'data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path)
    pd_raw = pd_raw.drop(['Lat','Long','Province/State'],axis=1)
    pd_raw = pd_raw.rename(columns={'Country/Region':'country'})
    pd_flat_table = pd_raw.set_index('country') \
                    .T \
                    .stack(level=[0]) \
                    .reset_index() \
                    .rename(columns={'level_0':'date',
                                    0:'confirmed'}
                                    )
    pd_flat_table['date'] = pd_flat_table.date.astype('datetime64[ns]')
    pd_flat_table = pd.pivot_table(pd_flat_table, values='confirmed', index='date', columns='country', aggfunc=np.sum, fill_value=0).reset_index()
    pd_flat_table.to_csv('data/processed/COVID_full_flat_table.csv',sep=';',index = False)
    #print(pd_flat_table.tail())
    print('Number of rows stored - Full Flat Table: '+str(pd_flat_table.shape[0]))

if __name__ == '__main__':

    store_relational_JH_data_confirmed()
    store_relational_JH_data_deaths()
    store_relational_JH_data_recovered()
    store_confimed_data_for_sir()