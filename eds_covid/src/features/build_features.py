import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal


def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        Doubling rate: double
    '''

    y = np.array(in_array)
    #print(y)
    X = np.arange(-1,2).reshape(-1, 1)
    #print(X)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_


    return intercept/slope


def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function 
        it ensures that the data structure is kept'''
    window=5, 
    degree=1
    df_result=df_input
    
    filter_in=df_input[column].fillna(0) # attention with the neutral element here
    
    result=signal.savgol_filter(np.array(filter_in),
                           5, # window size used for filtering
                           1)
    df_result[column+'_filtered']=result
    return df_result


def rolling_reg(df_input,col='confirmed'):
    ''' input has to be a data frame'''
    ''' return is single series (mandatory for group by apply)'''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result

def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()

    #print('--+++ after group by apply')
    #print(pd_filtered_result[pd_filtered_result['country']=='Germany'].tail())

    #df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    #print(df_output[df_output['country']=='Germany'].tail())
    return df_output.copy()

def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output


def get_daily_list(total_list):
    ''' Calculate Daily change in cases from the cummulative gathered list

        Parameters:
        ----------
        total_list: list
            A python list containing the cummulative cases
        
        Returns:
        ----------
        df_output: list
            the result will be a list containing daily change in values
    '''
    daily_list=[]
    daily_list.append(total_list.pop(0))
    for each in range(len(total_list)):
        if each == 0:
            daily_list.append(total_list[each] - total_list[0])
        else:
            daily_list.append(total_list[each] - total_list[each-1])
    
    return daily_list

def calc_daily_values_all_countries(all_countries):
    df_daily_all= pd.DataFrame()
    for each_country in all_countries:
        daily_list = get_daily_list(list(pd_daily[pd_daily['country']==each_country]['confirmed']))
        df_daily = pd.DataFrame(np.array(daily_list))

        
        df_daily_death = np.array(get_daily_list(list(pd_daily[pd_daily['country']==each_country]['deaths'])))
        df_daily_recov = np.array(get_daily_list(list(pd_daily_recov[pd_daily['country']==each_country]['recovered'])))


        df_daily = df_daily.rename(columns={0:'daily_confirmed'})
        df_daily['daily_deaths'] = df_daily_death
        df_daily['daily_recovered'] = df_daily_recov
        df_daily['date'] = np.array(pd_daily[pd_daily['country']==each_country]['date'])
        df_daily['country'] = np.array(pd_daily[pd_daily['country']==each_country]['country'])
        df_daily_all = pd.concat([df_daily_all,df_daily])

        return df_daily_all




if __name__ == '__main__':
    test_data_reg=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data_reg)
    print('the test slope is: '+str(result))
    pd_JH_data=pd.read_csv('data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()

    #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
    #                  (pd_JH_data['country']=='Germany'))]

    pd_result_larg=calc_filtered_data(pd_JH_data)
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')
    print(pd_result_larg.head())

    mask=pd_result_larg['confirmed']>100
    pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)

    pd_JH_data_deaths=pd.read_csv('../data/processed/COVID_relational_deaths.csv',sep=';',parse_dates=[0])
    pd_JH_data_deaths=pd_JH_data_deaths.sort_values('date',ascending=True).reset_index(drop=True).copy()
    pd_DR_result_death = pd_JH_data_deaths[['state','country','deaths']].reset_index()
    pd_result_larg=pd.merge(pd_result_larg,pd_DR_result_death[['index','deaths']],on=['index'],how='left')

    pd_result_larg.to_csv('data/processed/COVID_final_set.csv',sep=';',index=False)

    pd_JH_data_recov=pd.read_csv('../data/processed/COVID_relational_recovered.csv',sep=';',parse_dates=[0])
    pd_JH_data_recov=pd_JH_data_recov.sort_values('date',ascending=True).reset_index(drop=True).copy()
    pd_DR_result_recov = pd_JH_data_recov[['state','date','country','recovered']].reset_index()
    pd_DR_result_recov.to_csv('../data/processed/COVID_final_recov_set.csv',sep=';',index=False)


    pd_daily = pd_result_larg[['state','country','confirmed','date','deaths']].groupby(['country','date']).agg(np.sum).reset_index()
    pd_daily_recov = pd.read_csv('../data/processed/COVID_final_recov_set.csv',sep=';',parse_dates=[0])
    pd_daily_recov = pd_daily_recov[['state','country','recovered','date']].groupby(['country','date']).agg(np.sum).reset_index()

    df_daily_all= calc_daily_values_all_countries(pd_daily['country'].unique())
    df_daily_all = df_daily_all.reset_index()
    df_daily_all.daily_deaths = df_daily_all.daily_deaths.mask(df_daily_all.daily_deaths.lt(0), 0)
    df_daily_all.to_csv('../data/processed/COVID_final_daily_set.csv',sep=';',index=False)

    