import pandas as pd
import numpy as np
#from skfeature.function.similarity_based import fisher_score

def isfloat(str):
    try:
        int(str)
        return False
    except ValueError:
        return True
    
#1.1 Cleaning messy outcome labels
train_df = pd.read_csv("../data/cases_2021_train.csv") #link to dataset
train_df.groupby('outcome').size()

conditions = [
      (train_df['outcome'] == 'Discharged')
    | (train_df['outcome'] == 'Discharged from hospital')
    | (train_df['outcome'] == 'Hospitalized')
    | (train_df['outcome'] == 'critical condition')
    | (train_df['outcome'] == 'discharge')
    | (train_df['outcome'] == 'discharged')
    ,
      (train_df['outcome'] == 'Alive')
    | (train_df['outcome'] == 'Receiving Treatment')
    | (train_df['outcome'] == 'Stable')
    | (train_df['outcome'] == 'Under treatment')
    | (train_df['outcome'] == 'recovering at home 03.03.2020')
    | (train_df['outcome'] == 'released from quarantine')
    | (train_df['outcome'] == 'stable')
    | (train_df['outcome'] == 'stable condition')
    ,
      (train_df['outcome'] == 'Dead')
    | (train_df['outcome'] == 'Death')
    | (train_df['outcome'] == 'Stable')
    | (train_df['outcome'] == 'Deceased')
    | (train_df['outcome'] == 'Died')
    | (train_df['outcome'] == 'death')  
    | (train_df['outcome'] == 'died')
    ,
      (train_df['outcome'] == 'Recovered')  
    | (train_df['outcome'] == 'recovered'),
    ]
        
values = ['hospitalized', 'nonhospitalized', 'deceased', 'recovered']

for x in range(0,4): 
    train_df.loc[conditions[x],'outcome_group'] = values[x]

train_df.drop(columns=['outcome'],inplace=True)

#print(ct)



# 1.4


#train data
train_df = pd.read_csv('../data/cases_2021_train.csv')

#remove rows with missing age values
train_df = train_df.dropna(subset=['age'])


# fixing different formats in age column:

#reduce formats with "-"
train_age_list = train_df[train_df['age'].str.contains("-")].values.tolist()
train_age_ranges = []
k = 0
for range in train_age_list:
    train_age_ranges.append(train_age_list[k][0])
    k = k+1

train_age_dash = train_df[train_df['age'].str.contains("-")].index

train_age_values = []

for age in train_age_ranges:
    
    (x,y) = age.split("-")
    if x == '':
        x = 0
    if y == '':
        y = 0
    x = int(x)
    y = int(y)
    value = round((x+y)/2)
    train_age_values.append(value)

i = 0
for entry in train_age_dash:
    train_df.loc[entry, 'age'] = train_age_values[i]
    i = i+1


#reduce float formats
train_age_col = train_df['age']


for age in train_age_col:
    if isfloat(age) == True:
        num = age.split(".")
        row = train_df[train_df['age'] == age].index
        for entry in row:
            train_df.loc[entry, 'age'] = num[0]


train_df.drop(train_df[train_df['sex'].isnull()].index.tolist(),inplace=True)

prov_null = train_df[train_df['province'].isnull()].index.tolist()
for i in prov_null:

    new_df = train_df[(train_df['country']== train_df.loc[i][3]) 
    & (train_df['province'].notnull())].index.tolist()

    for k in new_df:
        if train_df.loc[k][3] ==  train_df.loc[i][3]:
            lat_value = int(train_df.loc[k][4]) - int(train_df.loc[i][4])
            long_value = int(train_df.loc[k][5]) - int(train_df.loc[i][5])
            if((abs(lat_value) <=1 or abs(long_value) >= 1) or (abs(lat_value) >=1 or abs(long_value) <= 1 )):
                train_df.loc[i,'province'] = train_df.loc[k][2]
                break

train_df['province'].fillna('Unknown',inplace=True)
train_df['country'].fillna('Unknown',inplace=True)
train_df['date_confirmation'].fillna('Unknown',inplace=True)
train_df['additional_information'].fillna('Unknown',inplace=True)
train_df['source'].fillna('Unknown',inplace=True)


#test data
test_df = pd.read_csv('../data/cases_2021_test.csv')
test_df = test_df.dropna(subset=['age'])

# fixing different formats in age column:

#reduce formats with "-"
test_age_list = test_df[test_df['age'].str.contains("-")].values.tolist()

test_age_ranges = []
k = 0
for range in test_age_list:
    test_age_ranges.append(test_age_list[k][0])
    k = k+1


test_age_dash = test_df[test_df['age'].str.contains("-")].index

test_age_values = []

for age in test_age_ranges:
    
    (x,y) = age.split("-")
    if x == '':
        x = 0
    if y == '':
        y = 0
    x = int(x)
    y = int(y)
    value = round((x+y)/2)
    
    test_age_values.append(value)

i = 0
for entry in test_age_dash:
    test_df.loc[entry, 'age'] = test_age_values[i]
    i = i+1


#reduce float formats
test_age_col = test_df['age']


for age in test_age_col:
    if isfloat(age) == True:
        num = age.split(".")
        row = test_df[test_df['age'] == age].index
        for entry in row:
            test_df.loc[entry, 'age'] = num[0]



test_df.drop(test_df[test_df['sex'].isnull()].index.tolist(),inplace=True)

test_prov_null = test_df[test_df['province'].isnull()].index.tolist()
for i in test_prov_null:

    test_new_df = test_df[(test_df['country']== test_df.loc[i][3]) 
    & (test_df['province'].notnull())].index.tolist()

    for k in test_new_df:
        if test_df.loc[k][3] ==  test_df.loc[i][3]:
            lat_value = int(test_df.loc[k][4]) - int(test_df.loc[i][4])
            long_value = int(test_df.loc[k][5]) - int(test_df.loc[i][5])
            if((abs(lat_value) <=1 or abs(long_value) >= 1) or (abs(lat_value) >=1 or abs(long_value) <= 1 )):
                test_df.loc[i,'province'] = test_df.loc[k][2]
                break

test_df['province'].fillna('Unknown',inplace=True)
test_df['country'].fillna('Unknown',inplace=True)
test_df['date_confirmation'].fillna('Unknown',inplace=True)
test_df['additional_information'].fillna('Unknown',inplace=True)
test_df['source'].fillna('Unknown',inplace=True)


#location data
location_df = pd.read_csv('../data/location_2021.csv')

#imputing missing values
location_df['Province_State'].fillna('Unknown',inplace=True)
location_df['Lat'].fillna(0,inplace=True)
location_df['Long_'].fillna(0,inplace=True)
location_df['Recovered'].fillna(0,inplace=True)
location_df['Active'].fillna(0,inplace=True)
location_df['Incident_Rate'].fillna(0,inplace=True)
location_df['Case_Fatality_Ratio'].fillna(0,inplace=True)




#1.5

# print("Searching for outliers in the train dataset.....")
# print()


train_age_Q1, train_age_Q3 = np.percentile(train_df['age'].astype(int),[25,75])


train_age_IQR = train_age_Q3 - train_age_Q1
train_age_upper =(train_age_Q3+1.5*train_age_IQR)
train_age_lower = (train_age_Q1-1.5*train_age_IQR)

train_age_outliers = []

for age in train_df['age'].astype(int):
    if age > train_age_upper:
        train_age_outliers.append(age)
    if age < train_age_lower:
        train_age_outliers.append(age)

# print("Outliers in age: " , train_age_outliers)
# print()


train_lat_Q1, train_lat_Q3 = np.percentile(train_df['latitude'].astype(int),[25,75])

train_lat_IQR = train_lat_Q3 - train_lat_Q1

train_lat_upper =(train_lat_Q3+1.5*train_lat_IQR)
train_lat_lower = (train_lat_Q1-1.5*train_lat_IQR)

train_lat_outliers = []

for lat in train_df['latitude'].astype(int):
    if lat > train_lat_upper:
        train_lat_outliers.append(lat)
    if lat < train_lat_lower:
        train_lat_outliers.append(lat)

# print("Outliers in latitude: " , train_lat_outliers)
# print()

train_long_Q1, train_long_Q3 = np.percentile(train_df['longitude'].astype(int),[25,75])


train_long_IQR = train_long_Q3 - train_long_Q1


train_long_upper =(train_long_Q3+1.5*train_long_IQR)
train_long_lower = (train_long_Q1-1.5*train_long_IQR)

train_long_outliers = []

for long in train_df['longitude'].astype(int):
    if long > train_long_upper:
        train_long_outliers.append(long)
    if long < train_long_lower:
        train_long_outliers.append(long)

# print("Outliers in longitude: " , train_long_outliers)
# print()


# print("Searching for outliers in the test dataset.....")
# print()

test_age_Q1, test_age_Q3 = np.percentile(test_df['age'].astype(int),[25,75])


test_age_IQR = test_age_Q3 - test_age_Q1


test_age_upper =(test_age_Q3+1.5*test_age_IQR)
test_age_lower = (test_age_Q1-1.5*test_age_IQR)

test_age_outliers = []

for age in test_df['age'].astype(int):
    if age > test_age_upper:
        test_age_outliers.append(age)
    if age < test_age_lower:
        test_age_outliers.append(age)

# print("Outliers in age: " , test_age_outliers)
# print()


test_lat_Q1, test_lat_Q3 = np.percentile(test_df['latitude'].astype(int),[25,75])


test_lat_IQR = test_lat_Q3 - test_lat_Q1


test_lat_upper =(test_lat_Q3+1.5*test_lat_IQR)
test_lat_lower = (test_lat_Q1-1.5*test_lat_IQR)


test_lat_outliers = []

for lat in test_df['latitude'].astype(int):
    if lat > test_lat_upper:
        test_lat_outliers.append(lat)
    if lat < test_lat_lower:
        test_lat_outliers.append(lat)

# print("Outliers in latitude: " , test_lat_outliers)
# print()

test_long_Q1, test_long_Q3 = np.percentile(test_df['longitude'].astype(int),[25,75])


test_long_IQR = test_long_Q3 - test_long_Q1


test_long_upper =(test_long_Q3+1.5*test_long_IQR)
test_long_lower = (test_long_Q1-1.5*test_long_IQR)

test_long_outliers = []

for long in test_df['longitude'].astype(int):
    if long > test_long_upper:
        test_long_outliers.append(long)
    if long < test_long_lower:
       test_long_outliers.append(long)

# print("Outliers in longitude: " , test_long_outliers)
# print()


# print("Searching for outliers in the locations dataset.....")
# print()

location_confirmed_Q1, location_confirmed_Q3 = np.percentile(location_df.groupby('Country_Region')['Confirmed'].sum(),[25,75])

location_confirmed_IQR = location_confirmed_Q3 - location_confirmed_Q1


location_confirmed_upper =(location_confirmed_Q3+1.5*location_confirmed_IQR)
location_confirmed_lower = (location_confirmed_Q1-1.5*location_confirmed_IQR)


location_confirmed_outliers = []

for confirm in location_df.groupby('Country_Region')['Confirmed'].sum():
    if confirm > location_confirmed_upper:
        location_confirmed_outliers.append(confirm)
    if confirm < location_confirmed_lower:
       location_confirmed_outliers.append(confirm)

# print("Outliers in confirmed cases: " , location_confirmed_outliers)
# print()



location_deaths_Q1, location_deaths_Q3 = np.percentile(location_df.groupby('Country_Region')['Deaths'].sum(),[25,75])

location_deaths_IQR = location_deaths_Q3 - location_deaths_Q1


location_deaths_upper =(location_deaths_Q3+1.5*location_deaths_IQR)
location_deaths_lower = (location_deaths_Q1-1.5*location_deaths_IQR)

location_deaths_outliers = []

for deaths in location_df.groupby('Country_Region')['Deaths'].sum():
    if deaths > location_deaths_upper:
        location_deaths_outliers.append(deaths)
    if deaths < location_deaths_lower:
       location_deaths_outliers.append(deaths)

# print("Outliers in deaths: " , location_deaths_outliers)
# print()


location_recovered_Q1, location_recovered_Q3 = np.percentile(location_df.groupby('Country_Region')['Recovered'].sum(),[25,75])

location_recovered_IQR = location_recovered_Q3 - location_recovered_Q1


location_recovered_upper =(location_recovered_Q3+1.5*location_recovered_IQR)
location_recovered_lower = (location_recovered_Q1-1.5*location_recovered_IQR)


location_recovered_outliers = []

for recover in location_df.groupby('Country_Region')['Recovered'].sum():
    if recover > location_recovered_upper:
        location_recovered_outliers.append(recover)
    if recover < location_recovered_lower:
       location_recovered_outliers.append(recover)

# print("Outliers in recovered cases: " , location_recovered_outliers)
# print()



location_active_Q1, location_active_Q3 = np.percentile(location_df.groupby('Country_Region')['Active'].sum(),[25,75])

location_active_IQR = location_active_Q3 - location_active_Q1


location_active_upper =(location_active_Q3+1.5*location_active_IQR)
location_active_lower = (location_active_Q1-1.5*location_active_IQR)


location_active_outliers = []

for active in location_df.groupby('Country_Region')['Active'].sum():
    if active > location_active_upper:
        location_active_outliers.append(active)
    if active < location_active_lower:
       location_active_outliers.append(active)

# print("Outliers in active cases: " , location_active_outliers)
# print()




location_incident_Q1, location_incident_Q3 = np.percentile(location_df.groupby('Country_Region')['Incident_Rate'].sum(),[25,75])


location_incident_IQR = location_incident_Q3 - location_incident_Q1


location_incident_upper =(location_incident_Q3+1.5*location_incident_IQR)
location_incident_lower = (location_incident_Q1-1.5*location_incident_IQR)


location_incident_outliers = []

for incident in location_df.groupby('Country_Region')['Incident_Rate'].sum():
    if incident > location_incident_upper:
        location_incident_outliers.append(incident)
    if incident < location_incident_lower:
       location_incident_outliers.append(incident)

# print("Outliers in incident rates: " , location_incident_outliers)
# print()


location_case_Q1, location_case_Q3 = np.percentile(location_df.groupby('Country_Region')['Case_Fatality_Ratio'].mean(),[25,75])


location_case_IQR = location_case_Q3 - location_case_Q1


location_case_upper =(location_case_Q3+1.5*location_case_IQR)
location_case_lower = (location_case_Q1-1.5*location_case_IQR)

location_case_outliers = []

for case in location_df.groupby('Country_Region')['Case_Fatality_Ratio'].mean():
    if case > location_case_upper:
        location_case_outliers.append(case)
    if case < location_case_lower:
       location_case_outliers.append(case)

# print("Outliers in case fatality ratio: " , location_case_outliers)

#1.6 Joining the cases and location dataset
#train_df = pd.read_csv("../data/cases_2021_train.csv")
#test_df = pd.read_csv("../data/cases_2021_test.csv")
#location_df = pd.read_csv("../data/location_2021.csv")

#make names consistent between datasets
location_df['Country_Region'].replace({'US': 'United States', 'Korea, South': 'South Korea', 'Taiwan*': 'Taiwan'}, inplace=True)

#rename locations columns
location_df = location_df.rename(columns={'Province_State': 'province', 'Country_Region': 'country'})

#aggregate over same province, country and take average values
location_df = location_df.groupby(['province','country']).aggregate('mean')

#join on province and country
cases_train =  pd.merge(train_df, location_df, how='inner', on=['province','country'])
cases_test = pd.merge(test_df, location_df, how='inner', on=['province','country'])

#output csv
cases_train.to_csv("../results/cases_2021_train_processed.csv")
cases_test.to_csv("../results/cases_2021_test_processed.csv")
location_df.to_csv("../results/location_2021_processed.csv")

#number of rows
print('Train Number of Cases: ', cases_train.shape[0])
print('Test Number of Cases: ', cases_test.shape[0])

#1.7 Feature selection

#y = cases_train['outcome'].to_numpy()
#X = cases_train.copy()
#X.drop(columns=['outcome','outcome_group'])
#X = X.to_numpy()
#score = fisher_score.fisher_score(X, y, mode='rank') 
#print(score)

train_features = cases_train[['age', 'sex', 'chronic_disease_binary', 'Confirmed', 'Deaths', 'Recovered',
                              'Active', 'Incident_Rate', 'Case_Fatality_Ratio']]
test_features = cases_test[['age', 'sex', 'chronic_disease_binary', 'Confirmed', 'Deaths', 'Recovered',
                            'Active', 'Incident_Rate', 'Case_Fatality_Ratio']]

train_features.to_csv("../results/cases_2021_train_processed_features.csv")
test_features.to_csv("../results/cases_2021_test_processed_features.csv")


    
