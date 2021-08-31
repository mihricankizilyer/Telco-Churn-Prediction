################################################
# TELCO CHURN PREDICTION
################################################

#############################
# IMPORT LIBRARIES & MODULES
#############################

######## REQUIRED INSTALL ########

!pip install geopy
!pip install yellowbrick
!pip install yellowbrick
!pip install catboost
!pip install xgboost
!pip install lightgbm
!pip install gbm

# Geocode Libraries
from geopy.geocoders import Nominatim

# Since the timeout decreases when the data set is large, we added:
from geopy.exc import GeocoderTimedOut

############ Import Libraries ###################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math as mt
import missingno as msno

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
import yellowbrick
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configurations:
pd.set_option('display.max_columns', 1500)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1500)

# Modelling Libraries:

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier , AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, cross_val_score,  cross_validate, GridSearchCV , validation_curve, RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score,roc_auc_score, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=ConvergenceWarning)

# Configurations:

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_rows', 500)

# EDA Modules:

from helpers.data_prep import *
from helpers.eda import *


# There are empty spaces in the field from the Total Charges table, we will use the services table for this variable:

df_5.drop(columns="Total Charges", axis=0, inplace=True)
df_5.rename(columns={"Total Charges_drop": "Total Charges"}, inplace=True)

df_5.shape

# Let's delete the second variable that is in more than one dataset:
df_5.drop([col for col in df_5.columns if '_drop' in col], axis=1, inplace=True)
df = df_5.copy()

def Create_data_label(ax):
    """ Display data label for given axis """
    for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/ 2
                    , bar.get_height() + 0.01
                    , str(round(100 * bar.get_height(),2)) + '%'
                    , ha = 'center'
                    , fontsize = 13)


def Categorical_var_churn_dist(data, cols, distribution_col):
    """ Distribution of categorical variable based on target variable """

    for i,feature in enumerate(cols):

        feature_summary = data[feature].value_counts(normalize=True).reset_index(name='Percentage')
        # cat_cols = grab_col_names(df)[1]
        plt_cat = sns.catplot(x=feature
                , y='Percentage'
                , data = feature_summary
                , col=distribution_col
                , kind='bar'
                , aspect = 0.8
                , alpha = 0.6)

        if feature == 'PaymentMethod':
            plt_cat.set_xticklabels(rotation= 65, horizontalalignment = 'right')


        for ax1, ax2 in plt_cat.axes:
            Create_data_label(ax1)
            Create_data_label(ax2)


        plt.ylim(top=1)
        plt.subplots_adjust(top = 0.9)
        plt.gcf().suptitle(feature+" distribution",fontsize=14)
    plt.show()

churn_summary = df.groupby('Churn Value')
cat_cols = grab_col_names(df)[1]
Categorical_var_churn_dist(churn_summary, cat_cols,'Churn Value')

# We have only examined this field, but this information may have been obtained after the customer left, so we will not use it as an input:
df.groupby("Churn Reason").agg({"Customer ID": "count"}).sort_values("Customer ID").head()

# Both are exactly the same, we can delete one
#df["Tenure Months"] - df["Tenure in Months"].sort_values(ascending=False)
df["Tenure Months"] - df["Tenure in Months"].sort_values(ascending=True)

df.shape

# Let's examine the customer status:
df.groupby(["Churn Label", "Churn Value", "Customer Status"]).agg({"Tenure Months": ["count", "max"]})

#***********************Variables in more than one dataset **********************************#

# Churn Category was associated with the churn reason in the status table, so we're using the one in the churn table so let's remove it:
df.drop(columns=["Device Protection","Count", "Tenure in Months", "Churn Category","Tech Support","Churn Reason","Churn Label","CLTV","Quarter","ID"], axis=1, inplace=True)

df.info()

check_df(df)

location_cols, cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in location_cols:
    print(col)

##############################
# Outlier Analysis
##############################

for col in num_cols:
    print(col, ":", check_outlier(df, col,q1=0.05,q3=0.95))

sns.distplot(df["Number of Dependents"], kde=True, bins=10)
plt.show();

##############################
# LOCAL OUTLIER FACTOR (LOF)
##############################

exclude_cols = ["Churn Value","Churn Score"]
num_cols = [col for col in num_cols if  col  not in exclude_cols]

df_ = df[num_cols]
df_.head(3)

clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(df_)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

##############################
# Visualization
##############################

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 12], style='.-')
plt.show()

# Observations under the Threshold, that is, those with outlier values:

th = - 1.87
# df[df_scores < th]

df.drop(df[df_scores < th].index, inplace=True)
df.shape

##############################
# Missing Values
##############################

df.isnull().sum()

df.isnull().sum().sum()

#########################################
# FEATURE ENGINEERING
#########################################

# Let's examine the class numbers in detail:

df["Country"].unique()

df["State"].unique()

df["City"].unique()

# Generating variables from Longitude and Latitude values:
df[["Longitude", "Latitude","City"]].head(5)

# Make a Nominatim object and initialize Nominatim API  with the geoapiExercises parameter
# initialize Nominatim API

# https://www.geeksforgeeks.org/get-the-city-state-and-country-names-from-latitude-and-longitude-using-python/

geolocator = Nominatim(user_agent="geoapiExercises")

# Latitude & Longitude input
Latitude = "33.978030"
Longitude = "-118.217141"

location = geolocator.reverse(Latitude+","+Longitude)

# Display
print(location)

address = location.raw['address']
print(address)

# Latitude & Longitude input
Latitude = "34.224377"
Longitude = "-118.632656"

location = geolocator.reverse(Latitude+","+Longitude)

# Display
print(location)

address = location.raw['address']
print(address)

# Latitude & Longitude input
Latitude = "33.964131"
Longitude = "-118.272783"

location = geolocator.reverse(Latitude+","+Longitude)

# Display
print(location)


address = location.raw['address']
print(address)

road = address.get('road', '')
city = address.get('city', '')
state = address.get('state', '')
country = address.get('country', '')
print('Road : ',road)
print('City : ',city)
print('State : ',state)
print('Country : ',country)

# Latitude and Longtitude will be concat:

df['New_Location_Coordinate'] = df.apply(lambda x: str(x['Latitude']) + ","+ str(x['Longitude']),axis = 1)
df['New_Location_Coordinate'].head(2)

# Let's work by deduplicating this field:

lc_unique =  pd.DataFrame({"LC":df["New_Location_Coordinate"].unique()} )
lc_unique.shape

lc_unique.head()

# Let's change it to address form and convert it to dictionary form:

lc_unique['New_Location'] =  lc_unique.apply(lambda x: geolocator.reverse(x['LC'], timeout=10000000),axis = 1)

lc_unique['New_Location_Address'] =  lc_unique.apply(lambda x: x['New_Location'].raw['address'],axis = 1)

lc_unique[["LC","New_Location"]].head()


# Let's backup:

lc_unique_ = lc_unique.copy()

lc_unique['New_Location_Address'] =  lc_unique.apply(lambda x: x['New_Location'].raw['address'],axis = 1)

lc_unique.head()


# Let's import the data into csv:

lc_unique.to_csv("location_Address.csv")

lc_unique["New_Address_Aeroway"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('aeroway', ''),axis = 1)
lc_unique["New_Address_Highway"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('highway', ''),axis = 1)
lc_unique["New_Address_Road"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('road', ''),axis = 1)

lc_unique["New_Address_Amenity"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('amenity', ''),axis = 1)
lc_unique["New_Address_Hamlet"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('hamlet', ''),axis = 1)

lc_unique["New_Address_Village"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('village', ''),axis = 1)
lc_unique["New_Address_Town"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('town', ''),axis = 1)
lc_unique["New_Address_Suburb"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('suburb', ''),axis = 1)

lc_unique["New_Address_Residential"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('residential', ''),axis = 1)
lc_unique["New_Address_Neighbourhood"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('neighbourhood', ''),axis = 1)
lc_unique["New_Address_Building"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('building', ''),axis = 1)

lc_unique["New_Address_Shop"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('shop', ''),axis = 1)
lc_unique["New_Address_Tourism"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('tourism', ''),axis = 1)
lc_unique["New_Address_Leisure"] = lc_unique.apply(lambda x:  x['New_Location_Address'].get('leisure', ''),axis = 1)

lc_unique.head()

lc_unique.columns
lc_unique.rename(columns={'LC':'New_Location_Coordinate'},inplace=True) # join için keyleri aynı forma dönüştürdük

df_final = df.merge(lc_unique, on='New_Location_Coordinate', how='left', suffixes=('', '_drop'))

# Let's see which columns we just added:

new_list = [ col for col in df_final.columns if "New_" in col and "New_Location" not in col]
new_list

# New Features From New Location Features:

df_final["New_Is_Aeroway"] = np.where(df_final["New_Address_Aeroway"]=="",0,1)
df_final["New_Is_Amenity_Tourism"] = np.where( ( (df_final["New_Address_Amenity"] =="") & (df_final["New_Address_Tourism"] == "")), 0, 1)

df_final["New_Address_Is_Shop"] = np.where(df_final["New_Address_Shop"]=="",0,1)
df_final["New_Address_Is_Neighbourhood"] = np.where(df_final["New_Address_Neighbourhood"]=="",0,1)
df_final["New_Address_Is_Highway"] = np.where(df_final["New_Address_Highway"]=="",0,1)

# All of these settlement types are similar, let's combine them into villages:

df_final["New_Is_Village"] = np.where(  ( ( df_final["New_Address_Hamlet"] =="")   &
                                        (df_final["New_Address_Village"] =="") &
                                        (df_final["New_Address_Town"] == "") &
                                        (df_final["New_Address_Suburb"] =="")
                                        ), 0, 1)

df_final["New_Region"] = np.where(df_final["New_Address_Village"] !="",df_final["New_Address_Village"],np.NaN)
df_final["New_Region"] = np.where(df_final["New_Address_Town"] !="",df_final["New_Address_Town"],df_final["New_Region"])
df_final["New_Region"] = np.where(df_final["New_Address_Suburb"] !="",df_final["New_Address_Suburb"],df_final["New_Region"])
df_final["New_Region"] = np.where(df_final["New_Address_Hamlet"] !="",df_final["New_Address_Hamlet"],df_final["New_Region"])
df_final["New_Region"] = np.where(df_final["New_Address_Road"] !="",df_final["New_Address_Road"],df_final["New_Region"])
df_final["New_Region"] = np.where(df_final["New_Address_Neighbourhood"] !="",df_final["New_Address_Neighbourhood"],df_final["New_Region"])
df_final["New_Region"] = np.where(df_final["New_Address_Residential"] !="",df_final["New_Address_Residential"],df_final["New_Region"])

df_final["New_Region"].head(20)

df_final["Senior Citizen"].replace(to_replace ="No", value ="Young", inplace = True)
df_final["Senior Citizen"].replace(to_replace ="Yes", value ="Senior", inplace = True)

## Additional Services
df_final["Online Security"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Online Security"].replace(to_replace =["No internet service","No"], value =0, inplace = True)

df_final["Online Backup"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Online Backup"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["Phone Service"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Phone Service"].replace(to_replace ="No", value = 0, inplace = True)

df_final["Device Protection Plan"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Device Protection Plan"].replace(to_replace ="No", value = 0, inplace = True)

df_final["Premium Tech Support"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Premium Tech Support"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["Streaming TV"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Streaming TV"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["Streaming Movies"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Streaming Movies"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["Unlimited Data"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Unlimited Data"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["Streaming Music"].replace(to_replace ="Yes", value =1, inplace = True)
df_final["Streaming Music"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)

df_final["New_Internet_Service"] = df_final["Internet Service"]
df_final["New_Internet_Service"].replace(to_replace =["No internet service","No"], value = 0, inplace = True)
df_final["New_Internet_Service"].replace(to_replace = ["DSL","Fiber optic"], value = 1, inplace = True)

# How many services does it receive?

df_final["New_Additional_Services"] =  df_final["Phone Service"] +    df_final["New_Internet_Service"] + df_final["Online Security"] +  df_final["Online Backup"] +  df_final["Streaming TV"] +  df_final["Streaming Movies"] +  df_final["Streaming Music"] +   df_final["Device Protection Plan"] +  df_final["Premium Tech Support"] +   df_final["Unlimited Data"]

df_final["New_Entertainment_Lovers"] = np.where(df_final["Streaming TV"] + df_final["Streaming Movies"] + df_final["Streaming Music"] >0,1,0)

df_final["New_Technology_Lovers"] = np.where(df_final["Online Security"] + df_final["Online Backup"]  + df_final["Device Protection Plan"] + df_final["Premium Tech Support"] >0,1,0)
df_final["New_Guaranteed_Customers"] = np.where( df_final["Device Protection Plan"] + df_final["Premium Tech Support"] >0,1,0)

sns.countplot(y ='Churn Value',hue= df_final["New_Entertainment_Lovers"], data = df_final)
plt.show();

sns.countplot(y ='Churn Value',hue= df_final["New_Entertainment_Lovers"], data = df_final)
plt.show();

df.info()

df_final["New_Total_Extra_Data_Charges_Ratio_InRevenue"] = df_final["Total Extra Data Charges"] / ( df_final["Total Revenue"] - df_final ["Total Refunds"])
df_final["New_Total_Extra_Data_Charges_Ratio_InTotalBill"] = df_final["Total Extra Data Charges"] / df_final["Total Charges"]
df_final["New_Total_Long_Distance_Charges_Ratio_InRevenue"] = df_final["Total Long Distance Charges"] / ( df_final["Total Revenue"] - df_final ["Total Refunds"])
df_final["New_Total_Long_Distance_Ratio_InTotalBill"] = df_final["Total Long Distance Charges"] / df_final["Total Charges"]
df_final["New_Total_Refund_Ratio_InRevenue"] = df_final["Total Refunds"] *100 / df_final["Total Revenue"]
df_final["New_Total_Refund_Ratio_TotalExtraCharges"] = df_final["Total Refunds"] *100 / (df_final["Total Extra Data Charges"] + df_final["Total Long Distance Charges"])
df_final["New_Total_Refund_Ratio_InTotalBill"] = df_final["Total Refunds"] *100 /  df_final["Total Charges"]

# Avg Monthly GB Download
df_final.loc[(df_final["Avg Monthly GB Download"] < 5), "New_Avg_Monthly_GB_Download"] = "low"
df_final.loc[(df_final["Avg Monthly GB Download"] >= 5) & (df_final["Avg Monthly GB Download"]< 20), "New_Avg_Monthly_GB_Download"] = "medium"
df_final.loc[(df_final["Avg Monthly GB Download"]>= 20) & (df_final["Avg Monthly GB Download"]< 50), "New_Avg_Monthly_GB_Download"] = "good"
df_final.loc[(df_final["Avg Monthly GB Download"]>= 50), "New_Avg_Monthly_GB_Download"] = "plus"


# Population
df_final.loc[(df_final["Population"] < 1000), "New_Population"] = "village"
df_final.loc[(df_final["Population"]>= 1000) & (df_final["Population"] < 8000), "New_Population"] = "town"
df_final.loc[(df_final["Population"]>= 8000) & (df_final["Population"] < 20000), "New_Population"] = "district"
df_final.loc[(df_final["Population"]>= 20000) & (df_final["Population"] < 70000), "New_Population"] = "province"
df_final.loc[(df_final["Population"]>= 70000), "New_Population"] = "metropolis"


# Standardizing client age duration by age, given that client age is a 'function' of one's age
df_final["New_Tenure_Months_Age"] = df_final["Tenure Months"] / df_final["Age"]

# Tenure Months
df_final.loc[(df_final["Tenure Months"] <=3), "New_Tenure_Cat"] = "New_Customer"
df_final.loc[(df_final["Tenure Months"]<= 6) & (df_final["Tenure Months"] >3), "New_Tenure_Cat"] = "Onboarding"
df_final.loc[(df_final["Tenure Months"]<= 36) & (df_final["Tenure Months"] > 6), "New_Tenure_Cat"] = "Old_Customer"
df_final.loc[df_final["Tenure Months"] > 36, "New_Tenure_Cat"] = "Loyal_Customer"


# sns.boxplot(y='New_Tenure_Months_Age',x = 'Churn Value', hue = 'Churn Value',data = df_final)
# plt.ylim(-1, 1)
# plt.show()

df_final.head(2)

# df_final.groupby("Senior Citizen").agg({"New_Entertainment_Lovers":"mean"})
df_final.groupby("Senior Citizen").agg({"New_Technology_Lovers":"mean"})

df_final["New_Senior_Entertainment_Lovers"] = np.where( (df_final["Senior Citizen"] =="Senior") & (df_final["New_Entertainment_Lovers"] ==1),"Crazy_Seniors" ,"Normal")
df_final["New_Senior_Tech_Lovers"] = np.where( (df_final["Senior Citizen"] =="Senior") & (df_final["New_Technology_Lovers"] ==1),"Tech_Followers_Seniors" ,"Normal")

df_final['New_Total_Revenue_Cat'] = pd.qcut(x=df_final['Total Revenue'], q=5, labels=[1,2,3,4,5])

#df_final.groupby("New_Total_Revenue_Cat").agg({"Total Revenue": ["count","mean"]})

# Alone / Not Alone:

df_final['New_NumofDependents_Cat'] = pd.qcut(x=df_final['Number of Dependents'].rank(method="first"), q=2, labels=[1,2])
# df_final.groupby("New_NumofDependents_Cat").agg({"Churn Value": ["count","mean"]})

df_final['New_Monthly_Charge_Cat'] = pd.qcut(x=df_final['Monthly Charge'], q=3, labels=[1,2,3])

df_final['New_Total_Monthly_Charge_Cat'] = pd.qcut(x=df_final['Total Charges'], q=3, labels=[1,2,3])

df_final["New_Tenure_Contract_Relation"] = np.where(df_final["Contract"] == "Month-to-month", df_final["Tenure Months"]/1, np.NaN)
df_final["New_Tenure_Contract_Relation"] = np.where(df_final["Contract"] == "One year", df_final["Tenure Months"]/12, df_final["New_Tenure_Contract_Relation"])
df_final["New_Tenure_Contract_Relation"] = np.where(df_final["Contract"] =='Two year' , df_final["Tenure Months"]/24, df_final["New_Tenure_Contract_Relation"])

df_final["New_HasChild"] = np.where( ((df_final["Married"] =='Yes')  & (df_final["Dependents"] =='Yes') ) , 1, 0)

df_final["New_AvgRevenue_Per_Services"] =  (df_final["Total Revenue"]  - df_final["Total Refunds"]  ) /df_final["New_Additional_Services"]

df_final["New_IsAlone"] = np.where( ((df_final["Dependents"] =='Yes')|(df_final["Partner"] =='Yes') ) , 1, 0)

df_final['New_NumberofReferrals'] = pd.qcut(x=df_final['Number of Referrals'].rank(method="first"), q=3, labels=[1,2,3])


df_final["New_Avg_MonthlyLongDistanceChar_Ratio"] = df_final["Avg Monthly Long Distance Charges"]/ df_final["Monthly Charges"]

#df_final.groupby("New_Total_Monthly_Charge_Cat").agg({"Churn Value": ["count","mean"], "Total Charges": ["min","max","mean"]})

df_final.head(2)

# We derive field for New Customer, we can delete Customer Status:
df_final.drop(columns="Customer Status", inplace=True,axis = 1)

df_final.loc[df_final["Tenure Months"] <=3,  "New_Tenure_Cat"] = "New_Customer"
df_final.loc[(df_final["Tenure Months"]<= 6) & (df_final["Tenure Months"] >3), "New_Tenure_Cat"] = "Onboarding"
df_final.loc[(df_final["Tenure Months"]<= 6) & (df_final["Tenure Months"] > 6), "New_Tenure_Cat"] = "Old_Customer"
df_final.loc[df_final["Tenure Months"] > 36, "New_Tenure_Cat"] = "Loyal_Customer"

df_final['New_Has_Promotion_Or_Complaint'] = np.where(df_final['Total Refunds'] >0 ,1,0)

# Has_Complaint

df_final["New_Has_Complaint"] = np.where( ( (df_final["New_Has_Promotion_Or_Complaint"] ==1 ) & (df_final["Satisfaction Score"] <=2) ), 1,0)

# Has_Promotion

df_final["New_Has_Promotion"] = np.where( ( (df_final["New_Has_Promotion_Or_Complaint"] ==1 ) & (df_final["Satisfaction Score"] >=4) ), 1,0)

# df_final.groupby(["New_Has_Promotion_Or_Complaint"]).agg({"Satisfaction Score":["mean","count"],   "Churn Value":"mean"})

# Let's break down electronic payments:
df_final["New_Payment_Method_Automatic"] = np.where( (df_final["Payment Method"] == 'Bank transfer (automatic)') | (df_final["Payment Method"] ==  'Credit card (automatic)'), 1,0)

df_final.loc[(df_final['Gender'] == 'Male') & (df_final['Age'] <= 21), 'New_Sex_Age_Cat'] = 'Youngmale'
df_final.loc[(df_final['Gender'] == 'Male') & ((df_final['Age'] > 21) & (df_final['Age']) < 50), 'New_Sex_Age_Cat'] = 'Maturemale'
df_final.loc[(df_final['Gender'] == 'Male') & (df_final['Age'] > 50), 'New_Sex_Age_Cat'] = 'Seniormale'
df_final.loc[(df_final['Gender'] == 'Female') & (df_final['Age'] <= 21), 'New_Sex_Age_Cat'] = 'Youngfemale'
df_final.loc[(df_final['Gender'] == 'Female') & ((df_final['Age'] > 21) & (df_final['Age']) < 50), 'New_Sex_Age_Cat'] = 'Maturefemale'
df_final.loc[(df_final['Gender'] == 'Female') & (df_final['Age'] > 50), 'New_Sex_Age_Cat'] = 'Seniorfemale'

df_final.info()

[col for col in df_final.columns if "New_" in col]

#########################################
# Bölge Bazlı Clustering (K-Means)
#########################################

df_location_summary = df_final.groupby("New_City_NewRegion").agg({"Population":"sum",
                                                                  "Satisfaction Score":"mean"})

df_location_summary.reset_index(inplace=True)
df_location_summary = df_location_summary[["Population", "Satisfaction Score"]]


df_location_summary.rename(columns={'Population':'Total_Population',
                         'Satisfaction Score':'Satisfaction_Score'},inplace=True)

df_location_summary.head()

df_location_summary = df_location_summary[["Total_Population", "Satisfaction_Score"]]

sc = MinMaxScaler((0, 1))
df_location_summary = sc.fit_transform(df_location_summary)
kmeans = KMeans()
k_fit = kmeans.fit(df_location_summary)

k_fit.get_params()
k_fit.labels_
k_fit.inertia_

!pip freeze

# Visualization of Clusters
from yellowbrick.cluster import KElbowVisualizer

# returns numpy array with fit operation, let's convert it to dataframe:
df_location_summary = pd.DataFrame(df_location_summary)

k_means = KMeans(n_clusters=4).fit(df_location_summary)
kumeler = k_means.labels_

# We chose 2 variables (0.index and 1.index) to visualize, that is, reduce to 2 dimensions.

# marking of centers
merkezler = k_means.cluster_centers_

plt.scatter(df_location_summary.iloc[:, 0],
            df_location_summary.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Determination of Optimum Number of Clusters

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_location_summary)
    ssd.append(kmeans.inertia_)



plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Against Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))

#2 Let's visualize for the number of clusters from -20:
visu = KElbowVisualizer(kmeans, k=(2,20))
visu.fit(df_location_summary)       # Fit the data to the visualizer
visu.poof()                         # Finalize and render the figure
visu.show()

# Optimum k value:

visu.elbow_value_

 # optimum k value: 6

# Let's fit and assign according to the optimum value:
kmeans = KMeans(n_clusters= elbow.elbow_value_).fit(df_location_summary)
kumeler = kmeans.labels_
df_location_summary["cluster_no"] = kumeler

# We chose 2 variables (0.index and 1.index) to visualize, that is, reduce to 2 dimensions.

# marking of centers
merkezler = k_means.cluster_centers_

plt.scatter(df_location_summary.iloc[:, 0],
            df_location_summary.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

# Added +1 to start from 1 instead of 0:
df_location_summary["cluster_no"] = df_location_summary["cluster_no"] + 1

df_location_summary["cluster_no"].max()

# Before the segment assignment, let's read the same data set again and add the clusters:
df_location_summary = df_final[["New_City_NewRegion","Population","Satisfaction Score"]]

df_location_summary = df_location_summary[["Population", "Satisfaction Score"]]

df_location_summary.rename(columns={'Population':'Total_Population',
                         'Satisfaction Score':'Satisfaction_Score'},inplace=True)

# index with inner join for left_index/right_index

df_location =  df_location_summary.merge(df_location, left_index=True, right_index=True)

df_location.head()

# Now let's delete unnecessary columns (0-1 : scaled ones)

df_location.drop(columns= [0, 1],axis=1, inplace=True)

df_location.head()

df_location = df_location[["New_City_NewRegion","cluster_no"]]

df_location.head()

df_final_copy = df_final.copy()

# City-New_Region must be created first to add it to our main dataset

df_final["New_City_NewRegion"] = df_final["City"] + '-'+ df_final["New_Region"]

# df_final.drop(columns="New_Region_ClusterNo",axis=1,inplace=True)

# Let's add the clusters to our df_final dataset:
df_final = df_final.merge(df_location, on ="New_City_NewRegion",how='left')
df_final.rename(columns={'cluster_no':'New_Region_ClusterNo'},inplace=True)

# df_final.groupby("New_Region_ClusterNo").agg({"Customer ID":"count"})

df_final.drop(["New_City_NewRegion"],axis=1, inplace=True)
df_final.head(1)

# cluster no: let's assign
# Cluster_no 'str' has been made against the error while replacing.
df_final['New_Region_ClusterNo'] = df_final['New_Region_ClusterNo'].astype('str')


df_final["New_Region_ClusterNo"] = df_final["New_Region_ClusterNo"].replace(to_replace =['1','2','3','4','5','6'],
                                         value = ['Low_Population_Normal_Satisfaction',
                                                  'Normal_Population_High_Satisfaction',
                                                  'High_Population_Low_Satisfaction',
                                                  'Normal_Population_Low_Satisfaction',
                                                  'High_Population_High_Satisfaction',
                                                  'Low_Population_High_Satisfaction'] )

df_final[df_final["New_Region_ClusterNo"] ==2].head(2)

df_final.groupby("New_Region_ClusterNo").agg({"Satisfaction Score": "mean",
                                              "Population": "mean",
                                              "Churn Value":"mean"})

df_final.drop(columns="New_Region_ClusterNo",axis=1,inplace=True)

df_final.shape

################################################
# Tenure + Clustering (K-Means)
################################################

# Clustered by 2 features, Total Revenue, Tenure Months
# Scaled first:
mms = MinMaxScaler()

df_final["Scaled_Tenure_Months"] = mms.fit_transform(df_final[["Tenure Months"]])
df_final["Scaled_Total_Revenue"] = mms.fit_transform(df_final[["Total Revenue"]])

# We worked on a separate dataframe:
df_kmeans = df_final[["Scaled_Total_Revenue","Scaled_Tenure_Months"]]

# Visualization:

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_kmeans)
elbow.show()


elbow.elbow_value_
# k = 6

# Let's fit and assign according to the optimum value:
kmeans = KMeans(n_clusters=6).fit(df_kmeans)
kumeler = kmeans.labels_
df_kmeans["cluster_no"] = kumeler

df_kmeans.head()



df_kmeans["cluster_no"] = df_kmeans["cluster_no"] + 1

# Let's Visualize How the Cluster is parsed:

merkezler = kmeans.cluster_centers_
plt.scatter(df_kmeans.iloc[:, 0],
            df_kmeans.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

df_final.shape

df_final_copy =df_final.copy()

# According to the result, let's combine it with df:
# let's concat:

df_final2_ = pd.concat([df_final, df_kmeans], axis=1)

df_final.shape

df_final2_.head(2)

df_final2_.rename(columns={'cluster_no':'New_Tenure_Revenue_ClusterNo'},inplace=True)

df_final2_.groupby("New_Tenure_Revenue_ClusterNo").agg({"Total Revenue":"mean",
                                                      "Tenure Months":"mean"})

# Let's do cluster based analysis:
data = df_final.groupby("New_Tenure_Revenue_ClusterNo").agg({"Total Revenue":"mean","Tenure Months":"mean"}).reset_index()
data["Average_Revenue"] = df_final["Total Revenue"] / df_final["Tenure Months"]

# cluster no: let's assign
# Cluster_no 'str' has been made against the error while replacing.
df_final2_['New_Tenure_Revenue_ClusterNo'] = df_final2_['New_Tenure_Revenue_ClusterNo'].astype('str')

df_final2_["New_Tenure_Revenue_ClusterNo"] = df_final2_["New_Tenure_Revenue_ClusterNo"].replace(to_replace =['1','2','3','4','5','6'],
                                         value = ['Low_Potential','High_Potential','Champions','New_Customer','Low_Profit','High_Profit'])


df_final.to_csv("df_final.csv")

df_final = pd.read_csv("df_final.csv")

df_final.shape

df_final.drop(columns = ["Scaled_Tenure_Months",
                          "Scaled_Total_Revenue",
                          "Scaled_Total_Revenue.1",
                          "Scaled_Tenure_Months.1",
                          "Unnamed: 0",  "Country",
                          "State",
                          "City",
                          "Zip Code",
                          "Lat Long",
                          "Latitude",
                          "Longitude",
                          'New_Location_Coordinate',
                          'New_Location',
                          'New_Location_Address',
                          'New_Address_Aeroway',
                          'New_Address_Highway',
                          'New_Address_Road',
                          'New_Address_Amenity',
                          'New_Address_Hamlet',
                          'New_Address_Village',
                          'New_Address_Town',
                          'New_Address_Suburb',
                          'New_Address_Residential',
                          'New_Address_Neighbourhood',
                          'New_Address_Building',
                          'New_Address_Shop',
                          'New_Address_Tourism',
                          'New_Address_Leisure',
                          'New_Region'], axis=1, inplace=True)

df_final.head()

# Let's look at Missing Values again:

df_final.isnull().sum()

df_final[df_final["New_Region_ClusterNo"].isnull()>0].agg({"Satisfaction Score": "mean",
                                              "Population": "mean",
                                              "Churn Value":"mean"})

df_final.groupby("New_Region_ClusterNo").agg({"Satisfaction Score": "mean",
                                              "Population": "mean",
                                              "Churn Value":"mean"})

df_final["New_Total_Refund_Ratio_TotalExtraCharges"] = df_final["New_Total_Refund_Ratio_TotalExtraCharges"].fillna(0)

df_final["New_Region_ClusterNo"] = df_final["New_Region_ClusterNo"].fillna("Low_Population_High_Satisfaction")

location_cols, cat_cols, num_cols, cat_but_car = grab_col_names(df_final)

df_final.dtypes

conv_cols = [col for col in df_final.columns if df_final[col].dtype == "uint8"]
df_final[conv_cols] = df_final[conv_cols].astype("int64")

df_final[df_final.isin([np.nan, np.inf, -np.inf]).any(1)].count().sum()

# There are infinite values (inf,-inf), let's fill it with NaN first then 0:

df_final = df_final.replace([np.inf, -np.inf], np.nan)
df_final= df_final.fillna(0)

df_final.head(2)

df_final.groupby("New_Region_ClusterNo").agg({"Churn Value":["mean","count"]})

cat_cols

df_final.groupby(["Dependents","Partner"]).agg({"Churn Value": ["mean","count"]})

df_final.groupby(["Married","Number of Dependents"]).agg({"Customer ID":"count"})

##################################
# Rare Encoding
##################################

cat_cols = [col for col in df_final.columns if df_final[col].dtypes == "O" and col not in location_cols and "Customer ID" not in col]
rare_analyser(df_final, "Churn Value", cat_cols)

cat_cols = [col for col in df_final.columns if df_final[col].dtypes == "O" and "Customer ID" not in col and col not in location_cols]

df_final.groupby("New_Avg_Monthly_GB_Download").agg({"New_Avg_Monthly_GB_Download":"count"})

df_final['New_NumberofReferrals'] = df_final['New_NumberofReferrals'].astype("int")

df_final = rare_encoder(df_final, 0.01, cat_cols)
rare_analyser(df_final, "Churn Value", cat_cols)

useless_cols = [col for col in cat_cols if df_final[col].nunique() == 1 or
                (df_final[col].nunique() == 2 and (df_final[col].value_counts() / len(df_final) <= 0.01).any(axis=None))]

cat_cols = [col for col in cat_cols if col not in useless_cols]

useless_cols

# Useless features should be deleted from cat_cols:
for col in useless_cols:
    df_final.drop(col, axis=1, inplace=True)

df_final.dtypes

df_final["New_Total_Revenue_Cat"] = df_final["New_Total_Revenue_Cat"].astype("O")

########################################
# Label Encoding & One-Hot Encoding
########################################

cat_cols = [col for col in df_final.columns if df_final[col].dtypes == "O"and "Customer ID" not in col and col not in location_cols]

one_hot_encoder(df_final, cat_cols, drop_first=True)

df_final = one_hot_encoder(df_final, cat_cols, drop_first=True)

df_final.head()


cat_cols = [col for col in df_final.columns if df_final[col].dtypes == "O"and "Customer ID" not in col and col not in location_cols]

conv_cols = [col for col in df_final.columns if df_final[col].dtype == "uint8"]
df_final[conv_cols] = df_final[conv_cols].astype("int64")

##Correlation:
# 0.70


def high_correlated_cols(dataframe, target, plot=False, corr_th=0.70, min_periods=1):
    location_cols, cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if target not in col]
    cor_matrix = dataframe[num_cols].corr().abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (35, 35)})
        plt.xlabel('xlabel', fontsize=18)
        sns.heatmap(cor_matrix, cmap="RdBu",annot=True)
        plt.show()
    return drop_list

high_corr_cols = high_correlated_cols(df_final, "Churn Value",plot=True)

high_corr_cols

df_final_copy = df_final.copy()

high_corr_cols=['Monthly Charge',
 'Total Charges',
 'Total Revenue',
 'New_Total_Extra_Data_Charges_Ratio_InTotalBill',
 'New_Total_Long_Distance_Charges_Ratio_InRevenue',
 'New_Total_Long_Distance_Ratio_InTotalBill',
 'New_Total_Refund_Ratio_InTotalBill',
 'New_Tenure_Months_Age',
 'New_Avg_MonthlyLongDistanceChar_Ratio']

df_final.drop(columns=high_corr_cols,axis=1, inplace=True)

#high_correlated_cols(df_final, "Churn Value", plot=False)


df_final.to_csv("df_final_26082028.csv")

import os
import pickle as pkl

def save_as_pickle_file(data, filename):
    pkl.dump(data, open(filename+'.pkl', 'wb'))


save_as_pickle_file(df_final,"df_final_26082028")

df_final.shape


# Save pickle:

def open_pickle_file(filename):
    return pkl.load(open(filename+'.pkl', 'rb'))

from google.colab import drive
drive.mount('/content/drive')

df_final.to_csv("df_final_20210828.csv")

df_final =pd.read_csv("df_final_20210828.csv")
# df_final.drop(columns="Unnamed: 0",axis=1,inplace=True)


######################################
# Modeling
######################################

X = df_final.drop(columns=["Customer ID","Churn Value","Churn Score","Satisfaction Score","New_NumofDependents_Cat","Population","Tenure Months","Number of Referrals"],axis=1)
y = df_final[["Churn Value"]]

# Train- test split:

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20, random_state = 112)

y_test.shape

y_test["Churn Value"].unique()

# RF Trial:

rf = RandomForestClassifier()
rf_model = rf.fit(X_train,y_train)

# PRIMARY TEST ERROR:

y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

##########################################
# Feature Importance
#########################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_*100, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Let's visualize 20 features:
plot_importance(rf_model, X_test, 25)

classifiers =  [
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('ExtraTrees', ExtraTreesClassifier()),
          ('SVC', SVC()),
          ('GBM', GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier(objective='reg:squarederror')),
          ("LightGBM", LGBMClassifier())]


for name, classifier in classifiers:
    print(name)
    print(classifier)
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")

# We hid the results:
MODEL_AUC=pd.read_excel("MODEL_AUC.xlsx")

sns.barplot(x="AUC", y="MODEL", data=MODEL_AUC.sort_values(by="AUC",ascending=False))
plt.title('Base Models Performance')
plt.figure(figsize=(2, 2))
sns.set(font_scale=10)
plt.tight_layout()
plt.show()

######################################################
# Automated Hyperparameter Optimization
######################################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, None],
             "max_features": ["sqrt" ,"auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [1000, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [500, 1000],
                   "colsample_bytree": [0.5, 0.7, 1]}


classifiers = [ ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


best_models = {}
for name, classifier, params in classifiers:
    print(f"########## {name} ##########")
    cv_results = cross_validate(classifier, X_train, y_train, cv=3, scoring=["roc_auc"])
    print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X_train, y_train, cv=3, scoring=["roc_auc"])
    print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

# GBM - Hyperparameter Optimization (We ran it one by one in the last study, the above is running slow!!!)
gbm_model = GradientBoostingClassifier(random_state=46)
gbm_params = {"learning_rate": [0.01, 0.05],
               "n_estimators": [2000, 1000],
               "max_depth": [6,None],
               "max_features": ["sqrt","auto"],
               # "min_samples_split" : [10, 20],
               "min_samples_leaf" : [12,20]
              }


gbm_gs_best = GridSearchCV(gbm_model,
                           gbm_params,
                           cv=3,
                           n_jobs=-1,
                           verbose=True).fit(X_train, y_train)

# Tuned Model

#gbm_tuned_model = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

gbm_tuned_model = GradientBoostingClassifier(min_samples_leaf = 10, learning_rate = 0.01, max_depth=6, max_features = "sqrt", n_estimators = 2000, random_state=46).fit(X_train, y_train)

# Train Error:
cv_results = cross_validate(gbm_tuned_model, X_train, y_train, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Test Error:
cv_results_test = cross_validate(gbm_tuned_model, X_test, y_test, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

print(cv_results_test['test_accuracy'].mean())
print(cv_results_test['test_f1'].mean())
print(cv_results_test['test_roc_auc'].mean())


#####################################
# Feature Importance
#####################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': gbm_tuned_model.feature_importances_*100, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('GBM Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Let's visualize 20 features:
plot_importance(gbm_tuned_model, X_train, 20)

feature_imp = pd.DataFrame({'Value': gbm_tuned_model.feature_importances_*100, 'Feature': X_train.columns})
feature_imp.sort_values("Value",ascending=False)

feature_imp = pd.DataFrame({'Value': gbm_tuned_model.feature_importances_, 'Feature': X.columns})

nonimp_cols = feature_imp[feature_imp["Value"] == 1]["Feature"].values
selected_cols = [col for col in X.columns if col not in nonimp_cols]
len(selected_cols)

#######################################
# Tuned Model
######################################

#gbm_tuned_model = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

gbm_tuned_model = GradientBoostingClassifier(min_samples_leaf = 10, learning_rate = 0.01, max_depth=6, max_features = "sqrt", n_estimators = 2000, random_state=46).fit(X_train[selected_cols], y_train)

# Train Error:
cv_results = cross_validate(gbm_tuned_model, X_train[selected_cols], y_train, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Test Error:
cv_results_test = cross_validate(gbm_tuned_model, X_test[selected_cols], y_test, cv=10, scoring=["accuracy", "f1", "roc_auc","precision","recall"])

print(cv_results_test['test_accuracy'].mean())
print(cv_results_test['test_f1'].mean())
print(cv_results_test['test_roc_auc'].mean())

########################################
# CONFUSION MATRIX
########################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

y_pred = gbm_tuned_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)

#####################################
# LGBM - Hyperparameter Optimization
#####################################

lgbm_model = LGBMClassifier(random_state=46)
lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [500, 1000,2000],
                "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                           lgbm_params,
                           cv=3,
                           n_jobs=-1,
                           verbose=True).fit(X_train, y_train)

######################################
# Tuned Model
######################################

#gbm_tuned_model = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

lgbm_tuned_model = LGBMClassifier(min_samples_leaf = 10, learning_rate = 0.01, max_depth=6, max_features = "sqrt", n_estimators = 2000, random_state=46).fit(X_train[selected_cols], y_train)

# Train Error:
cv_results = cross_validate(lgbm_tuned_model, X_train[selected_cols], y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Test Error:
cv_results_test = cross_validate(lgbm_tuned_model, X_test[selected_cols], y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results_test['test_accuracy'].mean())
print(cv_results_test['test_f1'].mean())
print(cv_results_test['test_roc_auc'].mean())

#####################################
# Feature Importance
#####################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': lgbm_tuned_model.feature_importances_*100, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('LGBM Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Let's visualize 20 features:
plot_importance(lgbm_tuned_model, X_train, 15)

#####################################
# CONFUSION MATRIX
#####################################

y_pred = lgbm_tuned_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)

##############################################
# XGBoost Forest - Hyperparameter Optimization
##############################################

xgb_model = XGBClassifier(random_state=46)
xgb_params =  {"learning_rate": [0.1, 0.01],
                  "max_depth": [5,8],
                  "n_estimators": [500, 200],
                  "colsample_bytree": [0.7, 0.8]}


rf_gs_best = GridSearchCV(xgb_model,
                          xgb_params,
                          cv=3,
                          n_jobs=-1,
                          verbose=True).fit(X_train, y_train)

################################
# Tuned Model
################################

xgb_tuned_model = XGBClassifier(learning_rate= 0.01, max_depth=8,n_estimators=500, colsample_bytree=0.7).fit(X_train, y_train)

# Train Error:
cv_results = cross_validate(xgb_tuned_model, X_train[selected_cols], y_train, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

# Test Error:
cv_results_test = cross_validate(xgb_tuned_model, X_test[selected_cols], y_test, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results_test['test_accuracy'].mean())
print(cv_results_test['test_f1'].mean())
print(cv_results_test['test_roc_auc'].mean())

################################
# Feature Importance
################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': xgb_tuned_model.feature_importances_*100, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Random Forest Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Let's visualize 20 features:
plot_importance(xgb_tuned_model, X_train, 15)

# Voting Classifer:

voting_clf = VotingClassifier(estimators= [('GBM',  GradientBoostingClassifier(min_samples_leaf = 10,
                                                     learning_rate = 0.01,
                                                     max_depth=6,
                                                     max_features = "sqrt",
                                                     n_estimators = 2000,
                                                     random_state=46)),
                                           ('XGB',XGBClassifier(learning_rate= 0.01, max_depth=8,n_estimators=500, colsample_bytree=0.7)),
                                           ('LGBM', LGBMClassifier(min_samples_leaf = 10,
                                                    learning_rate = 0.01, max_depth=6,
                                                    max_features = "sqrt",
                                                    n_estimators = 2000,
                                                    random_state=46))])

# Model Fit:
voting_clf.fit(X_train[selected_cols], y_train)
y_pred= voting_clf.predict(X_test)
plot_confision_matrix(y_test, y_pred)

##########################
# Success Evaluation
##########################

#########################
# Confusion Matrix
#########################

def plot_confision_matrix(y, y_red):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt=".0f")
    plt.figure(figsize=(1,1))
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy score: {0}".format((acc), size=15))
    plt.show()

plot_confision_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

# ROC CURVE

plot_roc_curve(gbm_tuned_model, X_test[selected_cols], y_test)
plt.title("ROC CURVE")
plt.plot([0, 1],[0,1], 'r--')
plt.xlabel(fontsize=20)
plt.show()

# Possibilities:

y_prob = xgb_tuned_model.predict_proba(X_test[selected_cols])
y_prob = pd.DataFrame(y_prob ,index= X_test.index)
y_prob

# Converting IBM score 0-1, threshold value is 50
y_prob["New_Model_Pred"] = y_prob[1].apply(lambda x: 1 if x > 0.50 else 0)

# left_index/right_index for inner join with index

y_test_son =  y_test.merge(y_prob, left_index=True, right_index=True)

y_test_son["Churn Value"].unique()

y_prob["New_Model_Pred"].unique()

y_test_son.groupby("New_Model_Pred").agg({"New_Model_Pred":"count"})

y_test_son.groupby("Churn Value").agg({"Churn Value":"count"})

# ACCURACY
accuracy_score(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

# PRECISION
precision_score(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

# RECALL
recall_score(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

# F1
f1_score(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

# AUC
roc_auc_score(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_son["Churn Value"], y_test_son["New_Model_Pred"])

print(classification_report(y_test_son["Churn Value"], y_test_son["New_Model_Pred"]))

#  Tüm dataya uygulayıp kırılımlara göre aksiyonlara bakalım:
y_prob_all = xgb_tuned_model.predict_proba(X[selected_cols])
y_prob_all = pd.DataFrame(y_prob_all ,index= X.index)
y_prob_all

# Converting IBM score 0-1, threshold value is 50
y_prob_all["New_Model_Pred"] = y_prob_all[1].apply(lambda x: 1 if x > 0.50 else 0)

# left_index/right_index for inner join with index
y_son =  y.merge(y_prob_all, left_index=True, right_index=True)

y_son.shape

y_son["Churn Value"].unique()

y_son["New_Model_Pred"].unique()

y_son.groupby("New_Model_Pred").agg({"New_Model_Pred":"count"})

y_son.groupby("Churn Value").agg({"Churn Value":"count"})

y_son

tum_data = pd.concat([X,y_son],axis=1)

tum_data.shape

tum_data = tum_data.rename(columns={1:"churn_prob"})

tum_data["New_churn_prob_cat"] = np.where(tum_data["churn_prob"] >=0.90,"Very High",np.NaN)
tum_data["New_churn_prob_cat"] = np.where(( (tum_data["churn_prob"] <0.90) & (tum_data["churn_prob"] >=0.80)),"High",tum_data["New_churn_prob_cat"])
tum_data["New_churn_prob_cat"] = np.where(( (tum_data["churn_prob"] <0.80) & (tum_data["churn_prob"] >=0.45)),"Medium",tum_data["New_churn_prob_cat"])
tum_data["New_churn_prob_cat"] = np.where(( (tum_data["churn_prob"] <0.45) & (tum_data["churn_prob"] >=0.10)),"Low",tum_data["New_churn_prob_cat"])
tum_data["New_churn_prob_cat"] = np.where(tum_data["churn_prob"] <0.10,"Very_Low",tum_data["New_churn_prob_cat"])

tum_data.groupby("New_churn_prob_cat").agg({"churn_prob":["mean","count"]})

tum_data.dtypes

df_final_ = pd.read_csv("df_final_26082021.csv")

tum_data.head(2)

# Let's merge with our first data, we will look at the breakdowns:

df.merge(tum_data[["Customer ID","New_churn_prob_cat"]], on = "Customer ID")

df_ = pd.concat([df, tum_data["New_churn_prob_cat",""]],axis=1)

df_["New_churn_prob_cat"].head(2)

df_.groupby(["New_churn_prob_cat","Contract"]).agg({"Customer ID":"count"})

df_.groupby(["New_churn_prob_cat","Contract"]).agg({"Customer ID":"count",
                                                    })

df_.groupby(["New_churn_prob_cat","Referred a Friend"]).agg({"Customer ID":"count",
                                                    "Churn Value":"mean"})

df_final.groupby(["New_Has_Complaint"]).agg({"Customer ID":"count",
                                                    "Churn Value":"mean"})

# IBM PERFORMANCE:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split


df_final["Churn Value"] # 1: Churned
df_final["Churn Score"] #IBM Score

# Converting IBM score 0-1, threshold value is 50
df_final["IBM_Churn_Score"] = df_final["Churn Score"].apply(lambda x: 1 if x > 50 else 0)

# ACCURACY
accuracy_score(df_final["Churn Value"], df_final["IBM_Churn_Score"])
# 0.6320754716981132

# PRECISION
precision_score(df_final["Churn Value"], df_final["IBM_Churn_Score"])
# 0.420253948590895

# RECALL
recall_score(df_final["Churn Value"], df_final["IBM_Churn_Score"])
# 1.0

# F1
f1_score(df_final["Churn Value"], df_final["IBM_Churn_Score"])
# 0.5918011338857392

# AUC
roc_auc_score(df_final["Churn Value"], df_final["IBM_Churn_Score"])
# 0.7491289198606272


from sklearn.metrics import confusion_matrix
confusion_matrix(df_final["Churn Value"], df_final["IBM_Churn_Score"])
