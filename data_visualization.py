#############################################
# KAPLAN-MEIER CURVES
############################################


########### IMPORT LIBARIES ###########
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from lifelines.statistics import (logrank_test,
                                  pairwise_logrank_test,
                                  multivariate_logrank_test,
                                  survival_difference_at_fixed_point_in_time_test)

plt.style.use('seaborn')

#Lifelines is a survival analysis package
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

eventvar = df['Churn Value']
timevar = df['Tenure Months']

df.columns

#Create a KaplanMeier object, imported from lifelines
kmf = KaplanMeierFitter()
# Calculate the K-M curve for all groups
kmf.fit(timevar,event_observed = eventvar,label = "All Customers")
#Plot the curve and assign labels
kmf.plot()
plt.ylabel('Probability of Customer Survival')
plt.xlabel('Tenure')
plt.title('Kaplan-Meier Curve');
plt.show()

""" There is a sudden decrease in the beginning and gradually over time.
continues. long to deal with it
may consider further discounting on future plans and
More customers can be subscribed to long-term plans. """


############# LOG - RANK TEST #############

"""
Comparing survival curves between different groups
non-parametric method log-rank test is used. log-rank
The test assumes that the hazards of the groups are proportional. None
Under the hypothesis, the probability of the event between groups at all time points
is the same for.
"""

male = (df["Gender"] == "Male")
female = (df["Gender"] == "Female")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[male], event_observed=eventvar[male], label="Male")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[female], event_observed=eventvar[female], label="Female")
plot2 = kmf.plot(ax=plot1)

plt.title('Survival of customers: Gender')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
groups = logrank_test(timevar[male], timevar[female], event_observed_A=eventvar[male],
                      event_observed_B=eventvar[female])
groups.print_summary()
plt.show()


############## Senior Citizen ####################

df.columns
SeniorCitizen = (df['Senior Citizen'] == "Yes")
no_SeniorCitizen = (df['Senior Citizen'] == "No")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[SeniorCitizen], event_observed=eventvar[SeniorCitizen], label="Senior Citizen")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[no_SeniorCitizen], event_observed=eventvar[no_SeniorCitizen], label="Not a Senior Citizen")
plot2 = kmf.plot(ax=plot1)

plt.title('Survival of customers: Senior Citizen')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
groups = logrank_test(timevar[SeniorCitizen], timevar[no_SeniorCitizen], event_observed_A=eventvar[SeniorCitizen],
                      event_observed_B=eventvar[no_SeniorCitizen])
groups.print_summary()
plt.show()


############## Internet Service ####################

Fiber_optic = (df['Internet Service'] == "Fiber optic")
No_Service = (df['Internet Service'] == "No")
DSL = (df['Internet Service']== "DSL")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[Fiber_optic], event_observed=eventvar[Fiber_optic], label="Fiber optic")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[No_Service], event_observed=eventvar[No_Service], label="No Service")
plot2 = kmf.plot(ax=plot1)

kmf.fit(timevar[DSL], event_observed=eventvar[DSL], label="DSL")
plot3 = kmf.plot(ax=plot2)

plt.title('Survival of customers: Internet Service')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
twoplusgroups_logrank = multivariate_logrank_test(df['Tenure Months'], df['Internet Service'], df['Churn Value'], alpha=0.95)
twoplusgroups_logrank.print_summary()
plt.show()

df["Tech Support"]


################ Teach Support  ################

no_internetService = (df['Tech Support'] == "No internet service")
TechSupport = (df['Tech Support'] == "Yes")
no_TechSupport = (df['Tech Support'] == "No")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[no_internetService], event_observed=eventvar[no_internetService], label="No Internet Service")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[TechSupport], event_observed=eventvar[TechSupport], label="Tech Support")
plot2 = kmf.plot(ax=plot1)

kmf.fit(timevar[no_TechSupport], event_observed=eventvar[no_TechSupport], label="No Tech Support")
plot3 = kmf.plot(ax=plot2)

plt.title('Survival of customers: Tech Support')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
twoplusgroups_logrank = multivariate_logrank_test(df['Tenure Months'], df['Tech Support'], df['Churn Value'], alpha=0.95)
twoplusgroups_logrank.print_summary()
plt.show()


################ Contract  ################

df.columns
df["Contract"]

Contract_One_year = (df['Contract'] == "One year")
Contract_Two_year = (df['Contract'] == "Two year")
Contract_month_to_month = (df['Contract'] == "Month-to-month")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[Contract_One_year], event_observed=eventvar[Contract_One_year], label="One year Contract")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[Contract_Two_year], event_observed=eventvar[Contract_Two_year], label="Two year Contract")
plot2 = kmf.plot(ax=plot1)

kmf.fit(timevar[Contract_month_to_month], event_observed=eventvar[Contract_month_to_month],
        label="Month to month Contract")
plot3 = kmf.plot(ax=plot2)

plt.title('Survival of customers: Contract')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
twoplusgroups_logrank = multivariate_logrank_test(df['Tenure Months'], df['Contract'], df['Churn Value'], alpha=0.95)
twoplusgroups_logrank.print_summary()
plt.show()


################ Payment Method  ################

df.columns
df["Payment Method"]

automatic_Credit_Card = (df['Payment Method'] == "Credit card (automatic)")
electronic_check = (df['Payment Method'] == "Electronic check")
mailed_check = (df['Payment Method'] == "Mailed check")
automatic_Bank_Transfer = (df['Payment Method'] == "Bank transfer (automatic)")

plt.figure()
ax = plt.subplot(1, 1, 1)

kmf.fit(timevar[automatic_Credit_Card], event_observed=eventvar[automatic_Credit_Card],label="Automatic Credit card Payment")
plot1 = kmf.plot(ax=ax)

kmf.fit(timevar[electronic_check], event_observed=eventvar[electronic_check], label="Electronic Check")
plot2 = kmf.plot(ax=plot1)

kmf.fit(timevar[mailed_check], event_observed=eventvar[mailed_check], label="Mailed_check")
plot3 = kmf.plot(ax=plot2)

kmf.fit(timevar[automatic_Bank_Transfer], event_observed=eventvar[automatic_Bank_Transfer],
        label="Automatic Bank Transfer")
plot4 = kmf.plot(ax=plot3)

plt.title('Survival of customers: PaymentMethod')
plt.xlabel('Tenure Months')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))
twoplusgroups_logrank = multivariate_logrank_test(df['Tenure Months'], df['Payment Method'], df['Churn Value'], alpha=0.95)
twoplusgroups_logrank.print_summary()
plt.show()


##############################
# OTHER VISUALIZATION
##############################

var_gender='Churn Label'
print(df[var_gender].unique())
y=df[var_gender]
ax = sns.countplot(y,label="Churn Label")
no, yes = y.value_counts()
print('Number of yes: ',yes)
print('Number of no : ',no)

# percentage
print('percentage value of yes:',float(yes)*100/float(yes+no),'%')
print('percentage value of no :',float(no)*100/float(yes+no),'%')
plt.show()

labels='No','Yes'
m=float(yes)*100/float(yes+no)
f=float(no)*100/float(yes+no)
sizes=df['Churn Label'].value_counts()
plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=140)
plt.legend(labels=labels)
plt.title("churm rate")
plt.axis('equal')
plt.show()

# Gender and Relative Churn Rates in Population
gb = df.groupby("Gender")["Churn Label"].value_counts().to_frame().rename({"Churn Label": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Gender", y = "Number of Customers", data = gb, hue = "Churn Label", palette = sns.color_palette("Set2", 15)).set_title("Gender and relative Churn Rates in our population");
plt.show()


