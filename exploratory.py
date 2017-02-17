# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:00:54 2017

@author: CISR-salaken
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

raw = pd.read_excel('data/Australia_BITRE_ARDD_Fatalities_Jan_2017.xlsx', sheetname = '2016', 
                     skiprows = [0,1,2,3])
                     
# age distribution
age_dist = raw.groupby('Age')['Age'].agg({'DeathCount': len}).reset_index()

# plot
fig = plt.figure()
plt.bar(age_dist['Age'], age_dist['DeathCount'])

plt.xlabel('Age') 
plt.ylabel('Number of people died')
plt.grid()
plt.minorticks_on()
fig.savefig('images/age_dist.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# which type of road users are more vulnerable
roaduser = raw.groupby('Road User')['Road User'].agg({'DeathCount': len}).reset_index()

# plot
fig = plt.figure()
#plt.bar(np.arange(0,7), roaduser['DeathCount'], align = 'center')
plt.barh(np.arange(0,7), roaduser['DeathCount'], align = 'center')
plt.ylabel('Road User') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.yticks(np.arange(0,7), roaduser['Road User'])
fig.tight_layout()
fig.savefig('images/roaduer_dist.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# which speed limit kills most
speedLimit = raw.groupby('Speed Limit')['Speed Limit'].agg({'DeathCount': len}).reset_index()

# remove the unknown values and outliers
speedLimit = speedLimit.loc[(speedLimit['Speed Limit'] > 0) & (speedLimit['Speed Limit'] < 150), :]


# plot
fig = plt.figure()
plt.bar(np.arange(1,speedLimit.shape[0]+1), speedLimit['DeathCount'], align = 'center')

plt.xlabel('Speed Limit') 
plt.ylabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.xticks(np.arange(1,speedLimit.shape[0]+1), speedLimit['Speed Limit'])
fig.savefig('images/SpeedLimit.png', dpi=300, figsize = (18.5, 10.5))
plt.close()

# which month kills most?
month = raw.groupby('Month', sort = False)['Month'].agg({'DeathCount': len}).reset_index()


# plot
fig = plt.figure()
plt.barh(np.arange(0,month.shape[0]), month['DeathCount'], align = 'center')

plt.ylabel('Month') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.yticks(np.arange(0,month.shape[0]), month['Month'])
fig.tight_layout()
fig.savefig('images/month.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# which day of week sees most death?
day = raw.groupby('Day')['Day'].agg({'DeathCount': len}).reset_index()


# plot
fig = plt.figure()
plt.bar(np.arange(1,day.shape[0]+1), day['DeathCount'], align = 'center')

plt.xlabel('Day of month') 
plt.ylabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.xticks(np.arange(1,32,2), np.arange(1,32,2))
plt.ylim([0,65])
fig.tight_layout()
fig.savefig('images/day.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# which crash type kills most
crashType = raw.groupby('Crash Type', sort = False)['Crash Type'].agg({'DeathCount': len}).reset_index()


# plot
fig = plt.figure()
plt.barh(np.arange(0,crashType.shape[0]), crashType['DeathCount'], align = 'center')

plt.ylabel('Crash Type') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.yticks(np.arange(0,crashType.shape[0]), crashType['Crash Type'])
fig.tight_layout()
fig.savefig('images/crashType.png', dpi=300, figsize = (18.5, 10.5))
plt.close()



# which day of week is killing most?
DoW = raw.groupby('Dayweek', sort = False)['Dayweek'].agg({'DeathCount': len}).reset_index()


# plot
fig = plt.figure()
plt.barh(np.arange(0,DoW.shape[0]), DoW['DeathCount'], align = 'center')

plt.ylabel('Day of Week') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.yticks(np.arange(0,DoW.shape[0]), DoW['Dayweek'])
fig.tight_layout()
fig.savefig('images/DoW.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# since Friday, Saturday and Sunday has most deaths, what hours are the worst?
tmp = raw.loc[(raw['Dayweek'] == 'Friday') | (raw['Dayweek'] == 'Saturday') | (raw['Dayweek'] == 'Sunday'), :]
deadlyHours = tmp.groupby(['Dayweek', 'Hour'])['Crash ID'].agg({'DeathCount': len}).reset_index()

#g = sns.factorplot(x="Hour", y="DeathCount", hue="Dayweek", data=deadlyHours,
#                   size=6, kind="bar", palette="muted", legend_out = False)

g = sns.FacetGrid(col="Dayweek", data=deadlyHours,
                  sharey = True, despine = True,
                  palette="muted", legend_out = False)

g.map(plt.bar, 'Hour','DeathCount')
titles = ["Friday", "Saturday", "Sunday"]
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)
sns.set_style('whitegrid', {'axes.grid' : True})
g.despine(left=True)
g.set_ylabels("Number of people died on road")
g.set_xlabels('Hour of the Day')
g.set_xticklabels(step = 1)
g.savefig('images/DeadlyHours.png', dpi=300, figsize = (18.5, 10.5))
plt.close()

# Brekdown a kde of age among for this days.
tmp = raw.loc[(raw['Dayweek'] == 'Friday') | (raw['Dayweek'] == 'Saturday') | (raw['Dayweek'] == 'Sunday'), :]

g = sns.FacetGrid(col="Dayweek", data=tmp,
                  sharey = True, despine = True, 
                  palette="muted", legend_out = False)

g.map(sns.kdeplot, 'Age', clip = [20,100])
titles = ["Friday", "Saturday", "Sunday"]
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)
g.despine(left=True)
g.set_ylabels("Number of people died on road")
g.set_xlabels('Age')
g.set_xticklabels(step = 1)
g.savefig('images/age_deadlyHours.png', dpi=300, figsize = (18.5, 10.5))
plt.close()







# does gender has any effect? Since driver dies more, would that indicate any correlation?

gender = raw.groupby('Gender')['Gender'].agg({'DeathCount': len}).reset_index()

# discard the only sample with unknown gender
gender = gender.loc[gender['Gender'] != -9,:]

# plot
fig = plt.figure()
plt.barh(np.arange(0,gender.shape[0]), gender['DeathCount'], align = 'center')

plt.ylabel('Gender') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.yticks(np.arange(0,gender.shape[0]), gender['Gender'])
fig.tight_layout()
fig.savefig('images/gender.png', dpi=300, figsize = (18.5, 10.5))
plt.close()



# is there any pattern between young age, gender and death rate?
tmp = raw.loc[(raw['Age'] > 15) & (raw['Age'] <= 60),['Age', 'Gender'] ]
youngAge_gender = tmp.groupby(['Age','Gender'])['Gender'].agg({'DeathCount':len}).reset_index()


g = sns.factorplot(x="Age", y="DeathCount", hue="Gender", data=youngAge_gender,
                   size=6, kind="bar", palette="muted", legend_out = False)
g.despine(left=True)
g.set_ylabels("Death Count")
g.set_xlabels('Age')
g.set_xticklabels(step = 2)
g.savefig('images/AgeSex.png', dpi=300, figsize = (18.5, 10.5))



# what percetage of driver are male and female? What percentage of passangers 
# are male and female?
tmp = raw.loc[(raw['Road User'] == 'Driver') | (raw['Road User'] == 'Passenger'),['Road User', 'Gender']]
DriverPassengerGender = tmp.groupby(['Road User', 'Gender'])['Gender'].agg({'DeathCount': len}).reset_index()

 
##  among the drivers, what is the distribution of age and sex
tmp = raw.loc[raw['Road User'] == 'Driver', ['Road User', 'Age', 'Gender']]

# discard the outliers
tmp = tmp.loc[(tmp['Age'] > 18) & (tmp['Age'] <= 65),:]


Driver_gender = tmp.groupby(['Age','Gender'])['Gender'].agg({'DeathCount':len}).reset_index()

g = sns.factorplot(x="Age", y="DeathCount", hue="Gender", data=Driver_gender,
                   size=6, kind="bar", palette="muted", legend_out = False)

g.despine(left=True)
g.set_ylabels("Death Count (Driver only)")
g.set_xlabels('Age')
g.set_xticklabels(step = 3)
g.savefig('images/Driver_genderSex.png', dpi=300, figsize = (18.5, 10.5))


# which state is more deadly? May be NSW as they have more cars and population?
states = raw.groupby('State')['Crash ID'].agg({'DeathCount': len}).reset_index()
# we need to compare the deaths by percentage of registered vehicles in each state
# data from http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/9309.031%20Jan%202016?OpenDocument#Data
registardVehicleCounts2016 = {'NSW' : 5374419, 'VIC': 4681337, 'QLD': 3854205,
                              'SA': 1364700, 'WA': 2208812, 'TAS': 457629,
                              'NT': 157717, 'ACT': 288317}

registardVehicleCounts2016DF = pd.DataFrame.from_dict(registardVehicleCounts2016, orient = 'index')  
registardVehicleCounts2016DF = registardVehicleCounts2016DF.reset_index()
registardVehicleCounts2016DF.columns = ['State', 'TotalVehicleCount']

states = states.merge(registardVehicleCounts2016DF)                              

# plot
fig = plt.figure()
plt.barh(np.arange(0,states.shape[0]), states['DeathCount'], align = 'center')

plt.ylabel('State') 
plt.xlabel('Number of people died')
plt.grid()
plt.minorticks_on()
plt.grid(b=1)
plt.yticks(np.arange(0,states.shape[0]), states['State'])
fig.tight_layout()
fig.savefig('images/states.png', dpi=300, figsize = (18.5, 10.5))
plt.close()


# a KDE plot for drivers only
# Male
tmp = Driver_gender.loc[Driver_gender['Gender'] == 'Male', 'Age']
g = sns.kdeplot(tmp, legend=False)
g.set_ylabel("Krnel density of Age (Dead Male Driver only)")
#g.set_xticklabels(labels = [], visible = False)
plt.xticks()
f = g.get_figure()
f.savefig('images/MaleDriverAgeDistribution.png', dpi=300, figsize = (18.5, 10.5))
plt.close()

# Female
tmp = Driver_gender.loc[Driver_gender['Gender'] == 'Female', 'Age']
g = sns.kdeplot(tmp, legend=False)
g.set_ylabel("Krnel density of Age (Dead Female Driver only)")
g.set_xticklabels(labels = [], visible = False)
f = g.get_figure()
f.savefig('images/FemaleDriverAgeDistribution.png', dpi=300, figsize = (18.5, 10.5))
plt.close()

