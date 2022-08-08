# -*- coding: utf-8 -*-
"""
Exploratory data analysis (EDA) and Statistical hypothesis testing on Kaggle survey data 2019
@author: Reza Ghasemzadeh
Data source : https://www.kaggle.com/c/kaggle-survey-2019/data
EDA plots:
Fig 1: Professional Experience in ML vs. Average Salary by different educational degree
Fig 2: Average Salary vs. Age groups in different countries
Fig 3: Salary Frequency By Different Educational Degree
Hypothesis Testing:
T-test and ANOVA on:
1. Are female and male salary different?
2. Does formal education affect salary?
#**Creating The Data Frame**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

Data=pd.read_csv("/content/drive/My Drive/MIE1624/clean_kaggle_data.csv")
df=pd.DataFrame(Data)

"""This dataset is a 12497*247 2D matrix.
It has 12497 rows indexed from 0 to 12496. It has 247 columns where 31 of them are integer values and the reminders are string.

#**Q1: Exploratory Data Analysis**
"""

#Percentage of the participants in each salary class
Salary_distribution = (df.groupby('Q10').size()/len(df.Q10)*100).round(2)

"""##**Descriptive Tables**"""

#age and salary
Age=df.groupby('Q1')['Q10']
print(Age.describe().astype(int))

#country and salary
Country=df.groupby('Q3')['Q10']
print(Country.describe().sort_values('count', ascending=False).round(2))

#Eduacation and Salary
Edu = df.groupby('Q4')['Q10'] #Salaries based on Education level
print(Edu.describe().sort_values('mean', ascending=False).round(2))

#Box plot for Edu

#Salaries by Professioanl experience in Machine Learning
ProfExpML=df.groupby('Q23')['Q10']
print(ProfExpML.describe().sort_values('mean', ascending=False).round(2))

"""##**Fig.1: Professional Experience in ML vs. Average Salary by different educational degree**"""

#Professional Experience: For how many years have you used machine learning methods?
plt.figure(figsize=(10,7))
ProfExp_BSc=pd.DataFrame(df[df.Q4=='Bachelor’s degree'].groupby(['Q23'])['Q10'].mean())
#the following line sorts the years bin in ascending order
ProfExp_BSc_sorted=pd.DataFrame(ProfExp_BSc.iloc[[-1, 0, 2, 4, 5, -2, 1, 3]])
plt.plot(ProfExp_BSc_sorted, label='BSc')
###
ProfExp_Master=pd.DataFrame(df[df.Q4=='Master’s degree'].groupby(['Q23'])['Q10'].mean())
ProfExp_Master_sorted=pd.DataFrame(ProfExp_Master.iloc[[-1, 0, 2, 4, 5, -2, 1, 3]])
plt.plot(ProfExp_Master_sorted, label='Master')
###
ProfExp_PhD=pd.DataFrame(df[df.Q4=='Doctoral degree'].groupby(['Q23'])['Q10'].mean())
ProfExp_PhD_sorted=pd.DataFrame(ProfExp_PhD.iloc[[-1, 0, 2, 4, 5, -2, 1, 3]])
plt.plot(ProfExp_PhD_sorted, label='PhD')
plt.legend()
plt.xlabel('Professional Experience in ML')
plt.ylabel('Average Salary (USD)')
plt.title('Professional Experience in ML vs. Average Salary by different educational degree')

"""From the above graph, as the participant's professional experience increases, we can figure out:

  - when a person starts working with **bachelor** degree, his/her income increases **exponentially**.
  - when a person starts working with **master** degree, his/her income increases almost **linearly**.
  - when a person starts working with **PhD** degree, his/her income increases almost **sub-linearly**.

We can also conclude that:

  - **Bachelor** degree holders get **lower** salary than others when their professional experience is less than **4 years** and vice versa.
  - **PhD** degree holders get **higher** salary than others when their professional experience is less than **4 years** and vice versa.

##**Fig.2: Average Salary vs. Age groups in different countries**
"""

#Average Salary vs. Age groups
plt.figure(figsize=(10,7))
plt.plot(df.groupby('Q1')['Q10'].mean().round(-1), label='Globe', ls='--')
plt.plot(df[df.Q3=='United States of America'].groupby('Q1')['Q10'].mean().round(-1), label='USA')
plt.plot(df[df.Q3=='India'].groupby('Q1')['Q10'].mean().round(-1), label='India')
plt.plot(df[df.Q3=='Canada'].groupby('Q1')['Q10'].mean().round(-1), label='Canada')
plt.xlabel("Age")
plt.ylabel('Avg. Salary (USD)')
plt.legend()
plt.title('Average Salary vs. Age groups in different countries')

# #Average Salary vs. Age groups in different continents. USA and India are treated exclusively because otherwise their large number of participants will completely deviate the impressions.
# plt.figure(figsize=(10,7))
# plt.plot(df.groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Globe', ls='--')
# plt.plot(df[df['Continent']=='USA'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='USA')
# plt.plot(df[df['Continent']=='India'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='India')
# plt.plot(df[df['Continent']=='Europe'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Europe')
# plt.plot(df[df['Continent']=='Oceania'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Oceania')
# plt.plot(df[df['Continent']=='SouthAmerica'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='SouthAmerica')
# plt.plot(df[df['Continent']=='NorthAmerica'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='NorthAmerica')
# plt.plot(df[df['Continent']=='Africa'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Africa')
# plt.plot(df[df['Continent']=='Other'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Other')
# plt.plot(df[df['Continent']=='Asia'].groupby('Q1')['Q10_Encoded'].mean().round(-1), label='Asia')
# plt.xlabel("Age")
# plt.ylabel('Avg. Salary (USD)')
# plt.legend()
# plt.title('Average Salary vs. Age groups in different continents')

"""**Interpretations:**

- In general, as a person's age increases, his/her salary increases, too.
- In USA, the avg. income are almost 3x highier than the global values
- Average income in Canada oscillate around the global trend for age buckets greater 30 yrs. 
- In India, average income is blow the global trend for all age categories. It diverges futher as the age increases.

##**Fig.3 Salary Frequency By Different Educational Degree**

Because the salary increamental is somewhat exponential, it seem good to show the plot in **logaritmic** scale for X axis:
"""

np_bachelor=np.array(df[df.Q4=='Bachelor’s degree']['Q10'])
(bachelor_salary, bachelor_counts)=np.unique(np_bachelor, return_counts=True)
bachelor_percentage=bachelor_counts/(np.sum(bachelor_counts))
#
np_master=np.array(df[df.Q4=='Master’s degree']['Q10'])
(master_salary, master_counts)=np.unique(np_master, return_counts=True)
master_percentage=master_counts/(np.sum(master_counts))
#
np_phd=np.array(df[df.Q4=='Doctoral degree']['Q10'])
(phd_salary, phd_counts)=np.unique(np_phd, return_counts=True)
phd_percentage=phd_counts/(np.sum(phd_counts))
#
plt.figure(figsize=(10,7))
plt.plot(np.log10(bachelor_salary), bachelor_percentage, label='Bachelor')
plt.plot(np.log10(master_salary), master_percentage , label="Master")
plt.plot(np.log10(phd_salary), phd_percentage , label="PhD")
plt.xlabel('Log Salary in USD')
plt.ylabel('Probability (%)')
plt.title('Salary by Education Level -Logarithmic Scale')
plt.legend()

"""**Interpretation:**
  - Generally, the chance of getting higher salary decreases for all categories. 
  - People with Bachelor Degree getting higher salaries than two other groups for salaries up to 25,000 USD
  - People with Master degree are almost always falling in the middle of the salary bound
  - People with PhD Degree getting higher salaries than two other groups for salaries after 50,000 USD

#**Q2: Female and Male Salary Comparison**

##**Q2.a: Descriptive Statistics**
"""

#Filtering the original data by gender(Q2) and picking the salary column (Q10)
female=pd.DataFrame(df[df.Q2=='Female']['Q10'])
male=pd.DataFrame(df[df.Q2=='Male']['Q10'])

Stat_Description=female.describe()
Stat_Description.columns=['Female Salary($)']
Stat_Description['Male Salary($)']=male.describe()
Stat_Description['Difference($)']=Stat_Description['Female Salary($)']-Stat_Description['Male Salary($)']
Stat_Description.astype(int)

plt.boxplot([df[df.Q2=='Male']['Q10'], df[df.Q2=='Female']['Q10']], showfliers = False)
plt.gca().set(title = "Boxplot of Salary vs. Gender ", xticklabels = ['Male', "Female"], ylabel = 'Salary (USD)');
plt.grid()
plt.show()

#Grouping salary values for each gender
np_female=np.array(df[df.Q2=='Female']['Q10'])
(fem_salary, fem_counts)=np.unique(np_female, return_counts=True)
fem_percentage=fem_counts/(np.sum(fem_counts))
#
np_male=np.array(df[df.Q2=='Male']['Q10'])
(male_salary, male_counts)=np.unique(np_male, return_counts=True)
male_percentage=male_counts/(np.sum(male_counts))

#**LOGARITHMIC VIEW**
plt.figure(figsize=(10,7))
plt.plot(np.log10(fem_salary), fem_percentage, label='Female')
plt.plot(np.log10(male_salary), male_percentage , label="Male")
plt.xlabel('Log Salary in USD')
plt.ylabel('Frequency (%)')
plt.title('Logarithmic View')
plt.legend()

"""From the above graphs, female's are dominating the salary up to 7500 USD while after this point, male become dominant. In other words, the probability of a man getting, for example, 10,000 USD (or any number above 7500) is higher than a woman.

##**Q2.b: T-test** 
Before performing the test, we need to make sure that the underlying assumptions -listed below- are met:  

T-test assumptions ([See Reference](https://www.investopedia.com/ask/answers/073115/what-assumptions-are-made-when-conducting-ttest.asp#:~:text=The%20common%20assumptions%20made%20when,of%20variance%20in%20standard%20deviation.)):
  - Have bell-shaped normal distribution 
  - Data points have continuous or ordinal scale
  - Randomly collected
  - Large sample size
  - Equality of the variances of the samples

So, we should plot the groups distribution to see whether they are normal or not.
"""

#**NORMAL VIEW**
plt.figure(figsize=(10,7))
plt.plot(fem_salary, fem_percentage, label='Female')
plt.plot(male_salary, male_percentage , label="Male")
plt.xlabel('Salary in USD')
plt.ylabel('Frequency (%)')
plt.legend()

"""**Because the distribution of the datapoints either for male or females are not normal, we cannot perform t-test (i.e. the first assumption is not met)**

##**Q2.c: Bootstraping**
"""

male_bootmean=[]
female_bootmean=[]
diff_bootmean=[]
for i in range(0,1000):
  # creating bootsrap sample
  male_bootsample=np.random.choice(np_male ,np_male.size, replace=True)
  female_bootsample=np.random.choice(np_female ,np_female.size, replace=True)
  #Calculating the mean value of the samples and their difference in the mean value, then storing them
  male_bootmean.append(np.mean(male_bootsample).astype(int))
  female_bootmean.append(np.mean(female_bootsample).astype(int))
  diff_bootmean.append((np.mean(male_bootsample)-np.mean(female_bootsample)).astype(int))

#Descriptive Statistic of the bootstraped sample means
df_gender=pd.DataFrame([male_bootmean,female_bootmean, diff_bootmean]).T
df_gender.columns= ['Male BT Mean Salary', 'Female BT Mean Salary', 'Difference in BT Mean Salary']
df_gender.describe().round()

plt.figure(figsize=(20,7))
plt.hist(male_bootmean,50, label='Male')
plt.hist(female_bootmean, 50, label='Female')
plt.xlabel('Mean Salary (USD)')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution')
plt.legend()

#Standard Error of the Two Distribution
male_bootmean_sterr=stats.sem(male_bootmean).round()
female_bootmean_sterr=stats.sem(female_bootmean).round()
print('Standard Error of the Male bootstrap mean distribution:',male_bootmean_sterr)
print('Standard Error of the Female bootstrap mean distribution:',female_bootmean_sterr)

plt.hist(diff_bootmean,30, label='Difference')
plt.xlabel('Mean Salary Difference (USD)')
plt.ylabel('Frequency')
plt.legend()

# Mean, Standard Deviation, and 95% Confidence Interval of the Differences in Mean Salary
mu, sigma = np.mean(diff_bootmean), np.std(diff_bootmean)
print ('Mean=', mu.round(2), ' and ', 'Standard Deviation=', sigma.round(2))
conf_int = stats.norm.interval( 0.95, loc = mu, scale = sigma/np.sqrt(1000) )
print('95% Confidence Interval:',conf_int)

"""##**Q2.d: T-test of bootstrap**

As it can be seen from the above graphs, mean values of both of the bootstrapped groups have normal distribution, so we can run t-test on them.
"""

#T-Stat of the two distibution
tstat, pvalue = stats.ttest_ind(male_bootmean, female_bootmean) 
# tstat, pvalue
print('T-stat  = ', tstat.round(3))
print('P-Value = ', pvalue)
Pcr=0.05
if pvalue>Pcr:
  print('Null Hypothesis ACCEPTED; The Average Salary is independent of the gender')
else:
  print('Null Hypothesis REJECTED; The Average Salary depend on the gender')

"""##**Q2e: Comments on Findings**

a. 
- The portion of the male participants is almost 5x of the portion of the female participants. So, in general, male datapoint are reliable than the female datapoints meaning that they better represent their corresponding population in the world than the female although both are not enough. It should be noted that the samples either in male or female are not sufficiently large to meet the CLT criteria. 
- According to these two samples, on average, men’s salary is 12775 USD higher than female’s salary. Even though their minimum and maximum compensation are equal, but the median of the male’s salary is 10,000 USD higher than the female’s salary.
- There is much variation in the 4th quantile (75% to Max) of either group.
- From the Q2.a second graph, we can see females are dominating the salary up to 7500 USD while after this point, male become dominant. In other words, the probability of a man getting, for example, 10,000 USD (or any number above 7500) is higher than a woman. 
Also, as the salary increases the probability decreases in both cases.

b. 

Before performing the test, we need to make sure that the underlying assumptions -listed below- are met:  

T-test assumptions ([See Reference](https://www.investopedia.com/ask/answers/073115/what-assumptions-are-made-when-conducting-ttest.asp#:~:text=The%20common%20assumptions%20made%20when,of%20variance%20in%20standard%20deviation.)):
  - Have bell-shaped normal distribution 
  - Data points have continuous or ordinal scale
  - Randomly collected
  - Large sample size
  - Equality of the variances of the samples

Based on the Q2.b graph, the distribution of the datapoints either for male or females are not normal, we cannot perform two-sample t-test (i.e. the first assumption is not met)

c. 
- By bootstrapping, we can see it fixed the un-normality of the original samples distribution and making them a better representative of the corresponding population without changing their mean. 
- By plotting the bootstrapping mean distribution for each group, we can see that mean of the salary of the male group is 12739 USD higher than female groups as of before. The 95% of the confidence interval for the differences in mean is ($12638, $12840)
- The standard deviation of these two distributions of the means represents the standard error of the data. So, the standard error of the male salaries is 720 while for the female is 1442, 2x larger!


d. 

Because the distributions of the means of the groups are normal, we performed two-sample t-test. 

The Null Hypothesis: “Males and Females getting same salary on average”

The t-stat values found to be 249.8 with P-value of 0.0

Since the calculated p-value is less than the threshold (0.05), so we **reject the null hypothesis and conclude that the average salary depends on the gender!**

#**Q3: Formal Education Comparison**
"""

# Filtering the original dataset
bachelor=pd.DataFrame(df[df.Q4=='Bachelor’s degree']['Q10'])
master=pd.DataFrame(df[df.Q4=='Master’s degree']['Q10'])
phd=pd.DataFrame(df[df.Q4=='Doctoral degree']['Q10'])

"""##**Q3.a: Descriptive Statistics**"""

edu_statdesc=bachelor.describe().astype(int)
edu_statdesc.columns=['Bachelor']
edu_statdesc['Master']=master.describe().astype(int)
edu_statdesc['PhD']=phd.describe().astype(int)
edu_statdesc['Difference($) Master&Bachelor']=edu_statdesc['Master']-edu_statdesc['Bachelor']
edu_statdesc['Difference($) PhD&Bachelor']=edu_statdesc['PhD']-edu_statdesc['Bachelor']
edu_statdesc['Difference($) PhD&Master']=edu_statdesc['PhD']-edu_statdesc['Master']
edu_statdesc

plt.boxplot([df[df.Q4=='Bachelor’s degree']['Q10'], df[df.Q4=='Master’s degree']['Q10'], 
            df[df.Q4=='Doctoral degree']['Q10']], showfliers = False)
plt.gca().set(title = "Boxplot of Salary vs. Educational Degree ", xticklabels = ['Bachelor', 'Master', 'PhD'], ylabel = 'Salary (USD)');
plt.grid()
plt.show()

"""##**Q3.b: ANOVA for Original Sample**
Assumptions for One-Way ANOVA Test Section ([see reference](https://online.stat.psu.edu/stat500/lesson/10/10.2/10.2.1#:~:text=There%20are%20three%20primary%20assumptions,The%20data%20are%20independent.))

  - The responses for each factor level have a normal population distribution.
  - These distributions have the same variance.
  - The data are independent.

So, we should plot the groups distribution to see whether they are normal or not.
"""

np_bachelor=np.array(df[df.Q4=='Bachelor’s degree']['Q10'])
(bachelor_salary, bachelor_counts)=np.unique(np_bachelor, return_counts=True)
bachelor_percentage=bachelor_counts/(np.sum(bachelor_counts))
#
np_master=np.array(df[df.Q4=='Master’s degree']['Q10'])
(master_salary, master_counts)=np.unique(np_master, return_counts=True)
master_percentage=master_counts/(np.sum(master_counts))
#
np_phd=np.array(df[df.Q4=='Doctoral degree']['Q10'])
(phd_salary, phd_counts)=np.unique(np_phd, return_counts=True)
phd_percentage=phd_counts/(np.sum(phd_counts))
#
#**NORMAL VIEW**
plt.figure(figsize=(10,7))
plt.plot(bachelor_salary, bachelor_percentage, label='Bachelor')
plt.plot(master_salary, master_percentage , label="Master")
plt.plot(phd_salary, phd_percentage , label="PhD")
plt.xlabel('Salary in USD')
plt.ylabel('Probability (%)')
plt.title('Salary by Education Level -Normal Scale')
plt.legend()

"""
**As we can see from the above graphs, the first assumption (i.e. having normal distriburion) is not met. So, we cannot perform ANOVA on these datapoints.**"""

# fstat1, pval1 = stats.f_oneway(bachelor, master, phd)
# print('F-stat  = ', fstat1.round(3))
# print('P-Value = ', pval1)

"""##**Q3.c: Bootsrapping**"""

bachelor_bootmean=[]
master_bootmean=[]
phd_bootmean=[]
diff_bachelor_master=[]
diff_bachelor_phd=[]
diff_master_phd=[]
for i in range(0,1000):
  #creating bootstrap sample for each group
  bachelor_bootsample=np.random.choice(np_bachelor ,np_bachelor.size, replace=True)
  master_bootsample=np.random.choice(np_master ,np_master.size, replace=True)
  phd_bootsample=np.random.choice(np_phd ,np_phd.size, replace=True)
  #calculating the mean of the samples and storing them
  bachelor_bootmean.append(np.mean(bachelor_bootsample).astype(int))
  master_bootmean.append(np.mean(master_bootsample).astype(int))
  phd_bootmean.append(np.mean(phd_bootsample).astype(int))
  #Calculating the differences between each group
  diff_bachelor_master.append((np.mean(master_bootsample)-np.mean(bachelor_bootsample)).astype(int))
  diff_bachelor_phd.append((np.mean(phd_bootsample)-np.mean(bachelor_bootsample)).astype(int))
  diff_master_phd.append((np.mean(phd_bootsample)-np.mean(master_bootsample)).astype(int))

df_edu_boot=pd.DataFrame([bachelor_bootmean, master_bootmean, phd_bootmean, diff_bachelor_master, diff_bachelor_phd, diff_master_phd]).T
df_edu_boot.columns= ['Bachelor Mean Salary', 'Master Mean Salary', 'PhD Mean Salary', 'Master and Bachelor Diff. in Mean Salary', 
                    'PhD and Bachelor Diff. in Mean Salary', 'PhD and Master Diff. in Mean Salary']
df_edu_boot.describe().round()

#boxplot

plt.figure(figsize=(20,7))
plt.hist(bachelor_bootmean,15, label='Bachelor')
plt.hist(master_bootmean, 15, label='Master')
plt.hist(phd_bootmean, 15, label='PhD')
plt.xlabel('Mean Salary (USD)')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution')
plt.legend()

#Differnce Distribution in Combined view
# plt.hist(diff_bachelor_master, 15, label='Master and Bachelor', alpha=0.7)
# plt.hist(diff_bachelor_phd, 15, label='PhD and Bachelor')
# plt.hist(diff_master_phd, 15, label='PhD and Master', alpha=0.7)
# plt.title('Differnce Distributions in Combined view')
# plt.legend()
# plt.show()

plt.hist(diff_bachelor_master, 15, label='Master and Bachelor')
plt.title('Master and Bachelor')
plt.legend()
plt.show()
#
plt.hist(diff_bachelor_phd, 15, label='PhD and Bachelor', color='orange')
plt.title('PhD and Bachelor')
plt.legend()
plt.show()
#
plt.hist(diff_master_phd, 15, label='PhD and Master', color='g')
plt.title('Master and PhD')
plt.legend()
plt.show()

mu2, sigma2 = np.mean(diff_bachelor_master), np.std(diff_bachelor_master)
print ('Mean=', mu2.round(2), ' and ', 'Standard Deviation=', sigma2.round(2))
conf_int = stats.norm.interval( 0.95, loc = mu2, scale = sigma2/np.sqrt(1000) )
print('95% Confidence Interval of Difference between Bachelor and Master:',conf_int)

mu3, sigma3 = np.mean(diff_bachelor_phd), np.std(diff_bachelor_phd)
print ('Mean=', mu3.round(2), ' and ', 'Standard Deviation=', sigma3.round(2))
conf_int = stats.norm.interval( 0.95, loc = mu3, scale = sigma3/np.sqrt(1000) )
print('95% Confidence Interval of Difference between Bachelor and PhD:',conf_int)

mu4, sigma4 = np.mean(diff_master_phd), np.std(diff_master_phd)
print ('Mean=', mu4.round(2), ' and ', 'Standard Deviation=', sigma4.round(2))
conf_int = stats.norm.interval( 0.95, loc = mu4, scale = sigma4/np.sqrt(1000) )
print('95% Confidence Interval of Difference between PhD and Master:',conf_int)



"""##**Q3.d: ANOVA on Bootsrapped Samples**

Since we have only one variable so we should perform one way ANOVA
"""

fstat, pval = stats.f_oneway(bachelor_bootmean, master_bootmean, phd_bootmean)
print('F-stat  = ', fstat.round(3))
print('P-Value = ', pval)
#
Pcr=0.05
if pval>Pcr:
  print('By the P-value definition, Null Hypothesis ACCEPTED; The Average Salary is independent of the Educational Degree')
else:
  print('By the P-value definition, Null Hypothesis REJECTED; The Average Salary depends on the Educational Degree')
#
Fcr=2.9957    #When DoF of numerator(SSB) is 2 and DoF of denumerator(SSW) is 3*999
if fstat<Fcr:
  print('By the F-stat  definition, Null Hypothesis ACCEPTED; The Average Salary is independent of the Educational Degree')
else:
  print('By the F-stat  definition, Null Hypothesis REJECTED; The Average Salary depends on the Educational Degree')

"""##**Q3e: Comments on Findings**

a. 
- The portion of the master holders is almost 2x of bachelor holders and 3x of PhD holders. So, in general, master datapoint are reliable than the two other datapoints meaning that they better represent their corresponding population in the world than the other although none of these three groups are not enough. It should be noted that number of the datapoints in each group is not sufficiently large to meet the CLT criteria. 
- According to these original samples, on average, PhD holder salaries are, on average, 30762 USD higher than bachelor’s salary and 16983 USD higher than master’s salary. Even though their minimum and maximum compensation are equal, but the median of the PhD holder salary is 4x larger than the bachelor’s salary and 1.5x larger than the bachelor’s salary.
- There is much variation in the 4th quantile (75% to Max) of all groups.
- From the Q3.a graph, we can see:
 People with Bachelor Degree getting higher salaries than two other groups for salaries up to 25,000 USD
People with Master degree are almost always falling in the middle of the salary bound
People with PhD Degree getting higher salaries than two other groups for salaries after 50,000 USD
Also, as the salary increases the probability decreases in all cases.

b. Assumptions for One-Way ANOVA Test Section ([see reference](https://online.stat.psu.edu/stat500/lesson/10/10.2/10.2.1#:~:text=There%20are%20three%20primary%20assumptions,The%20data%20are%20independent.))

  - The responses for each factor level have a normal population distribution.
  - These distributions have the same variance.
  - The data are independent.

As we can see from the Q3.b graph, the first assumption (i.e. having normal distriburion) is not met. So, **we cannot perform ANOVA on these datapoints.**

c. 
- By bootstrapping, we can see it fixed the un-normality of the original samples distribution and making them a better representative of the corresponding population without changing their mean. 
- By plotting the bootstrapping mean distribution for each group, we can see that mean of the salary of the groups are preserved as of before.

95% Confidence Interval of Difference between average salary of Bachelor and Master: (13726, 13909)

95% Confidence Interval of Difference between average salary of Bachelor and PhD: (30675, 30939)

95% Confidence Interval of Difference between average salary of PhD and Master: (16867, 17112)

-The standard deviation of these three distributions of the means represents the standard error of the data. So, the standard error of Master and Bachelor Diff. in Mean Salary, PhD and Bachelor Diff. in Mean Salary', 'PhD and Master Diff. in Mean Salary' are 1476,	2132, and	1978, respectively. 

d. Because the distributions of the means of the groups are normal, we performed ANOVA. Since we have only one variable so we should perform one-way ANOVA.

The Null Hypothesis: “Bachelor, Master, and PhD holders get same salary on average, regardless of their educational degree”

Here are the results of ANOVA test:
  - F-stat = 131534
  - P-Value = 0.0

By the P-value definition, since the calculated p-value is less than the threshold (0.05), Null Hypothesis REJECTED; The Average Salary depends on the Educational Degree

By the F-stat  definition, , since the calculated f-value is greater than the f-critical which is 2.997 for alpha=0.05 when DoF of nominator (SSB) is 2 and DoF of denominator  (SSW) is 3*999, Null Hypothesis REJECTED; The Average Salary depends on the Educational Degree.
