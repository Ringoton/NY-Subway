import pandas
import numpy
import datetime
import matplotlib.pyplot as plt
from ggplot import *

data = pandas.read_csv('data.csv')
result = pandas.DataFrame()
dateOne_index = data.loc[0][1].split('-')
dateLast_index = data.loc[len(data)-1][1].split('-')
startDate = datetime.date(int(dateOne_index[2]), int(dateOne_index[0]), int(dateOne_index[1]))
startDay = int(data.loc[0][9])
endDate = datetime.date(int(dateLast_index[2]), int(dateLast_index[0]), int(dateLast_index[1]))
totalDays = str(endDate - startDate).split(' ')
Amount = pandas.Series([0, 0, 0, 0, 0, 0, 0])
for i in range(0, int(totalDays[0]) + 1):
    Amount[startDay] += 1
    startDay = (startDay + 1)%7

Days = pandas.Series([0,1,2,3,4,5,6])

total = []
for day1 in Days:
    a = [numpy.sum((data['ENTRIESn_hourly'][data['day_week'] == day1]))]
    total.extend(a)

result['Day'] = pandas.Series(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
result['Amount'] = Amount
result['Total'] = pandas.Series(total)
result['Avg_Day'] = result['Total'] / result['Amount']


plt.figure()
plt.bar(left=Days+1 , height=result['Avg_Day']/100000, align='center')
plt.xticks(Days+1, result['Day'], rotation=45)
plt.ylabel("Ridership in hundred-thousand ")
plt.title("Average Ridership per Day")
plt.subplots_adjust(bottom=0.15)
plt.show()

