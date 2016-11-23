import pandas
import numpy
from ggplot import *

data = pandas.read_csv('data.csv')
result = pandas.DataFrame()

Days = pandas.Series([0,1,2,3,4,5,6])
total = []
for day1 in Days:
    a = [numpy.sum((data['ENTRIESn_hourly'][data['day_week'] == day1]))]
    total.extend(a)



result['Total'] = pandas.Series(total)
gg = ggplot(result, aes(x='Days', weight='Total')) + geom_bar() + ggtitle("Total Ridership per Day of Week") + ylab("Total Amount")

print(gg)