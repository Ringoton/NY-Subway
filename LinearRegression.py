import numpy
import pandas
import matplotlib.pyplot as plt


def dummy_norm(data, features):
    dummy_units = pandas.get_dummies(data['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    features_NORM = (features - features.mean()) / features.std()
    features_NORM = numpy.array(features_NORM)

    return features_NORM

def cost(features, values, theta):
    m = len(values)
    h = numpy.dot(features, theta)
    term = numpy.sum(numpy.square(h - values))
    costf = 1/(2*m) * term

    return costf

def LG(features, values, theta, alpha, iterations):
    costs = []
    for i in range(0, iterations):
        h = numpy.dot(features, theta)
        term = numpy.dot((values-h),features)
        theta = theta + (alpha/len(values))*term
        costs.append(cost(features, values, theta))

    return theta, costs

def R_Squared(values, Predictions):
    top = numpy.sum(numpy.square(values - Predictions))
    bottom = numpy.sum(numpy.square(values - numpy.mean(values)))
    R = 1 - (top / bottom)

    return R

alpha = 0.1
data = pandas.read_csv('data.csv')

features = data[['rain', 'precipi', 'hour', 'meantempi']]
values = numpy.array(data['ENTRIESn_hourly'])
ones = numpy.ones(len(values))


features_NORM = numpy.column_stack([dummy_norm(data, features), ones])
theta = [0] * len(features_NORM[1])
iterations = 40

Results = LG(features_NORM, values, theta, alpha, iterations)

Predictions = numpy.dot(features_NORM, Results[0])

print(R_Squared(values, Predictions))

"""
plt.figure()
plt.plot(range(len(Results[1])), Results[1],  'ro')
plt.show()
"""