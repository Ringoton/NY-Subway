import re
import pandas
import numpy
import random
import matplotlib.pyplot as plt
random.seed(version=10)

raw_data = open("Auto.txt")
data = raw_data.read()
data = data.strip().split('\n')
header = re.split('\t| ', data[0])
list = []
for word in data[1:]:
    part1 = re.split('\t', word)
    part2 = part1[0].split()
    if '?' in part2:
        continue
    list.append(part2)
final1 = [pandas.to_numeric(x) for x in list]
final2 = numpy.array(final1)
final = pandas.DataFrame(final2, columns=header[0:8])


corr_matrix = final.corr(method='pearson', min_periods=1)
#print(corr_matrix[(corr_matrix > 0.90) | (corr_matrix < -0.90)])
#Looking for high correlations


continuousVar = ['displacement', 'horsepower', 'weight', 'acceleration']
ordinals = []
beta = [1] * (len(continuousVar) + len(ordinals) + 1)
k = 5
alpha = 0.08
iterations = 40

def normalize(final, continuousVar):
    final_norm = (final[continuousVar] - final[continuousVar].mean()) / final[continuousVar].std()
    #print(type(final_norm))
    
    return final_norm
    
    
def categorize(final_norm, final, ordinals):
    for item in ordinals:
        #print(item)
        final_norm[item] = final[item].astype('category', ordered=True)
        
    return final_norm
    

def k_fold_cross(final_norm_cat, k):
    length = len(final_norm_cat)
    size = length // k
    part1 = [x for x in range(length)]
    selection = []
    for _ in range(k):
        selection.append(random.sample(part1, size))
        part1 = [x for x in part1 if x not in numpy.array(selection)]
        
    return selection
    
def k_fold_cross2(selection, final_norm_cat):
    compliment = set(range(len(final_norm_cat))) - set(selection)
    
    return sorted(compliment)
    
    
def ones(final_norm_cat):
    final_norm_cat['Constant'] = 1
    
    return final_norm_cat
    
    
def costFunction(features, beta, outcome):
    m = len(outcome)
    h = numpy.dot(features, beta)
    term = numpy.sum(numpy.square(h - outcome))
    costf = 1/(2*m) * term
    
    return costf
    
    
def regression(alpha, iterations, features, beta, outcome):
    costs = []
    for i in range(0, iterations):
        h = numpy.dot(features, beta)
        term = numpy.dot((outcome-h),features)
        beta = beta + (alpha/len(outcome))*term
        costs.append(costFunction(features, beta, outcome))

    return beta, costs
        
def R_Squared(values, Predictions):
    top = numpy.sum(numpy.square(values - Predictions))
    bottom = numpy.sum(numpy.square(values - numpy.mean(values)))
    R = 1 - (top / bottom)

    return R
    


final_norm = normalize(final, continuousVar)
final_norm_cat = categorize(final_norm, final, ordinals)
features = ones(final_norm_cat)
selection = k_fold_cross(features, k)
outcome = pandas.DataFrame(normalize(final, 'mpg'))

results_beta = []
results_cost = []
results_R = []


for i in range(k):
    train = k_fold_cross2(selection[i], features)
    data_train = numpy.array(features.iloc[train,:])
    outcome1 = outcome.iloc[train,:]
    outcome_train = numpy.array(outcome1['mpg'])
    fit = regression(alpha, iterations, data_train, beta, outcome_train)
    results_beta.append([fit[0]])
    results_cost.append(fit[1])
    
    test = k_fold_cross2(selection[i], features)
    data_test = numpy.array(features.iloc[test,:])
    outcome1_test = outcome.iloc[test,:]
    outcome_test = numpy.array(outcome1_test['mpg'])
    Predictions =  numpy.dot(data_test, fit[0])
    R = R_Squared(outcome_test, Predictions)
    results_R.append(R)
    
    





