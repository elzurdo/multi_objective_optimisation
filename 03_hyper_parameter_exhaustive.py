# -*- coding: utf-8 -*-
# +
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score, precision_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# +
# Visualising
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

FIG_WIDTH, FIG_HEIGHT = 8, 6

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.figsize"] = FIG_WIDTH, FIG_HEIGHT
plt.rcParams["hatch.linewidth"] = 0.2

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# -

# # Data

# +
X_all, y_all = datasets.load_diabetes(return_X_y=True,as_frame=True)
y_all = pd.DataFrame(y_all)

col_target = "under_median"

X_all.index.name = 'idx'
y_all.index.name = 'idx'
y_all[col_target] = (y_all['target'] <  y_all['target'].median()).astype(int)

plt.hist(y_all["target"], color="purple", alpha=0.4)

print(y_all["target"].describe())
print('-' * 20)
print(y_all[col_target].value_counts())



print(X_all.shape)
X_all.head(4)
# -

# # Vanilla ML

# +
# preprocessing

test_size = 0.2
seed = 1

X_train, X_test, y_train, y_test  = train_test_split(X_all, y_all[col_target], test_size=test_size, random_state=seed)

print(X_train.shape, X_test.shape)
X_train.head(2)

# +
# classifier fit and score
seed = 5
model = RandomForestClassifier(n_estimators=20, random_state=seed)

model.fit(X_train, y_train)

accuracy_vanilla = model.score(X_test, y_test)
accuracy_vanilla

# +
df_y_test = pd.DataFrame({'truth': y_test})

df_y_test['prob_1'] = list(map(lambda x: x[1], model.predict_proba(X_test.loc[df_y_test.index])))

df_y_test.head(2)
# -

bins = np.arange(0, 1, 0.05)
df_y_test.query('truth == 0')['prob_1'].hist(bins=bins, color='red', alpha=0.7, hatch="/", label="target=0")
df_y_test.query('truth == 1')['prob_1'].hist(bins=bins, color='green', alpha=0.7, label="target=1")
plt.legend()

# +
thresh = 0.5
predictions = df_y_test['prob_1'] >= thresh

precision_vanilla = precision_score(df_y_test['truth'], predictions)
recall_vanilla = recall_score(df_y_test['truth'], predictions)

# ---
thresh_values = np.arange(0.1, 1., 0.02)
recalls_vanilla = {}
precisions_vanilla = {}


for thresh in thresh_values:
    predictions = df_y_test['prob_1'] >= thresh
    recalls_vanilla[thresh] = recall_score(df_y_test['truth'], predictions)
    precisions_vanilla[thresh] = precision_score(df_y_test['truth'], predictions)
    
    
plt.plot(recalls_vanilla.values(), precisions_vanilla.values(), '-o', color="gray", alpha=0.7)
plt.scatter(recall_vanilla, precision_vanilla, color="green", s=200)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim(0.,1)
plt.ylim(0.,1)
# -

# # Optimisation

# ## Decision Space

# +
n_estimators = [2, 5, 10, 15, 20, 30, 50, 100, 200, 300] #list(_n_estimators) + list(_n_estimators * 10) + list(_n_estimators * 100) + [1000, 2000, 3000]
max_depth = [None, 3, 5, 10, 15, 20] #[None, 2, 3, 5, 7, 10, 15, 20]
#thresh_values = np.arange(0.1, 1., 0.1) #np.arange(0.1, 1., 0.02)

n_combinations = len(max_depth) * len(n_estimators) # len(thresh_values) * 
n_combinations

# +
from itertools import product 

decision_combinations = product(n_estimators, max_depth) # , thresh_values)

# +
models = {}
for idx_model, param_values in enumerate(decision_combinations):
    models[idx_model] = {}
    models[idx_model]['params'] = param_values
    
print('This is what the first four models look like:')
{idx_model: model for idx_model, model in models.items() if idx_model<4}
# -

print(f'The number of models {len(models):,}\nshould be the same as the combination that we previously calculated ({int(n_combinations):,})')


# # Mapping To Objective Space

# +
# TODO: delete probs_to_precision_recall. Not required
def probs_to_precision_recall(truth, probs, thresh_values):
    recall_values = {}
    precision_values = {}

    for thresh in thresh_values:
        predictions = probs >= thresh
        recall_values[thresh] = recall_score(truth, predictions)
        precision_values[thresh] = precision_score(truth, predictions)
        
    return precision_values , recall_values


def decisions_to_objectives(decisions, seed=None, thresh=0.5):
    n_estimators = decisions[0]
    max_depth = decisions[1]
    # thresh_value = decisions[2]
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)
    
    probs = np.array(list(map(lambda x: x[1], model.predict_proba(X_test.loc[df_y_test.index]))))
    
    #precision_values, recall_values = probs_to_precision_recall(y_test.values, probs, thresh_values)
    #accuracy_vanilla = model.score(X_test, y_test)
    
    predictions = probs >= thresh

    precision_ = precision_score(df_y_test['truth'], predictions)
    recall_ = recall_score(df_y_test['truth'], predictions)
    
    #return precision_values[thresh_value], recall_values[thresh_value]
    return precision_, recall_


# +
#for idx_model in models:
#    models[idx_model]['precision'], models[idx_model]['recall'] = decisions_to_objectives(models[idx_model]['params'], seed=seed)    
    
for idx_model in models:
    models[idx_model]['precision'], models[idx_model]['recall'] = decisions_to_objectives(models[idx_model]['params'], seed=seed)    

# +
precision_values = [model_params["precision"] for idx_model, model_params in models.items()]
recall_values = [model_params["recall"] for idx_model, model_params in models.items()]

plt.scatter(recall_values, precision_values, color="gray")
#plt.scatter(accuracy_vanilla, np.max(list(precision_vanilla.values())))
plt.scatter(recall_vanilla, precision_vanilla, color="green", s=200)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim(0.,1)
plt.ylim(0.,1)
# -

accuracy_vanilla



# +
recall_values = [model_params["recall"] for idx_model, model_params in models.items()]
precision_values = [model_params["precision"] for idx_model, model_params in models.items()]

plt.scatter(recall_values, precision_values)
plt.xlabel("recall")
plt.ylabel("precision")
plt.plot(recall_vanilla.values(), precision_vanilla.values(), '-o', label="vanilla", color="gray")
# -

# # Old

# +
X_all, y_all = datasets.load_digits(n_class=10, return_X_y=True, as_frame=True)
y_all = pd.DataFrame(y_all)


X_all.index.name = 'idx'
y_all.index.name = 'idx'

col_target = 'is_even'
y_all[col_target] = (y_all['target'] % 2 == 0).astype(int)

print(y_all['target'].value_counts())
print('-' * 20)
print(y_all[col_target].value_counts())

print(X_all.shape)
X_all.head(4)

# +
test_size = 0.2
seed = 1

X_train, X_test, y_train, y_test  = train_test_split(X_all, y_all[col_target], test_size=test_size, random_state=seed)

X_train.head(2)

# +
seed = 5
model = RandomForestClassifier(n_estimators=20, random_state=seed)

model.fit(X_train, y_train)
# -

model.score(X_test, y_test)

# +
df_feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index(name="fraction").set_index("index")
df_feature_importances.index.name = "feature"

df_feature_importances["cumsum"] = np.cumsum(df_feature_importances["fraction"])

df_feature_importances.head(10)

# +
information_limit = 0.98

features_limited = df_feature_importances[df_feature_importances["cumsum"] >= information_limit].index.tolist()

print(len(features_limited))
df_feature_importances.loc[features_limited]["fraction"].sum()

# +
model_limited = RandomForestClassifier(n_estimators=20, random_state=seed)

model_limited.fit(X_train[features_limited], y_train)
# -

model_limited.score(X_test[features_limited], y_test)

len(model_limited.feature_importances_)

# +
df_y_test = pd.DataFrame({'truth': y_test})

df_y_test['prob_1'] = list(map(lambda x: x[1], model_limited.predict_proba(X_test[features_limited].loc[df_y_test.index])))

df_y_test.head(2)

# +
thresh_values = np.arange(0.1, 1., 0.02)
recall_values = {}
precision_values = {}


for thresh in thresh_values:
    predictions = df_y_test['prob_1'] >= thresh
    recall_values[thresh] = recall_score(df_y_test['truth'], predictions)
    precision_values[thresh] = precision_score(df_y_test['truth'], predictions)
# -



plt.plot(recall_values.values(), precision_values.values(), '-o')
#plt.xlim(0.9, 1.)
#plt.ylim(0.9, 1.)

df_y_test.query('truth == 0')['prob_1'].hist(bins=40, color='red', alpha=0.7)
df_y_test.query('truth == 1')['prob_1'].hist(bins=40, color='green', alpha=0.7)

# # Decision Space

# +
#model = RandomForestClassifier(n_estimators=20, random_state=seed)

_n_estimators = np.array([1, 2, 3, 5, 7])

n_estimators = [2, 10, 30, 100 ] #list(_n_estimators) + list(_n_estimators * 10) + list(_n_estimators * 100) + [1000, 2000, 3000]
max_depth = [None, 3, 10] #[None, 2, 3, 5, 7, 10, 15, 20]
thresh_values = np.arange(0.1, 1., 0.1) #np.arange(0.1, 1., 0.02)

n_combinations = len(thresh_values) * len(max_depth) * len(n_estimators)
n_combinations

# +
from itertools import product 

decision_combinations = product(n_estimators, max_depth, thresh_values)

# +

models = {}
for idx_model, param_values in enumerate(decision_combinations):
    models[idx_model] = {}
    models[idx_model]['params'] = param_values
    
print('This is what the first four knapsacks look like:')
{idx_model: model for idx_model, model in models.items() if idx_model<4}
# -

print(f'The number of knapsacks {len(models):,}\nshould be the same as the combination that we previously calculated ({int(n_combinations):,})')


# ## Objective Space Mapping

# +
def probs_to_precision_recall(truth, probs, thresh_values):
    recall_values = {}
    precision_values = {}

    for thresh in thresh_values:
        predictions = probs >= thresh
        recall_values[thresh] = recall_score(truth, predictions)
        precision_values[thresh] = precision_score(truth, predictions)
        
    return precision_values , recall_values


def decisions_to_objectives(decisions, seed=None):
    n_estimators = decisions[0]
    max_depth = decisions[1]
    thresh_value = decisions[2]
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)
    
    probs = list(map(lambda x: x[1], model.predict_proba(X_test.loc[df_y_test.index])))
    
    precision_values, recall_values = probs_to_precision_recall(y_test.values, probs, [thresh_value])
    
    return precision_values[thresh_value], recall_values[thresh_value]
    
    


# +
for idx_model in models:
    models[idx_model]['precision'], models[idx_model]['recall'] = decisions_to_objectives(models[idx_model]['params'], seed=seed)    
        
print('This is what the first four knapsacks look like now that we added the Objective Space:')
{idx_knapsack: knapsack for idx_knapsack, knapsack in knapsacks.items() if idx_knapsack<4}
# -

print('This is what the first four models look like now that we added the Objective Space:')
{idx_model: model for idx_model, model in models.items() if idx_model<4}

# +
# visualising the Objective Space

plt.scatter([models[idx]['recall'] for idx in models], 
            [models[idx]['precision'] for idx in models], 
            alpha=0.4, color='purple')

plt.xlabel('recall')
plt.ylabel('precision')
#plt.title('Knapsacks')
plt.xlim(0.9, 1.)
plt.ylim(0.9, 1.)
# -

# # Optimisation Objective

from operator import lt as less_than, gt as greater_than
from operator import (truediv as div, mul)

# +
# Objective Space declaration

# CHANGE ME!
#objective_mode, heuristic, soh_unit = {'weight': 'min', 'value': 'max'}, 'value/weight', '£/kg'
#objective_mode, heuristic, soh_unit = {'weight': 'min', 'value': 'min'}, '1/value/weight', '1/£/kg'
objective_mode, heuristic, soh_unit = {'precision': 'max', 'recall': 'max'}, 'value*weight',  '£*kg'
#objective_mode, heuristic, soh_unit = {'weight': 'max', 'value': 'min'}, 'weight/value', 'kg/£'

# +
# These objects are used to calculate the relationships between the knapsacks.

# for Single Objective Optimisation
direction_to_multiplier = {'min': div, 'max': mul}

# for Pareto Optimal selection
mode_to_operator = {'min': less_than, 'max': greater_than}
objective_operator = {key: mode_to_operator[objective_mode[key]] for key in objective_mode.keys()}


# -

# # Single Objective Optimisation F1 Score

# +
# calculating the single objective heuristic for each knapsack (soh) 

def model_f1_score(model):
    # generalising 
    # (weight,value) : returns single-objective-heuristic
    # (↓, ↑) : returns knapsack['value'] / knapsack['weight'] 
    # (↑, ↑) : returns knapsack['value'] * knapsack['weight'] 
    # etc.
    
    #value_heuristic    = direction_to_multiplier[objective_mode['value']](1, knapsack['value'])
    #weight_heuristic   = direction_to_multiplier[objective_mode['weight']](1, knapsack['weight'])
    #return  value_heuristic * weight_heuristic
    return 2 * model['precision'] * model['recall'] / (model['precision'] + model['recall'])
    
    

models_soh = {idx: model_f1_score(model) for idx, model in models.items()}

# +
# visualising the histogram of all the single objective heuristic values

plt.hist(list(models_soh.values()), alpha=0.4, color='purple', bins=40)
#plt.xlabel(f'{heuristic} [{soh_unit}]')
#plt.title('Knapsack Single Objective Heuristic')
#plt.yscale('log')
#plt.ylabel('Knapsacks')
pass

# +
# identifying the knapsack with the highest soh.

# soh - single objective heuristic
# soo - single objective optimal

max_soh = -1
imodel_soo = None

for imodel, soh in models_soh.items():
    if soh > max_soh:
        imodel_soo = int(imodel)
        max_soh = float(soh)
        
print(f'For Single Objective Optimisation \nmodel {imodel_soo} is considered "optimal" with a heuristic value of {max_soh:0.3f}.')

# +
# # visualising the weight-value distribution color coded by the heuristic

# plt.scatter(knapsacks[iknapsack_soo]['weight'], knapsacks[iknapsack_soo]['value'], 
#             marker='x', s=100, alpha=0.9, label=f'{heuristic} optimal', color='black', linewidth=3)

# plt.scatter([knapsacks[idx]['weight'] for idx in knapsacks], 
#             [knapsacks[idx]['value'] for idx in knapsacks], 
#             alpha=0.7, c=np.array(list(knapsacks_soh.values())), cmap='viridis')

# plt.xlabel('weight [kg]')
# plt.ylabel('value [£]')
# plt.title('Knapsacks')
# plt.colorbar(label=f'{heuristic} [{soh_unit}]')
# plt.legend()
# -

# # Pareto Optimal Solutions

# +
# setting up the environment for the calculation of the Pareto Front
# objective_values contains the values of the objectives of the knapsacks in the same order of the knapsacks object

objective_values = {}

for objective in ['precision', 'recall']:
    objective_values[objective] = [models[idx][objective] for idx in models]

objective_values.keys()

# +
# The Pareto Front calculation

idxs_pareto = []  # stores the indices of the Pareto optimal knapsacks

for idx in range(len(objective_values[objective])):
    is_pareto = True  #  we assume on Pareto Front until proven otherwise
    
    # objective values of this knapsack
    this_weight = objective_values['precision'][idx]
    this_value = objective_values['recall'][idx]
    
    # objective values of all the other knapsacks
    other_weights = np.array(objective_values['precision'][:idx] + objective_values['precision'][idx + 1:])
    other_values = np.array(objective_values['recall'][:idx] + objective_values['recall'][idx + 1:])
    
    for jdx in range(len(other_weights)):
        other_dominates = objective_operator['precision'](other_weights[jdx], this_weight) & objective_operator['recall'](other_values[jdx], this_value)   
        
        if other_dominates:
            #  knapsack dominated by at least another, hence not pareto optimal.
            is_pareto = False
            break  #  no need to compare with the rest of the other knapsacks
            
    if is_pareto:
        idxs_pareto.append(idx)

# +
# visualising all the solutions with an emphasis on the Pareto Front

plt.scatter([models[idx]['recall'] for idx in models], [models[idx]['precision'] for idx in models], alpha=0.1, color='purple')
plt.scatter([models[idx]['recall'] for idx in idxs_pareto], [models[idx]['precision'] for idx in idxs_pareto], 
            marker='x', s=100, linewidth=4, color='green', label='Pareto Front')

plt.xlabel('weight [kg]')
plt.ylabel('value [£]')
#plt.title('Knapsacks')
plt.legend()
# -


