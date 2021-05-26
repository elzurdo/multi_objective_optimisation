import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from operator import lt as less_than, gt as greater_than
from operator import (truediv as div, mul)

feature1, feature2 = "objective_1", "objective_2"

"""
# Wack a Pareto Front

**In interactive demo to learn to identfiy Pareto Optimal solutions**
"""

def generate_objectives(n_packages = 20,
                      weight_mean = 0.1, weight_sd = 10,
                      value_mean = 0.1, value_sd = 100,
                      value_distribution_mode = 'random',
                      noise_factor = 1./2,
                      seed = 3141592653, visualise=True):
    """
    n_packages: int. the total number of packages to choose from
    weight_mean: float. the mean of the weight distribution
    weight_sd: float. the standard deviation of the weight distribution
    value_mean: float. when value_distribution_mode='random': the mean of the monetary
    value_distribution_mode: str. (default: 'random', 'squared')
    noise_factor: float. when value_distribution_mode='squared': the standard deviation of the noise introduced
    """

    np.random.seed(seed)

    weights = np.abs(np.random.normal(weight_mean, weight_sd, n_packages))

    if 'squared' == value_distribution_mode:
        values =  weights ** 2
        values += values * np.random.normal(0, noise_factor, len(weights))
        values = np.abs(values)
    elif 'random' == value_distribution_mode:
        values = np.abs(np.random.normal(value_mean, value_sd, n_packages))

    #packages = {idx: {'objective1': weights[idx], 'objective2': values[idx]} for idx in range(n_packages)}

    if visualise:
        visualise_packages(packages)


    return {"objective_1": weights, "objective_2": values}


seed = st.number_input('Wack No. (change this to see a different distribution)', min_value=1, value=1, max_value=1000)


n_packages = st.sidebar.number_input('No. of Solutions', min_value=5, value=50, max_value=400)
mode_ = st.sidebar.selectbox(f'Optimisation {feature1}, {feature2}', ["min min","min max","max min", "max max"])



objective_values = generate_objectives(n_packages, seed=seed, value_distribution_mode="random", visualise=False)


# Objective Space declaration

# CHANGE ME!
if mode_ == "min max":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'max'}, 'value/weight', '£/kg'
elif mode_ == "min min":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'min'}, '1/value/weight', '1/£/kg'
elif mode_ == "max max":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'max'}, 'value*weight',  '£*kg'
elif mode_ == "max min":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'min'}, 'weight/value', 'kg/£'


# These objects are used to calculate the relationships between the knapsacks.

# for Single Objective Optimisation
#direction_to_multiplier = {'min': div, 'max': mul}

# for Pareto Optimal selection
mode_to_operator = {'min': less_than, 'max': greater_than}
objective_operator = {key: mode_to_operator[objective_mode[key]] for key in objective_mode.keys()}


def objectives_to_pareto_front(objective_values):
    feature1 = list(objective_values.keys())[0]
    feature2 = list(objective_values.keys())[1]

    # objective_values = {}
    # for objective in objectives:
    #    objective_values[objective] = [knapsacks[idx][objective] for idx in knapsacks]

    idxs_pareto = []

    idx_objects = np.arange(len(objective_values[feature1]))

    for idx in idx_objects:
        is_pareto = True

        this_weight = objective_values[feature1][idx]
        this_value = objective_values[feature2][idx]

        other_weights = np.array(list(objective_values[feature1][:idx]) + list(
            objective_values[feature1][idx + 1:]))
        other_values = np.array(list(objective_values[feature2][:idx]) + list(
            objective_values[feature2][idx + 1:]))

        for jdx in range(len(other_weights)):
            other_dominates = objective_operator[feature1](other_weights[jdx],
                                                           this_weight) & \
                              objective_operator[feature2](other_values[jdx],
                                                           this_value)

            if other_dominates:
                is_pareto = False
                break

        if is_pareto:
            idxs_pareto.append(idx_objects[idx])

    return idxs_pareto


pareto_idxs = objectives_to_pareto_front(objective_values)


plt.scatter(objective_values[feature1], objective_values[feature2], s=10, alpha=0.7, color='purple')

guess_question = f'How many solutions are Pareto Optimal when optimising for {mode_}?'
guess = st.text_input(guess_question)



show_pareto = False
show_pareto = st.button('Show me the Pareto Front!')

if show_pareto and guess != '':
    guess_int = int(guess)

    correct = guess_int == len(pareto_idxs)

    if correct:
        guess_result = f"You are correct! The Pareto Front consists of {guess_int} solutions"
    else:
        guess_result = f"Nope ...the Pareto Front consists of {len(pareto_idxs)} solutions (some might be subtle of the human eye)"

    f"""{guess_result}"""


    plt.scatter([val for idx, val in enumerate(objective_values[feature1]) if idx in pareto_idxs],
                [val for idx, val in enumerate(objective_values[feature2]) if idx in pareto_idxs],
                marker='x', s=100, linewidth=4, color='green')
elif show_pareto and guess == '':
    """
    You need to guess the number of solutions before I reveal ... ;-) 
    """

plt.xlabel(feature1)
plt.ylabel(feature2)


st.pyplot(plt.gcf())


wack_file = "https://www.connections.com/hs-fs/hubfs/WhackAMoleTech.jpg"
st.image(wack_file) # , width=1000)
