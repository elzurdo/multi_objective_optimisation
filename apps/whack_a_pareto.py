import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from operator import lt as less_than, gt as greater_than
from operator import (truediv as div, mul)

value_distribution_mode = st.sidebar.selectbox('Select distribution type', ['uncorrelated','squared correlation'])

feature1_default, feature2_default = "Objective 1", "Objective 2"

feature1 = st.sidebar.text_input('Change Objective 1 name', value= feature1_default)
feature2 = st.sidebar.text_input('Change Objective 2 name', value= feature2_default)

"""
# Whack a Pareto Front!

**An interactive demo to learn to identify Pareto Optimal solutions**


You are presented with distributions of solutions in an objective space, and   
the task is to correctly identify the Pareto Front of each distribution. 

"""

paretofront_file = "https://upload.wikimedia.org/wikipedia/commons/2/27/Pareto_Efficient_Frontier_1024x1024.png"

text_pareto = """
When making decision based on two or more objectives that are in conflict, 
e.g, minimising product price while maximising its value, one might be required to make a trade-off decision.  

A Pareto Front is a tool to assist in the decision process making. 

It's easiest to understand what a Pareto Front is by examining a scatter plot of 
solutions, as displayed above. 

All dots are solutions and they are color coded by the following logic, assuming that 
the objective is to maximise both *Quantity of Item 1* and *2*:  
* A solution that is better another in all objectives (in our case both dimensions) is said to *dominate* it. E.g, **N** 
dominates **K** in both objectives. Also note that **E** dominates both **N** and **K** in both objectives.
* Solutions that are dominated by at least on other solution is called a ***Dominated Solution***. These are marked 
here in gray, e.g, **K** is dominated by **N** (and **E**) and hence is labelled Dominated. Even though **N** dominates **K**, 
the fact that **E** dominates it makes **N** Dominated, too.
* A solution that is not dominated by any other solution is called a ***Non Dominated Solution*** and is marked here in red. 
E.g, we see that **E** is not dominated by any other solution. Even though **F** dominates it in the horizontal objective, 
it does not in the vertical one. The same but vice versa is in its relationship with solution **D**. 

To make sure you understand this, I suggest you scan all lettered solutions A-H, K and N and try to understand the 
color coding according to this logic.

Once this is understood we can define:  
A ***Pareto Optimal*** solution is one that is Non Dominated.  

A ***Pareto Front*** is a set of Pareto Optimal solutions.  

Pareto Optimal solutions are all considered equally optimal, i.e, no reason to prefer one over the other, 
until a subjective weighting is applied which enables their ranking according to a domain expert.   

With this definition in mind, you are challenged in this interactive demo to identify Pareto Fronts within scatter plots. 

Good luck!
"""

expander_pareto = st.beta_expander("What is a Pareto Front?")
expander_pareto.image(paretofront_file, width=400)
expander_pareto.write(text_pareto)




text_instructions = """
* Look at the distribution below     
* Fill in **Your Guess** with the number of solutions that are Pareto Optimal  
* Press the **Show me the Pareto Front!** button
* Change **Wack Number** - to get a different distribution

*More parameters may be modified in the left side bar.*
"""


expander_instructions = st.beta_expander("Instructions")
expander_instructions.write(text_instructions)


def generate_objectives(n_packages = 20,
                      weight_mean = 0.1, weight_sd = 10,
                      value_mean = 0.1, value_sd = 100,
                      value_distribution_mode = 'uncorrelated',
                      noise_factor = 1./2,
                      seed = 3141592653, visualise=True):
    """
    n_packages: int. the total number of packages to choose from
    weight_mean: float. the mean of the weight distribution
    weight_sd: float. the standard deviation of the weight distribution
    value_mean: float. when value_distribution_mode='uncorrelated': the mean of the monetary
    value_distribution_mode: str. (default: 'uncorrelated', 'squared')
    noise_factor: float. when value_distribution_mode='squared': the standard deviation of the noise introduced
    """

    np.random.seed(seed)

    weights = np.abs(np.random.normal(weight_mean, weight_sd, n_packages))

    if 'squared correlation' == value_distribution_mode:
        values =  weights ** 2
        values += values * np.random.normal(0, noise_factor, len(weights))
        values = np.abs(values)
    elif 'uncorrelated' == value_distribution_mode:
        values = np.abs(np.random.normal(value_mean, value_sd, n_packages))

    #packages = {idx: {'objective1': weights[idx], 'objective2': values[idx]} for idx in range(n_packages)}

    if visualise:
        visualise_packages(packages)


    return {feature1: weights, feature2: values}


seed = st.number_input('Wack Number: change this to see a different distribution', min_value=1, value=1, max_value=1000)

n_packages = st.sidebar.number_input('No. of Solutions', min_value=5, value=50, max_value=400)
mode_ = st.sidebar.selectbox(f'Optimisation direction of {feature1}, {feature2}', ["min, min","min, max","max, min", "max, max"])


objective_values = generate_objectives(n_packages, seed=seed, value_distribution_mode=value_distribution_mode, visualise=False)


if mode_ == "min, max":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'max'}, 'value/weight', '£/kg'
elif mode_ == "min, min":
    objective_mode, heuristic, soh_unit = {feature1: 'min', feature2: 'min'}, '1/value/weight', '1/£/kg'
elif mode_ == "max, max":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'max'}, 'value*weight',  '£*kg'
elif mode_ == "max, min":
    objective_mode, heuristic, soh_unit = {feature1: 'max', feature2: 'min'}, 'weight/value', 'kg/£'


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

guess_question = f'Your Guess: How many of the {n_packages:,} solutions are Pareto Optimal when optimising for {mode_}?'
guess = st.text_input(guess_question)


show_pareto = False
show_pareto = st.button('Show me the Pareto Front!')

if show_pareto and guess != '':
    guess_int = int(guess)

    correct = guess_int == len(pareto_idxs)

    if correct:
        guess_result = f"You are correct! The Pareto Front consists of {guess_int} solutions."
    else:
        guess_result = f"Nope ...the Pareto Front consists of {len(pareto_idxs)} solutions (some might be subtle to the human eye)."

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


vilfredo_file = "https://upload.wikimedia.org/wikipedia/commons/f/fd/Vilfredo_Pareto_1870s2.jpg"

text = """
> Vilfredo Federico Damaso Pareto born Wilfried Fritz Pareto (15 July 1848 – 19 August 1923) was an Italian civil engineer, sociologist, economist, political scientist, and philosopher. He made several important contributions to economics, particularly in the study of income distribution and in the analysis of individuals' choices. He was also responsible for popularising the use of the term "elite" in social analysis.
[Wikipedia](https://en.wikipedia.org/wiki/Vilfredo_Pareto)
"""


expander = st.beta_expander("Who was Pareto?")
expander.image(vilfredo_file, width=200)
expander.write(text)

text_intro = """
Decision making for multiple objectives is a non-trivial task, especially when they are in conflict. For example, how can one best overcome the classic trade-off between quality and cost of production, when the monetary value of quality is not defined? 


When applicable, this Pareto Optimisation method provides better results than the common practice of combining multiple parameters into a single parameter heuristic. 
The reason for this is quite simple. The single heuristic approach is like horse binders limiting the view of the solution space, whereas Pareto Optimisation enables a bird’s eye view.

Real world applications span from supply chain management, manufacturing, aircraft design, chemical engineering to land use planning and therapeutics discovery.
"""

paretofront_file = "https://upload.wikimedia.org/wikipedia/commons/2/27/Pareto_Efficient_Frontier_1024x1024.png"

expander_intro = st.beta_expander("Improved Decisions with Pareto Fronts")
expander_intro.image(paretofront_file, width=200)
expander_intro.write(text_intro)


text_more = """

Check out my tutorial at: [http://bit.ly/improved-decisions-pareto](http://bit.ly/improved-decisions-pareto)  

This free MOOC(ish) tutorial is presented on a Google Slide with embedded videos - so no adverts or promotions ... :-)   
The main content may be watched on videos for 30 minutes.  
For the hands on parts I suggest a further 60-90 minutes, or so.  


### Abstract
Decision making for multiple objectives is a non-trivial task, especially when they are in conflict. For example, how can one best overcome the classic trade-off between quality and cost of production, when the monetary value of quality is not defined? 
In this Python hands on tutorial you will learn about Pareto Fronts and how to use them in order to make better data driven decisions.  

### Description
When applicable, this Pareto Optimisation method provides better results than the common practice of combining multiple parameters into a single parameter heuristic. 
The reason for this is quite simple. The single heuristic approach is like horse binders limiting the view of the solution space, whereas Pareto Optimisation enables a bird’s eye view.

Real world applications span from supply chain management, manufacturing, aircraft design, chemical engineering to land use planning and therapeutics discovery.

### Target Audience  

This introduction is geared towards anyone who makes data driven decisions, 
 e.g, practitioners looking to improve their optimisation skills or 
 or management interested in improving their communication with data providers, such as analysts.  
 
  
 You will learn the advantages and shortcomings of the technique and be able to assess applicability for your own projects.
"""

expander_more = st.beta_expander("Where may I learn more?")
expander_more.write(text_more)


"""
Created by: [Eyal Kazin Ph.D](https://www.linkedin.com/in/eyal-kazin-0b96227a/)  
Learn more: [*"Improved Decisions with Pareto Fronts"*](http://bit.ly/improved-decisions-pareto)  
"""

