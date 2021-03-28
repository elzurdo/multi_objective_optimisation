# -*- coding: utf-8 -*-
# +
import numpy as np

import matplotlib.pyplot as plt

# +
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['figure.figsize'] = 12, 8


# +
def visualise_packages(packages):
    plt.scatter([packages[idx_package]['weight'] for idx_package in packages],
                [packages[idx_package]['value'] for idx_package in packages],
                color = 'purple', alpha=0.04
               )

    for idx_package, package in packages.items():
        plt.annotate(f'{idx_package:02.0f}', xy=(package['weight'], package['value']), fontsize=MEDIUM_SIZE)

    plt.xlabel('weight [kg]')
    plt.ylabel('value [£]')
    plt.title('Packages')

def generate_packages(n_packages = 20, 
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
        
    packages = {idx: {'weight': weights[idx], 'value': values[idx]} for idx in range(n_packages)}
    
    if visualise:
        visualise_packages(packages)

        
    return packages


n_packages = 20
n_attributes = 2
packages = generate_packages(n_packages)
# -

# # `DEAP` Setup

from deap import creator
from deap.base import Fitness, Toolbox
from deap.tools import (
    initRepeat, 
    selNSGA2,
    ParetoFront,
    Statistics
)
from deap.algorithms import eaMuPlusLambda 
import random

# +
# Creating a class call Fitness and another called Individual

# weights: -1 is for minmising and 1 is for maximising

creator.create("Fitness", Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

# +
# item_id : package (as in: an item in a knapsack) - (instead of attr_item)
# individual: knapsack
# population: population of knapsacks

n_packages_per_knapsack = 4

toolbox = Toolbox()

toolbox.register("item_id", random.randrange, n_packages)
toolbox.register("individual", initRepeat, creator.Individual, 
                 toolbox.item_id, n_packages_per_knapsack)
toolbox.register("population", initRepeat, list, toolbox.individual)
# -

print(f"toolbox.item_id selects package id,\ne.g, {toolbox.item_id()}")

print(f"toolbox.individual generates a knapsack from package_ids,\ne.g, {toolbox.individual()}")

print(f"toolbox.population generates a population of knapsacks from package_ids,\ne.g, {toolbox.population(5)}")


# # Custom Functions

# +
def eval_knapsack(individual):
    weight = 0.0
    value = 0.0
    for package_id in individual:
        weight += packages[package_id]["weight"]
        value += packages[package_id]["value"]
    #if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
    #    return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value


eval_knapsack(toolbox.individual())


# +
def crossover_set(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

knapsack1, knapsack2 = toolbox.individual(), toolbox.individual()
print(knapsack1, knapsack2)

crossover_set(knapsack1, knapsack2)


# +
def mutate_set(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(n_packages))
    return individual,

knapsack3 = toolbox.individual()
print(knapsack3)
mutate_set(knapsack3)
# -

toolbox.register("evaluate", eval_knapsack)
toolbox.register("mate", crossover_set)
toolbox.register("mutate", mutate_set)
toolbox.register("select", selNSGA2)


# # Running

# +
def genetic_algorithm(verbose=False):
    NGEN = 3
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    
    pop = toolbox.population(n=MU)
    hof = ParetoFront() # retrieve the best non dominated individuals of the evolution
    
    # Statistics created for compiling four different statistics over the generations
    stats = Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0) # axis=0: compute the statistics on each objective independently
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose=verbose)
    
    return pop, stats, hof

pop, stats, hof = genetic_algorithm()
# -

hof.keys



# # References
# * [DEAP ga_knapsack](https://deap.readthedocs.io/en/master/examples/ga_knapsack.html)
# * [DEAP `creator`](https://deap.readthedocs.io/en/master/api/creator.html?highlight=creator)
# * [DEAP `base.Toolbox.register`](https://deap.readthedocs.io/en/master/api/base.html?highlight=register#deap.base.Toolbox.register)


