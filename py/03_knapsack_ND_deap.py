# -*- coding: utf-8 -*-
# +
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


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


n_packages = 300
n_attributes = 2
packages = generate_packages(n_packages, value_distribution_mode="squared")
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

n_packages_per_knapsack = 50

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
#
# `eaMuPlusLambda` algorithm ([documentation](https://deap.readthedocs.io/en/master/api/algo.html), [Github](https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L248))
#
# ```python
# evaluate(population)
# for g in range(ngen):
#     offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
#     evaluate(offspring)
#     population = select(population + offspring, mu)
# ```
#
# Using VarOr algorithm ([documentation](https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.varOr), [Github](https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L192))

# +
def genetic_algorithm(verbose=False):
    NGEN = 100
    MU = 50
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.3
    
    pop = toolbox.population(n=MU)
    hof = ParetoFront() # retrieve the best non dominated individuals of the evolution
    
    # Statistics created for compiling four different statistics over the generations
    stats = Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0) # axis=0: compute the statistics on each objective independently
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    #_, logbook = \
    #eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
    #                          halloffame=hof, verbose=verbose)
    
    #return pop, stats, hof, logbook
    
    _, logbook, all_generations = \
    eaMuPlusLambda_hack(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose=verbose)
    
    return pop, stats, hof, logbook, all_generations

#pop, stats, hof, logbook = genetic_algorithm()
pop, stats, hof, logbook, all_generations = genetic_algorithm()
# -



scatter_pop(all_generations[1], color="purple", label=None)
scatter_pop(all_generations[2], color="green", label=None)
scatter_pop(all_generations[3], color="blue", label=None)
scatter_pop(all_generations[3], color="red", label=None)
scatter_pop(all_generations[99], color="black", label=None)
scatter_pop(hof, color="green", label=None)

population_df(all_generations[1])

# +
pop_values = [np.sum([packages[package_id]["value"] for package_id in indv]) for indv in pop]
pop_weights = [np.sum([packages[package_id]["weight"] for package_id in indv]) for indv in pop]

df_population = pd.DataFrame({"value": pop_values, "weight": pop_weights})
df_population = pd.DataFrame(df_population.groupby(["value", "weight"]).size(), columns=["counts"]).reset_index()
df_population.index.name = "knapsack_id"

df_population.sort_values("counts", ascending=False)


# +
def scatter_pop(pop, color="purple", label=None):
    df_population = population_df(pop)
    
    marker_size = 50 * df_population["counts"]
    plt.scatter(df_population["weight"], df_population["value"], s=marker_size, alpha=0.3, color=color, label=label)

def population_df(pop):
    pop_values = [np.sum([packages[package_id]["value"] for package_id in indv]) for indv in pop]
    pop_weights = [np.sum([packages[package_id]["weight"] for package_id in indv]) for indv in pop]

    df_population = pd.DataFrame({"value": pop_values, "weight": pop_weights})
    df_population = pd.DataFrame(df_population.groupby(["value", "weight"]).size(), columns=["counts"]).reset_index()
    df_population.index.name = "knapsack_id"

    #df_population.sort_values("counts", ascending=False)
    return df_population

df_hof = population_df(hof)    
# -

df_hof

len(df_hof), len(df_population)

# +


marker_size = 50 * df_population["counts"]
plt.scatter(df_population["weight"], df_population["value"], s=marker_size, alpha=0.3, color="purple", label="Current Population")
plt.scatter(df_hof["weight"], df_hof["value"], marker="x", alpha=0.3, color="green", label="Pareto Front")
plt.legend()


# -

# # Hack
#
# Goals:
# * Improve diversity  
# * Yield all solutions
#
#
# Future work:
# * Restarting

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


# +
from deap.tools import Logbook

def eaMuPlusLambda_hack(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
        
    # --- Hack ---
    all_generations = {}
    # ------------

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # --- Hack ---
        all_generations[gen] = population + offspring
        # ------------

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, all_generations
# -

for _ in range(['a', 'b']):
    print(_)



# # References
# * [DEAP ga_knapsack](https://deap.readthedocs.io/en/master/examples/ga_knapsack.html)
# * [DEAP `creator`](https://deap.readthedocs.io/en/master/api/creator.html?highlight=creator)
# * [DEAP `base.Toolbox.register`](https://deap.readthedocs.io/en/master/api/base.html?highlight=register#deap.base.Toolbox.register)


