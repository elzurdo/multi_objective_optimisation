# Multi Objective Optimisation
This repository is dedicated to tutorials to learn and practice using Pareto Fronts 
as a means for Multi-Objective Optimisation.

## PyData Global 2020
<img src="https://global.pydata.org/assets/images/logo.png" width="300">

Welcome and thanks for attending this tutorial session!    
If you have just listened to the intro video feel free to dive into notebook via the Colab buttons in the Outline section.  
Otherwise, you might want to go over the TL;DR and Motivation sections to have a better understanding of what to expect from this tutorial.

### Outline
* Welcome and Intro (5 minutes)
* Introduction to Pareto Front Optimisation (15 minutes)
* Hands On: Pareto Front Optimisation (30 minutes) 
<a href="https://colab.research.google.com/github/elzurdo/multi_objective_optimisation/blob/master/01_knapsack%202D_exhaustive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="First: Open In Colab"/></a>
* Introduction to applying Multi-Objective Optimisation to Genetic Algorithms (20 minutes)
* Hands On: applying Multi-Objective Optimisation to Genetic Algorithms (40 minutes) <a href="https://colab.research.google.com/github/elzurdo/multi_objective_optimisation/blob/master/02_knapsack_2D_stochastic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="First: Open In Colab"/></a>
* Summary and Discussion (10 minutes)

### Hands On Instructions

These notebooks will help you learn to solve for multiple objective optimisation problems and visualise results.

* In `01_knapsack 2D_exhaustive.ipynb` you will optimise for the knapsack problem in an exhaustive solution space. Here you will learn:
    * about the main limitation of 1D optimisation: a reduced search space.
    * what Pareto Fronts are and how it is used to overcome 1D limitations
* In `02_knapsack_2D_stochastic.ipynb` you optimise for the knapsack problem in an intractable search space. Here you will learn:
    * to compare Pareto Fronts of multiple instances 
    * to apply Pareto techniques to Genetic Algorithms

## TL;DR
Optimising for multiple objectives is a non-trivial task, especially when they are in conflict. For example how can one best overcome the classic trade-off between quality and cost of production, when the monetary value of quality is not defined?  In this hands-on Python tutorial you will learn about Pareto Fronts and use them to optimise for multiple objectives simultaneously.

This hands-on tutorial is geared towards anyone interested in improving their optimisation skills (e.g, analysts, scientists, engineers, economists), in which you will learn and implement

The only requirements are basic usage of `numpy` and `matplotlib`. The maths required is highschool level. 

Here you will find Jupyter notebooks with which you will apply lessons and tools learned to the simple [Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). 
You will program for filling a bag with packages with the objective of minimising the bag weight while maximising its content value. 

<img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Knapsack.svg" width="200">



## Motivation
Multi-Objective Optimisation, also known as Pareto Optimisation, is a method to optimise for multiple parameters simultaneously. When applicable, this method provides better results than the common practice of combining multiple parameters into a single parameter heuristic. The reason for this is quite simple. The single heuristic approach is like horse binders limiting the view of the solution space, whereas Pareto Optimisation enables a bird’s eye view

Real world applications span from supply chain management, manufacturing, aircraft design to land use planning. For example when developing therapeutics, Pareto optimisation may help a biologist maximise protein properties like effectiveness and manufacturability while simultaneously minimising toxicity.

This hands-on tutorial is geared towards anyone interested in improving their optimisation skills (e.g, analysts, scientists, engineers, economists), in which you will learn and implement:

*  The limitations of the common practice of combining multiple parameters into one heuristic.
*  *Pareto Fronts*: the notion in which there may be a set of trade-off solutions which are considered equally optimal.   
* Application of the Pareto Front method to Evolutionary Algorithms to find optimal solutions in an intractable search space.
* Applicability in the real world

We will use Python and work in a Jupyter notebook environment. 





## More Resources
For those interested in material before the tutorial, I kindly refer you to:   
* [DEAP](https://deap.readthedocs.io/en/master/) - a package for quick prototyping of evolutionary algorithms  
* [Pymoo](https://pymoo.org/) - a package with multi-objective optimization algorithms 
*  [PyData London talk](https://www.youtube.com/watch?v=_9x4cmQWZ6g) and [its slides](https://drive.google.com/file/d/1UMPGkeA_Tsc5PYWktpjquhIhOa9OD8Gb/view).  
*  For those who want extra curriculum I highly suggest any paper on the topic by [Eckart Zitzler](https://scholar.google.ch/citations?user=GW8tPekAAAAJ&hl=de). His Ph.D thesis is an excellent read.
