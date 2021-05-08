# Multi Objective Optimisation
This repository is dedicated to tutorials to learn and practice optimising for multiple parameters 
using Pareto Fronts.

This tutorial will be presented on March 13th 2021 in PyCon USA ([abstract](https://us.pycon.org/2021/schedule/presentation/39/))  

<img src="https://tinyurl.com/vrdj29w9" width="300">

It has been delivered in PyData Global 2020 ([abstract](https://global.pydata.org/talks/82)).  
<img src="https://global.pydata.org/assets/images/logo.png" width="300">


## Tutorial

Welcome and thanks for attending this tutorial on multi-objective optimisation!  

This  tutorial was created for the NumFocus Academia who host it [here](https://academy.numfocus.org/about-course/?introduction-multi-objective-optimisation).  
<img src="https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png" width="300">
  
You can also access all the material in this repository, including video recordings.

## Abstract
Optimising for multiple objectives is a non-trivial task, especially when they are in conflict. For example how can one best overcome the classic trade-off between quality and cost of production, when the monetary value of quality is not defined?  In this hands-on Python tutorial you will learn about Pareto Fronts and use them to optimise for multiple objectives simultaneously.

This hands-on tutorial is geared towards anyone interested in improving their optimisation skills (e.g, analysts, scientists, engineers, economists), in which you will learn and implement

The only requirements are basic usage of `numpy` and `matplotlib`. The maths required is highschool level. 

Here you will find Jupyter notebooks with which you will apply lessons and tools learned to the simple [Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). 
You will program for filling a bag with packages with the objective of minimising the bag weight while maximising its content value. 

<img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Knapsack.svg" width="200">



## Description
Multi-Objective Optimisation, also known as Pareto Optimisation, is a method to optimise for multiple parameters simultaneously. When applicable, this method provides better results than the common practice of combining multiple parameters into a single parameter heuristic. The reason for this is quite simple. The single heuristic approach is like horse binders limiting the view of the solution space, whereas Pareto Optimisation enables a bird’s eye view

Real world applications span from supply chain management, manufacturing, aircraft design to land use planning. For example when developing therapeutics, Pareto optimisation may help a biologist maximise protein properties like effectiveness and manufacturability while simultaneously minimising toxicity.

This hands-on tutorial is geared towards anyone interested in improving their optimisation skills (e.g, analysts, scientists, engineers, economists), in which you will learn and implement:

*  The limitations of the common practice of combining multiple parameters into one heuristic.
*  *Pareto Fronts*: the notion in which there may be a set of trade-off solutions which are considered equally optimal.   
* Application of the Pareto Front method to Evolutionary Algorithms to find optimal solutions in an intractable search space.
* Applicability in the real world

We will use Python and work in a Jupyter notebook environment. 



  


### Outline 

If you know about Pareto Fronts and Genetic Algorithms, feel free to jump right into the notebooks.      

Otherwise you might benefit from the context given the videos. The total time of the intro videos is 30 minutes.  
The hands-on videos are roughly 60 minutes, in which I guide you through the notebooks. 
Since the notebooks are completely annotated, you can feel free to skip the hands-on videos.  

 
* Welcome and Intro (5 minute [video](http://bit.ly/moo-youtube-intro))
* Introduction to Pareto Fronts  (13 minutes in 2 videos: [1](http://bit.ly/moo-youtube-pareto1), [2](http://bit.ly/moo-youtube-pareto2))
* Hands On: Pareto Fronts (25 minutes in 3 videos: [1](https://bit.ly/moo-youtube-handson-pf1), [2](https://bit.ly/moo-youtube-handson-pf2), [3](https://bit.ly/moo-youtube-handson-pf3)) 
<a href="https://bit.ly/pareto-front-colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="First: Open In Colab"/></a>
* Decision and Objective Space (3 minute [video](https://bit.ly/moo-youtube-decision-space))
* Introduction to Pareto Optimisation with Genetic Algorithms (12 minutes in 2 videos: [1](https://bit.ly/moo-youtube-ga1), [2](https://bit.ly/moo-youtube-ga2))
* Hands On: Pareto Optimisation with Genetic Algorithms (35 minutes in 5 videos [1](https://bit.ly/moo-youtube-handson-ga1), [2](https://bit.ly/moo-youtube-handson-ga2), [3](https://bit.ly/moo-youtube-handson-ga3), [4](https://bit.ly/moo-youtube-handson-ga4), [5](https://bit.ly/moo-youtube-handson-ga5)) <a href="https://bit.ly/genetic-algorithm-colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="First: Open In Colab"/></a>
* Summary and Discussion (5 minute [video](https://bit.ly/moo-youtube-summary))


### Main Take Aways From Hands The On Sessions

The notebooks will help you learn to solve for multiple objective optimisation problems and visualise results.


* In `01_knapsack 2D_exhaustive.ipynb` you will optimise for the knapsack problem in an exhaustive solution space. Here you will learn:
    * The limitations of the commonly practiced Single Objective Optimisation.  
    * Pareto Fronts: identifying by eye Pareto optimal solutions within a distribution as well as pseudocode for selecting these *Non-Dominated* solutions.
    * Optimisation concepts *Decision Spaces* and *Objective Spaces*. These concepts are useful to determine the applicability of Pareto Optimisation.
* In `02_knapsack_2D_stochastic.ipynb` you optimise for the knapsack problem in an intractable search space. Here you will learn:
    * Stochasticity and Pareto front approximations 
    * Genetic Algorithm basics with an emphasis on Pareto fronts as the selection function.   
    * To track the learning progress of a Genetic Algorithm by means of the evolution of Pareto front approximations.
    * Improving population diversity by niching.
    
## More Resources
For those interested in material before the tutorial, I kindly refer you to:   
* [DEAP](https://deap.readthedocs.io/en/master/) - a package for quick prototyping of evolutionary algorithms  
* [Pymoo](https://pymoo.org/) - a package with multi-objective optimization algorithms 
*  [PyData London talk](https://www.youtube.com/watch?v=_9x4cmQWZ6g) and [its slides](https://drive.google.com/file/d/1UMPGkeA_Tsc5PYWktpjquhIhOa9OD8Gb/view).  
*  For those who want extra curriculum I highly suggest any paper on the topic by [Eckart Zitzler](https://scholar.google.ch/citations?user=GW8tPekAAAAJ&hl=de). His Ph.D thesis is an excellent read.  
* Pareto Front example in an [Excel workbook](http://www.vertexvortex.com/r/excel/Pareto_Frontier.xlsx) (provided by [vertexvortex from Reddit](https://www.reddit.com/r/excel/comments/104fcb/pareto_frontier/)). 
