# Traveling Salesman Problem

@author: Marco Bramini (s285913)  
(based on the template provided by Prof. Giovanni Squillero)

An implementation of a TSP solver based on a genetic algorithm.

For each generation:

- **Directly copies elites individuals into the new generation** (the number of elite individuals is controlled by the constant ELITISM_SIZE).
- **Selects the parents for the new generation using a fitness proportional roulette wheel strategy** (the POPULATION_SIZE constant controls the population size; selected individuals will be POPULATION_SIZE-ELITISM_SIZE).
- **Executes the crossover between parents and generates new individuals** (the probability of the crossover is controlled by the constant XOVER_PROB).
- **Mutates the new individuals** (the probability of the mutation is controlled by the constant MUTATION_PROB).
- **Checks the fitness of the best individual and stops if no improvement has occurred from a number of steps** (the number of steps is controlled by the constant STEADY_STATE).
