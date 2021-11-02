# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 23
STEADY_STATE = 1000
POPULATION_SIZE = 50
ELITISM_SIZE = 3
PROB_XOVER = .5
PROB_MUTATION = .5


class Tsp:
    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(
                c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def find_best_individual(problem, population):
    population_fitness = [problem.evaluate_solution(i) for i in population]
    minIndex = np.argmin(population_fitness)
    return (population[minIndex], population_fitness[minIndex])


def initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        p = np.array(range(NUM_CITIES))
        np.random.shuffle(p)
        population.append(p)

    return population


def select_individuals(problem, population, population_size, elite_size):
    elite_individuals = []
    selected_individuals = []

    population_fitness = np.array(
        [problem.evaluate_solution(i) for i in population])

    sorted_indexes = np.argsort(population_fitness)
    sorted_population = np.array(population)[sorted_indexes]
    sorted_population_fitness = population_fitness[sorted_indexes]

    # Select elite individuals
    for i in range(elite_size):
        elite_individuals.append(sorted_population[i])

    # Calculate base selection probability
    sorted_population_fitness_perc = 1 - \
        (np.cumsum(sorted_population_fitness)/np.sum(sorted_population_fitness))

    # Run the fitness roulette
    i = 0
    count = 0
    while count < population_size-elite_size:
        selection_threshold = np.random.random()
        if sorted_population_fitness_perc[i] >= selection_threshold:
            selected_individuals.append(sorted_population[i])
            count += 1
        i += 1
        if i >= population_size-elite_size:
            i = 0

    return elite_individuals, selected_individuals


def mutate_individual(individual):
    new_individual = np.copy(individual)
    i1 = np.random.randint(0, individual.shape[0])
    i2 = np.random.randint(0, individual.shape[0])
    temp = new_individual[i1]
    new_individual[i1] = new_individual[i2]
    new_individual[i2] = temp
    return new_individual


def mutate_population(population):
    mutated_population = []
    for i in range(len(population)):
        individual = population[i]

        should_mutate = np.random.uniform()
        if should_mutate >= 1-PROB_MUTATION:
            individual = mutate_individual(individual)

        mutated_population.append(individual)

    return mutated_population


def crossover_individuals(indA, indB):

    cIndA = []
    cIndB = []

    ga = np.random.randint(NUM_CITIES)
    gb = np.random.randint(NUM_CITIES)

    gmin = min(ga, gb)
    gmax = max(ga, gb)

    seqA = indA[gmin:gmax]
    seqB = indB[gmin:gmax]

    cIndA = [i for i in indA if i not in seqB]
    cIndB = [i for i in indB if i not in seqA]

    for i in range(gmin, gmax):
        cIndA.insert(i, seqB[i-gmin])
        cIndB.insert(i, seqA[i-gmin])

    return np.array([cIndA, cIndB])


def crossover_population(population):
    crossover_population = []
    i = 0
    while i < len(population):
        individual = population[i]

        should_crossover = np.random.uniform()
        if should_crossover >= 1-PROB_XOVER:
            another_individual = population[np.random.randint(len(population))]

            crossedover_individuals = crossover_individuals(
                individual, another_individual)

            crossover_population.extend(crossedover_individuals)
            i += 2
            continue

        crossover_population.append(individual)
        i += 1

    return crossover_population[0:POPULATION_SIZE]


def main():
    problem = Tsp(NUM_CITIES)

    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    solution_cost = problem.evaluate_solution(solution)
    # problem.plot(solution)
    print(solution_cost)

    history = [(0, solution_cost)]
    steady_state = 0
    step = 0

    # Build initial population
    population = initial_population()
    best_individual, best_individual_fitness = find_best_individual(
        problem, population)
    print(best_individual, best_individual_fitness)

    # Start evolving
    # Stop when no optimization occurs after STEADY_STATE iterations
    while steady_state < STEADY_STATE:
        step += 1
        steady_state += 1

        new_population = []

        # Select elite and individuals for the new generation
        elite_individuals, selected_individuals = select_individuals(
            problem, population, POPULATION_SIZE, ELITISM_SIZE)

        # Execute the crossover between the selected individuals
        selected_individuals = crossover_population(selected_individuals)

        # Mutate the selected individuals
        selected_individuals = mutate_population(selected_individuals)

        new_population.extend(elite_individuals)
        new_population.extend(selected_individuals)

        # Calculate the fitness for the best generation's individual
        new_best_individual, new_best_individual_fitness = find_best_individual(
            problem, new_population)

        # If the new fitness score is better then the previous, update the best individual and
        # reset the steady state counter
        if new_best_individual_fitness < best_individual_fitness:
            population = new_population
            best_individual, best_individual_fitness = new_best_individual, new_best_individual_fitness
            print(best_individual, best_individual_fitness)
            history.append((step, best_individual_fitness))
            steady_state = 0

    problem.plot(best_individual)


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
