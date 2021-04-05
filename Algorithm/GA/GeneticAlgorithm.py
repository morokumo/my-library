import random
from copy import copy
from typing import List
import numpy as np


class GeneticAlgorithm:
    def __init__(self,
                 population: int,
                 size: int,
                 n_range: int,
                 evaluation_functions: List[object],
                 evaluation_factors: List[float],
                 copy_probabilities: float,
                 crossover_probabilities: float,
                 mutation_probabilities: float,
                 tournament_size: int = 5,
                 objective_data: List[int] = None,
                 ):

        assert copy_probabilities + crossover_probabilities + mutation_probabilities == 1, (
            copy_probabilities, crossover_probabilities, mutation_probabilities)

        self.population = population
        self.size = size
        self.n_range = n_range
        self.tournament_size = tournament_size
        self.objective_data = objective_data

        assert len(self.objective_data) == self.size, (len(self.objective_data), self.size)

        self.copy_probabilities = copy_probabilities
        self.crossover_probabilities = crossover_probabilities
        self.mutation_probabilities = mutation_probabilities

        self.now_generation = []
        self.next_generation = []
        self.generate()
        self.evaluation_functions = evaluation_functions
        self.evaluation_factors = evaluation_factors
        self.fitness_values = []
        self.best_population = None
        self.best_fitness_value = 0

    def generate(self) -> None:
        for _ in range(self.population):
            self.now_generation.append([random.randint(0, self.n_range) for _ in range(self.size)])

    def train(self, epoch) -> (float, List[int]):
        for e in range(epoch):
            self.evolution()
            print(f'epoch:{e} | max_loss={max(self.fitness_values)}   min_loss={min(self.fitness_values)}')

        self.calc_fitness()
        for index in range(self.population):
            if self.fitness_values[index] > self.best_fitness_value:
                self.best_fitness_value = copy(self.fitness_values[index])
                self.best_population = copy(self.now_generation[index])
        return self.best_fitness_value, self.best_population,

    def calc_fitness(self) -> None:
        self.fitness_values = []
        for index in range(self.population):
            fitness_value = 0.0
            if self.objective_data is None:
                for factor, evaluation_function in zip(self.evaluation_factors, self.evaluation_functions):
                    fitness_value += factor * evaluation_function(self.now_generation[index])
            else:
                for factor, evaluation_function in zip(self.evaluation_factors, self.evaluation_functions):
                    fitness_value += factor * evaluation_function(self.now_generation[index], self.objective_data)
            self.fitness_values.append(fitness_value)

    def evolution(self) -> None:
        assert len(self.next_generation) == 0
        self.calc_fitness()
        for _ in range(0, self.population, 2):
            gene1 = self.choice()
            gene2 = self.choice()
            gene1, gene2 = self.crossover(gene1, gene2)
            gene1 = self.mutation(gene1)
            gene2 = self.mutation(gene2)
            self.next_generation.append(gene1)
            self.next_generation.append(gene2)

        self.now_generation = self.next_generation
        self.next_generation = []

    def choice(self) -> List[int]:
        operation = random.randint(0, 1)
        if operation == 0:
            return self.roulette_choice()
        elif operation == 1:
            return self.tournament_choice()
        assert True

    def crossover(self, gene1, gene2) -> [List[int], List[int]]:
        gene1 = copy(gene1)
        gene2 = copy(gene2)
        if random.random() < self.crossover_probabilities:
            operation = random.randint(0, 3)
            if operation == 0:
                return self.uniform_crossover(gene1, gene2)
            elif operation == 1:
                return self.two_point_crossover(gene1, gene2)
            elif operation == 2:
                return self.one_point_crossover(gene1, gene2)
        return gene1, gene2

    def mutation(self, gene):
        for index in range(self.size):
            if random.random() < self.mutation_probabilities:
                gene[index] = random.randint(0, self.n_range)
        return gene

    def roulette_choice(self) -> List[int]:
        assert len(self.fitness_values) != 0
        total_fitness = sum(self.fitness_values)
        if total_fitness == 0:
            return self.now_generation[random.randint(0, self.population - 1)]
        each_probabilities = []
        for fitness in self.fitness_values:
            each_probabilities.append(fitness / total_fitness)
        index = np.random.choice(self.population, p=each_probabilities)
        return self.now_generation[index]

    def tournament_choice(self) -> List[int]:
        choice = random.randint(0, self.population - 1)
        for _ in range(self.tournament_size - 1):
            index = random.randint(0, self.population - 1)
            if self.fitness_values[index] > self.fitness_values[choice]:
                choice = index
        return self.now_generation[choice]

    def uniform_crossover(self, gene1, gene2) -> [List[int], List[int]]:
        for index in range(self.size):
            probabilities = random.random()
            if probabilities <= 0.5:
                gene1[index], gene2[index] = gene2[index], gene1[index]
        return gene1, gene2

    def one_point_crossover(self, gene1, gene2) -> [List[int], List[int]]:
        p = random.randint(0, self.size - 2)
        gene1[p:], gene2[p:] = gene2[p:], gene1[p:]
        return gene1, gene2

    def two_point_crossover(self, gene1, gene2) -> [List[int], List[int]]:
        p1 = random.randint(0, self.size - 1)
        p2 = random.randint(0, self.size - 1)
        if p1 > p2:
            p1, p2 = p2, p1
        gene1[p1:p2], gene2[p1:p2] = gene2[p1:p2], gene1[p1:p2]
        return gene1, gene2
