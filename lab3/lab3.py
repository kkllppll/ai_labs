import numpy as np
import matplotlib.pyplot as plt
import random

# визначення функції
def function(x):
    return 5 * np.sin(10 * x) * np.sin(3 * x) / (x ** 0.5)

# побудова графіка функції
x = np.linspace(1, 7, 1000)
y = function(x)

plt.plot(x, y)
plt.title('Графік функції: 5 * np.sin(10 * x) * np.sin(3 * x) / (x ** 0.5)')
plt.xlabel('x')
plt.ylabel('Y(x)')
plt.grid(True)
plt.show()

# параметри генетичного алгоритму
population_size = 100  # позмір популяції
generations = 1000  # кількість поколінь
mutation_rate = 0.02  # швидкість мутацій
RANDOM_SEED = 42  # фіксований випадковий сід для відтворюваності
X_MIN = 1
X_MAX = 7
E = 0.000001

# кількість значень N які потрібно закодувати
N = int((X_MAX - X_MIN) / E)
NB = 0
# потім підбирається таке значення кількості бітів NB, що задовольняє умові
while (2 ** NB) < N:
    NB += 1
# далі обчислюється крок дискретизації заданого діапазону
D = (X_MAX - X_MIN) / (2 ** NB)

# знаючи крок дискретизації ми можемо закодувати значення X натуральним числом Nx за формулою
# кодування значення x в бінарну хромосому
def encode(x, x_min, D, NB):
    normalized = (x - x_min) / D
    # після чого Nx кодується у двійковій системі числення
    binary = bin(int(normalized))[2:].zfill(NB)
    return binary

# розкодування бінарної хромосоми в значення x
def decode(binary, x_min, D):
    integer = int(binary, 2)
    x = x_min + integer * D
    return x

# ініціалізація популяції
def initialize_population(population_size, NB):
    random.seed(RANDOM_SEED)
    return [''.join(random.choice('01') for _ in range(NB)) for _ in range(population_size)]

# функція пристосованості
def fitness(chromosome, x_min, D):
    x = decode(chromosome, x_min, D)
    return function(x)

# відбір
def selection(population, x_min, D):
    # сортує популяцію за значенням фітнесу у спадному порядку.
    sorted_population = sorted(population, key=lambda c: fitness(c, x_min, D), reverse=True)
    return sorted_population[:len(population) // 2] # половину відсортованої популяції.

# кросовер
# відповідає за створення нових індивідів (дітей) 
# шляхом комбінування генетичної інформації від двох батьківських індивідів.
def crossover(parent1, parent2):
    # генерує випадкову точку кросовера де саме розірветься кожен з батьківських індивідів
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# мутація
def mutate(chromosome):
    index = random.randint(0, len(chromosome) - 1)
    mutated = list(chromosome)
    mutated[index] = '0' if mutated[index] == '1' else '1'
    return ''.join(mutated)

# генетичний алгоритм
def genetic_algorithm(population_size, generation, x_min, x_max, NB):
    population = initialize_population(population_size, NB)
    
    for generation in range(generation):
        selected = selection(population, x_min, D) # найкращі хромосоми
        next_generation = []

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            next_generation.extend([child1, child2])

        population = next_generation

    return population

# знаходження максимального та мінімального значень
population = genetic_algorithm(population_size, generations, X_MIN, X_MAX, NB)
max_individual = max(population, key=lambda c: fitness(c, X_MIN, D))
min_individual = min(population, key=lambda c: fitness(c, X_MIN, D))

max_value = fitness(max_individual, X_MIN, D)
min_value = fitness(min_individual, X_MIN, D)
max_x = decode(max_individual, X_MIN, D)
min_x = decode(min_individual, X_MIN, D)

print(f"Максимальне значення: {max_value} при x = {max_x}")
print(f"Хромосома для максимального значення: {max_individual}")
print(f"Мінімальне значення: {min_value} при x = {min_x}")
print(f"Хромосома для мінімального значення: {min_individual}")