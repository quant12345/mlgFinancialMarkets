from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


''' Example

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score as metric
import deap_gen_model_feat as dgm

"""
Setting the model's string parameters is done directly
rows 113:115 clf = self.parameters_class['model'](
            **dict_parameters, max_features="sqrt", random_state=0
        )
"""

parameters_model = {
    'max_depth': [2, 4],
    'n_estimators': [5, 100],
    'learning_rate': [0.01, 0.9]
}

parameters = {
    'model': GradientBoostingClassifier,
    'metric': metric,
    'parameters' : parameters_model
}

dataset - for training and prediction
y - labels for classification
slice_split - to split dataset into training and testing sets

dgm.run(dataset, y, parameters, slice_split, percent_label=0.16)
'''

''' Describe

Default:

percent_label -  percentage of predicted class 0 labels in the
test sample from the total number. Default value = 0.16.

metric_more - the resulting metric must be >= this number. Default value = 0.54.

tolerance - the absolute difference between train and test 
metrics should not exceed this number. Default value = 0.02.

print_rows - print result rows that passed the conditions.
Default value = True.

Constants:
Need to set it up manually, but here are some recommendations.
population_size - chromosome * 10 - 20 recommended
penalty features - penalizes more if there are more features
probability_mutation - is best left as is, as increasing it 
will lead to too frequent diversity.
hall_of_fame_size - how many best results to show and also this number
for elitism (how many best to transfer to the next population).
'''

# Constants to change:
population_size = 450
probability_crossover = 0.9
probability_mutation = 0.1
generations_size = 100
hall_of_fame_size = 5
tournament_size = 2
penalty_features = 0.00001


random.seed(42)


class Classifier:
    def __init__(self, arr, y, split, param_class, params_label):
        self.dataset = arr
        self.y = y
        self.split = split
        self.parameters_class = param_class
        self.params_label = params_label

    def __len__(self):
        return self.dataset.shape[1]

    def get_metric_value(self, params):
        index_features = self.dataset.shape[1]
        parameters = params[index_features:]
        dict_parameters = dict(
            zip(self.parameters_class['parameters'].keys(), parameters)
        )

        if 0 in params[index_features:]:
            # print hyperparameters if they contain
            # zero, terminate the function
            print('there is 0 parameter !!!', params[index_features:])
            return 0.0

        indices = np.where(params[:index_features])[0]
        current_data = self.dataset[:, indices]

        if len(indices) == 0 or current_data.shape[1] == 0:
            # print if 0 features are selected or the
            # dataset is empty, terminate the function
            print('param !!!', params, 'currentX.shape[1]!!!',
                  current_data.shape[1])
            return 0.0

        clf = self.parameters_class['model'](
            **dict_parameters, max_features="sqrt", random_state=0
        )

        clf.fit(current_data[:self.split], self.y[:self.split])

        y_pred = clf.predict(current_data[self.split:])
        count = len(y_pred[y_pred == 0])

        # the number of predicted class 0 labels is not less than a
        # certain % of the total number of test labels
        if count > self.y[self.split:].shape[0] * self.params_label['percent_label']:
            y_pred_train = clf.predict(current_data[:self.split])
            result_train = round(
                self.parameters_class['metric'](
                self.y[:self.split], y_pred_train, pos_label=0, zero_division=np.nan
                ),
                3)
            result_test = round(
                self.parameters_class['metric'](
                    self.y[self.split:], y_pred, pos_label=0, zero_division=np.nan
                ),
                3)
            # consider the result of the test metric >= 'metric_more'
            # and the difference between the train and test metrics
            # is no more than 'tolerance'
            if (result_test >= self.params_label['metric_more']
                    and np.abs(result_train - result_test)
                    <= self.params_label['tolerance']):
                if self.params_label['print_rows']:
                    print('test', result_test, 'train',
                          result_train, 'param', params)
            else:
                result_test = 0.0
        else:
            result_test = 0.0

        return result_test


def elitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):

    if halloffame is None:
        raise ValueError('halloffame parameter must not be empty!')

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update Hall of Fame with the initial population
    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        # with size = len(population) - hof_size
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best(halloffame) back to population:
        offspring.extend(halloffame.items)  # *********************

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# Each data type has columns = hist. Each column is shifted forward
# starting from 1. For example, hist = 3. The first is shifted 1,
# the second 2, the third 3.
def preparing_dataset(data, hist):
    features = np.ones((len(data[0]), len(data) * hist), dtype='float64')
    index_ds = 0
    for series in data:
        displacement = 1
        for i in range(0, hist):
            features[:displacement, index_ds] = np.nan
            features[displacement:, index_ds] = series[:-displacement]
            displacement += 1
            index_ds += 1

    features = np.round(features, 5)

    return features

def features_standardized(data, split):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(data[:split])
    X_test_std = sc.transform(data[split:])
    standardized = np.concat([X_train_std, X_test_std])

    return standardized


def check_array(arr, name):
    if np.equal(arr, None).any():
        raise ValueError('The array contains values None in ' + name)
    elif np.any(np.isnan(arr)):
        raise ValueError('The array contains values np.nan in ' + name)



def register(toolbox, dataset, y, split, param_class, params_label):
    count_parameters = len(param_class['parameters'])
    list_key = list(param_class['parameters'])

    # create the Classifier test class:
    classifier = Classifier(dataset, y, split, param_class, params_label)

    # define a single objective, maximizing fitness strategy:
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))

    # create the Individual class based on list:
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # Define a binary vector for feature selection
    toolbox.register('features', random.randint, 0, 1)  # 0 or 1 for feature selection

    # Define model hyperparameters
    for key, value in param_class['parameters'].items():
        if isinstance(value[0], int) and isinstance(value[1], int):
            random_choice = random.randint
        elif isinstance(value[0], float) and isinstance(value[1], float):
            random_choice = random.uniform
        else:
            raise ValueError('Both parameter values must be of type int or float!!!')

        toolbox.register(key, random_choice, value[0], value[1])

    def create_individual():
        # Select features
        features = [toolbox.features() for _ in range(dataset.shape[1])]

        # Create list to hold parameters based on the dictionary passed
        param = []

        # Loop through each parameter key in the
        # toolbox and check if it's registered
        # assuming 'parameters' is a dictionary containing
        # the keys like 'tree', 'leaf', etc.
        for key_ in list_key:
            param_function = getattr(toolbox, key_, None)
            if param_function:  # If the function is registered in the toolbox
                # Add the generated hyperparameter value
                param.append(param_function())

        # Combine features and hyperparameters
        return creator.Individual(features + param)


    # Register the individual creation function
    toolbox.register('individualCreator', create_individual)

    # Create population operator to generate a list of individuals
    toolbox.register(
        'populationCreator', tools.initRepeat, list, toolbox.individualCreator
    )

    # fitness calculation
    def classification(individual):
        # count the number of features (model parameters are excluded)
        num_features_used = sum(individual[:-count_parameters])
        if num_features_used == 0:
            return 0.0,
        else:
            metric = classifier.get_metric_value(individual)

            # return a tuple
            return metric - penalty_features * num_features_used,

    toolbox.register('evaluate', classification)

    # Tournament selection with tournament_size:
    toolbox.register('select', tools.selTournament, tournsize=tournament_size)

    # Single-point crossover:
    toolbox.register('mate', tools.cxTwoPoint)

    def mutate_individual(individual):
        count_individual = sum(
            np.array(np.array(individual) != 0.0, dtype=int))
        slice_individual = len(individual) - count_parameters

        # Mutate the binary features (0 or 1) normally
        for i in range(slice_individual):  # Ignore model hyperparameters
            if random.random() < 1.0 / count_individual:
                individual[i] = 1 - individual[i]  # Flip the feature bit
        # Model hyperparameters
        for index, obj_ind in enumerate(individual[slice_individual:]):
            if random.random() < 1.0 / count_individual:
                start, end = param_class['parameters'][list_key[index]]
                if isinstance(start, int) and isinstance(end, int):
                    random_mutate = random.randint
                elif isinstance(start, float) and isinstance(end, float):
                    random_mutate = random.uniform
                else:
                    raise ValueError('Unsupported data type for mutate')

                individual[slice_individual+index] = random_mutate(start, end)

        return individual,

    toolbox.register('mutate', mutate_individual)

    return classifier


# Genetic Algorithm flow:
def run(dataset, y, parameters, split, **kwargs):

    check_array(dataset, 'dataset')
    check_array(y, 'y_labels')

    default_label = {
        'percent_label': 0.16,
        'metric_more': 0.54,
        'tolerance': 0.02,
        'print_rows': True
    }

    params_label = {**default_label, **kwargs}

    toolbox = base.Toolbox()
    classifier = register(toolbox, dataset, y, split, parameters, params_label)

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=population_size)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(hall_of_fame_size)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism(
        population, toolbox,
        cxpb=probability_crossover,
        mutpb=probability_mutation,
        ngen=generations_size,
        stats=stats,
        halloffame=hof,
        verbose=True)

    # print best solution
    print(f' {hall_of_fame_size} - Best solutions:')
    for i in range(hall_of_fame_size):
        print(i, ': ', hof.items[i], ', fitness = ', hof.items[i].fitness.values[0],
              ', precision = ', classifier.get_metric_value(hof.items[i]),
              ', features = ', sum(hof.items[i][:-len(parameters['parameters'])]))


    # extract statistics:
    max_fitness_values, mean_fitness_values = logbook.select('max', 'avg')

    # plot statistics:
    sns.set_style('whitegrid')
    plt.plot(max_fitness_values, color='red')
    plt.plot(mean_fitness_values, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

    return hof


if __name__ == '__main__':
    pass