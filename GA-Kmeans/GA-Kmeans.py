import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils import avg_iou, evaluate, decode, load_data_VOC2007, kmeans

path_VOC2012 = 'path\to\VOCdevkit\VOC2007\Annotations'

def initialize(individual_num=60):
    """
    Generate initial population , and each gene is encoded to binary system.
    :param individual_num: number of individual.
    :return: generation list in form of binary.
    """
    generation_list = []
    for i in range(0, individual_num):
        gene = ''
        for j in range(0,9):
            gene_part1 = bin(np.random.randint(0, 511))[2:].zfill(9)
            gene_part2 = bin(np.random.randint(0, 511))[2:].zfill(9)
            gene += gene_part1 + gene_part2
        generation_list.append(gene)

    return generation_list


def crossover(gen_list):
    """
    Populations interbreed, genes cross over
    :param population: generation list after encoding
    :return: generation list after crossing over
    """
    pop_len = len(gen_list)
    pop = gen_list
    individual_len = len(pop[0])

    for i in range(pop_len - 1):
        cpoint1 = int(random.randint(0, individual_len))
        cpoint2 = int(random.randint(cpoint1, individual_len))

        pop[i] = pop[i][0:cpoint1] + pop[i + 1][cpoint1:cpoint2] + pop[i][cpoint2:]
        pop[i + 1] = pop[i + 1][0:cpoint1] + pop[i][cpoint1:cpoint2] + pop[i + 1][cpoint2:]

    return pop


def mutation(generation_list: list, mut_p: float):
    """
    Mutate genes with each other , in order to get new genes.
    :param generation_list: generation list in form of binary.
    :param mut_p: mutation probability.
    :return:  new generation list after mutation.
    """
    idx_num = int((12 * mut_p) / 5)
    idx_set = []
    for j, gene in enumerate(generation_list):
        for i in range(0, idx_num):
            idx_set.append(random.randint(1, 43))
        for i in idx_set:
            generation_list[j] = gene[0:i] + gene[i:i + 3][::-1] + gene[i + 3:]


def decode(generation):
    """
    Decode genes to decimal system.
    :param generation:generation list in form of binary.
    :return: generation list after decoding.
    """
    generation_list = generation
    generation_list_decoded = []
    print(len(generation_list))
    for i in range(0, len(generation_list)):
        individual_list_decoded = []
        individual = generation_list[i]
        for j in range(0,9):
            width = round((int(individual[0 + 18 * j:9 + j * 18], 2)))
            height = round((int(individual[9 + j * 18:18 + j * 18], 2)))
            individual_list_decoded.append([width,height])
        generation_list_decoded.append(individual_list_decoded)
    return generation_list_decoded


def evaluate_fitness(data,generation_list_decoded):
    """
    Feed ACO with genes consist of four parameters in ACO and get corresponding fitness.
    :param individual_num: number of individual in each generation.
    :param cityCoordinates: list of city coordinates.
    :param generation_list_decoded: generation list after decoding.
    :param iterMax: number of iteration.
    :return: fitness list
    """
    fitness_list = []
    near_list = []
    for i in range(0, len(generation_list_decoded)):
        cluster = np.array(generation_list_decoded[i])
        fitness, near, cluster = kmeans(box=data,cluster=cluster, k=9)
        fitness_list.append(fitness)
        near_list.append(near)
    return fitness_list, near_list


def select(individual_num, generation_list, fitness_list, near_list):
    """
    By using tournament selection , we get new advantaged generations.
    :param individual_num: number of individual in each generation.
    :param generation_list: generation list in form of binary.
    :param fitness_list:
    :return: list of new advantaged generations
    """
    group_num = 10
    group_size = 10
    group_winner = individual_num // group_num
    winners = []
    winners_score = []
    for i in range(group_num):
        group = []
        score_list = []
        for j in range(group_size):
            player_score = random.choice(fitness_list)
            player = generation_list[fitness_list.index(player_score)]
            score_list.append(player_score)
            group.append(player)
        group, score_list, near_list = rank(group, score_list, near_list)
        winners += group[:group_winner]
        winners_score += score_list[0 : group_winner]



    return winners, winners_score, near_list


def rank(group, score_list, near_list):
    """
    Rank each competition group according to fitness.
    :param group: competition unit.
    :param score_list: fitness list.
    :return: group after ranking.
    """
    for i in range(1, len(group)):
        for j in range(0, len(group) - i):
            if score_list[j] < score_list[j + 1]:
                group[j], group[j + 1] = group[j + 1], group[j]
                score_list[j], score_list[j + 1] = score_list[j + 1], score_list[j]
                near_list[j], near_list[j + 1] = near_list[j + 1], near_list[j]

    return group, score_list, near_list


def GA_Kmeans(path, itermax=1 , individual_num=60, mut_p=0.6,input_shape=[512,512]):
    generation_list = initialize()
    iter = 1
    best_individual = []
    best_individual_score = 0
    data = load_data_VOC2007(path)
    data = data * np.array([input_shape[1], input_shape[0]])
    while iter <= itermax:
        generation_list = crossover(generation_list)
        mutation(generation_list, mut_p)
        generation_list_decoded = decode(generation_list)
        fitness_list, near_list, = evaluate_fitness(data=data, generation_list_decoded=generation_list_decoded)
        winners, winner_score, near_list = select(individual_num=individual_num, generation_list=generation_list, fitness_list=fitness_list, near_list=near_list)
        generation_list = winners
        if max(fitness_list) > best_individual_score:
            best_individual = decode([generation_list[np.argmax(winner_score)]])
            best_near = near_list[np.argmax(winner_score)]
            best_individual_score = np.max(winner_score)
        iter = iter + 1

    best_individual = np.squeeze(best_individual)
    print(best_individual)
    for j in range(len(best_individual)):
        plt.scatter(data[best_near == j][:,0], data[best_near == j][:,1])
        plt.scatter(best_individual[j][0], best_individual[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()

    cluster = []
    cluster =sorted(best_individual,key=lambda  x : x[0] * x[1])
    print('avg_ratio:{:.2f}'.format(avg_iou(data, cluster)))

    f = open("./model_data/new_anchors.txt", 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()





if __name__ == "__main__":
    GA_Kmeans(path=path_VOC2012)