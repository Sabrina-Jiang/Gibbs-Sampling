import networkx as nx
import itertools
from matplotlib import rc

import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing

# Defining the network
network = {
    "V": ["amenities", "neighborhood", "location", "children", "size", "schools", "age", "price"],
    "E": [["amenities, location"], ["neighborhood", "location"], ["neighborhood", "children"], ["children", "schools"],
          ["location", "age"], ["location", "price"], ["size", "price"], ["age", "price"], ["schools", "price"]],
    "data": {
        "amenities": {
            "variable_card": 2,
            "values": ["lots", "little"],
            "parents": None,
            "children": ["location"],
            "prob": [0.3, 0.7]
        },
        "neighborhood": {
            "variable_card": 2,
            "values": ["bad", "good"],
            "parents": None,
            "children": ["location", "children"],
            "prob": [0.4, 0.6]
        },
        "location": {
            "variable_card": 3,
            "values": ["good", "bad", "ugly"],
            "parents": ["amenities", "neighborhood"],
            "children": ["age", "price"],
            "prob": {
                "['lots','bad']": [0.3, 0.4, 0.3],
                "['lots','good']": [0.8, 0.15, 0.05],
                "['little','bad']": [0.2, 0.4, 0.4],
                "['little','good']": [0.5, 0.35, 0.15]
            }
        },
        "children": {
            "variable_card": 2,
            "values": ["bad", "good"],
            "parents": ["neighborhood"],
            "children": ["schools"],
            "prob": {
                "['bad']": [0.6, 0.4],
                "['good']": [0.3, 0.7]
            }
        },
        "size": {
            "variable_card": 3,
            "values": ["small", "medium", "large"],
            "parents": None,
            "children": ["price"],
            "prob": [0.33, 0.34, 0.33]
        },
        "schools": {
            "variable_card": 2,
            "values": ["bad", "good"],
            "parents": ["children"],
            "children": ["price"],
            "prob": {
                "['bad']": [0.7, 0.3],
                "['good']": [0.8, 0.2]
            }
        },
        "age": {
            "variable_card": 2,
            "values": ["old", "new"],
            "parents": ["location"],
            "children": ["price"],
            "prob": {
                "['good']": [0.3, 0.7],
                "['bad']": [0.6, 0.4],
                "['ugly']": [0.9, 0.1]
            }
        },
        "price": {
            "variable_card": 3,
            "values": ["cheap", "ok", "expensive"],
            "parents": ["location", "age", "schools", "size"],
            "children": None,
            "prob": {
                "['good','old','bad','small']": [0.5, 0.4, 0.1],
                "['good','old','bad','medium']": [0.4, 0.45, 0.15],
                "['good','old','bad','large']": [0.35, 0.45, 0.2],
                "['good','old','good','small']": [0.4, 0.3, 0.3],
                "['good','old','good','medium']": [0.35, 0.3, 0.35],
                "['good','old','good','large']": [0.3, 0.25, 0.45],
                "['good','new','bad','small']": [0.45, 0.4, 0.15],
                "['good','new','bad','medium']": [0.4, 0.45, 0.15],
                "['good','new','bad','large']": [0.35, 0.45, 0.2],
                "['good','new','good','small']": [0.25, 0.3, 0.45],
                "['good','new','good','medium']": [0.2, 0.25, 0.55],
                "['good','new','good','large']": [0.1, 0.2, 0.7],
                "['bad','old','bad','small']": [0.7, 0.299, 0.001],
                "['bad','old','bad','medium']": [0.65, 0.33, 0.02],
                "['bad','old','bad','large']": [0.65, 0.32, 0.03],
                "['bad','old','good','small']": [0.55, 0.35, 0.1],
                "['bad','old','good','medium']": [0.5, 0.35, 0.15],
                "['bad','old','good','large']": [0.45, 0.4, 0.15],
                "['bad','new','bad','small']": [0.6, 0.35, 0.05],
                "['bad','new','bad','medium']": [0.55, 0.35, 0.1],
                "['bad','new','bad','large']": [0.5, 0.4, 0.1],
                "['bad','new','good','small']": [0.4, 0.4, 0.2],
                "['bad','new','good','medium']": [0.3, 0.4, 0.3],
                "['bad','new','good','large']": [0.3, 0.3, 0.4],
                "['ugly','old','bad','small']": [0.8, 0.1999, 0.0001],
                "['ugly','old','bad','medium']": [0.75, 0.24, 0.01],
                "['ugly','old','bad','large']": [0.75, 0.23, 0.02],
                "['ugly','old','good','small']": [0.65, 0.3, 0.05],
                "['ugly','old','good','medium']": [0.6, 0.33, 0.07],
                "['ugly','old','good','large']": [0.55, 0.37, 0.08],
                "['ugly','new','bad','small']": [0.7, 0.27, 0.03],
                "['ugly','new','bad','medium']": [0.64, 0.3, 0.06],
                "['ugly','new','bad','large']": [0.61, 0.32, 0.07],
                "['ugly','new','good','small']": [0.48, 0.42, 0.1],
                "['ugly','new','good','medium']": [0.41, 0.39, 0.2],
                "['ugly','new','good','large']": [0.37, 0.33, 0.3]
            }
        },
    }
}

def choose_random_state(node, network):
    '''
    Assigns a random state to a given node, like assign 'small' to size
    @param node:
    @param network:
    @return:
    '''
    variable_card = network['data'][node]['variable_card']
    random_index = random.randint(0, variable_card - 1)
    return network['data'][node]['values'][random_index]


def choose_non_evidence_node(non_evidence_nodes):
    '''
    choose a random non-evidence node in the current iteration
    @param non_evidence_nodes:
    @return:
    '''
    return non_evidence_nodes[random.randint(0, len(non_evidence_nodes) - 1)]


def update_value(node, network, simulation):
    '''
    update the value of node in previous iteration
    @param node:
    @param network:
    @param simulation:
    @return:
    '''
    parents_current = network['data'][node]['parents']
    childrens_current = network['data'][node]['children']
    all_child_prob = 1
    parent_prob = network['data'][node]['prob']
    # prob = 1

    # temp = np.array(list(network['data'][node]['prob'].values()))[:, 0]
    if parents_current is not None:
        # The node has no parent and we can update it based on the prior
        values_parents = [simulation[-1][parent] for parent in parents_current]
        # print(values_parents)
        parent_prob = network['data'][node]['prob'][str(values_parents).replace(" ", "")]

    if childrens_current is not None:
        child_prob = []
        for child in childrens_current:
            condition_set = []
            child_value = simulation[-1][child]
            child_index = network['data'][child]['values'].index(child_value)
            parent_of_child = network['data'][child]['parents']
            node_index = network['data'][child]['parents'].index(node)
            values_parent_of_child = [simulation[-1][parent] for parent in parent_of_child]
            for state in network['data'][node]['values']:
                # print(state)
                values_parent_of_child[node_index] = state
                condition_set.append(network['data'][child]['prob'][str(values_parent_of_child).replace(" ", "")])
            condition_set = np.array(condition_set)
            child_prob.append(condition_set[:,child_index])
        for row in range(len(child_prob)):
            all_child_prob = np.multiply(all_child_prob, child_prob[row])


    prob = np.multiply(all_child_prob, parent_prob)
    prob_norm = prob / max(np.cumsum(prob))
    prob_norm = np.cumsum(prob_norm)

    choice = random.random()
    index = np.argmax(prob_norm > choice)
    # print(choice ,cumsum)
    return network['data'][node]['values'][index]


def gibbs_sampling(network, evidence, niter, num_drop):
    simulation = []
    nodes = network['V']
    non_evidence_nodes = [node for node in nodes if node not in evidence.keys()]
    # First iteration random value for all nodes
    d = {}
    for node in nodes:
        d[node] = choose_random_state(node, network)
    # Put evidence
    for node in evidence:
        d[node] = evidence[node]
    simulation.append(d.copy())
    # Now iterate
    for count in range(niter):
        # Pick up a random node to start
        current_node_to_update = choose_non_evidence_node(non_evidence_nodes)
        d[current_node_to_update] = update_value(current_node_to_update, network, simulation)
        simulation.append(d.copy())

    after_removing_burnt_samples = simulation[int(num_drop):]
    count = {val: 0 for val in network['data'][node_to_query]['values']}
    for assignment in after_removing_burnt_samples:
        count[assignment[node_to_query]] += 1

    for l in count:
        probabilites[l] = count[l] * 1.0 / (iterations - num_drop)
    return probabilites

def get_command():
    evidence_node = []
    parse = argparse.ArgumentParser()
    parse.add_argument("sampling_method", action="store", type=str)
    parse.add_argument("node_to_query", action="store", type=str)
    parse.add_argument("evidence_set", nargs='*', type=str)
    parse.add_argument('-u', option_strings='-u', type=int, help='Number of Updates to be made')
    parse.add_argument('-d', option_strings='-d', type=int,
                       help='Number of Updates to ignore before computing probability')

    args = parse.parse_args()
    try:
        node_to_query = args.node_to_query
    except:
        print('no nodes to query')
    try:
        evidence_set = args.evidence_set
    except:
        print('no given evidence')
    try:
        num_update = args.u
    except:
        print('no drop number')
    try:
        num_drop = args.d
    except:
        print('no nodes to query')
    # translated the string array to dict
    for item in evidence_set:
        item = item.split("=")
        evidence_node.append(item)
    evidence_node = dict(evidence_node)

    return node_to_query, evidence_node, num_update, num_drop

def fig_data_process(data, probability):
    temp = list(probability.values())
    data.append(temp)


def draw_figure(data, x_vector, title, xlabel, ylabel):
    data = np.array(data)
    for col in range(data.shape[1]):
        plt.plot(x_vector, data[:, col],label = network['data'][node_to_query]['values'][col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

def format_printing(node, count, drop, probability):
    print('With %d samples and %d discarded, the gibbs sampling gives:' % (count, drop))
    for key,value in probability.items():
        print('P(%s = %s) = %.4f ' % (node, key, round(value,4)))
    print('\n')

# print(choose_random_state('location',network))
# print()


if __name__ == '__main__':
    import argparse

    probabilites = {}

    node_to_query, evidence_node, iterations, num_drop = get_command()

    # converge test to see if numbers of iterations effect the converge
    num_update_list = np.arange(5000, 100000, 2000)
    num_drop = 1000
    fig1_data = []
    for iterations in num_update_list:
        probabilites = gibbs_sampling(network, evidence_node, iterations, num_drop)
        # print(probabilites)
        format_printing(node_to_query, iterations, num_drop, probabilites)
        fig_data_process(fig1_data, probabilites)
    # print(fig1_data)

    # converge test to see if numbers of drops effect the converge
    #  effect the converge
    num_drop_list = np.arange(1000, 11000, 1000)
    iterations = 50000
    fig2_data = []
    for num in num_drop_list:
        probabilites = gibbs_sampling(network, evidence_node, iterations, num)
        # print(probabilites)
        format_printing(node_to_query, iterations, num, probabilites)
        fig_data_process(fig2_data, probabilites)
    # print(fig1_data)

    plt.figure()
    plt.subplot(211)
    draw_figure(fig1_data, num_update_list, 'Sample number vs converge', 'Sample Count', 'Estimate probability')
    plt.subplot(212)
    draw_figure(fig2_data, num_drop_list, 'Drop number vs converge', 'Drop Count', 'Estimate probability')
    plt.show()
    print('Figure draw complete')


    # python gibbs_sampling.py gibbs size schools=good location=ugly -u 10000 -d 0
