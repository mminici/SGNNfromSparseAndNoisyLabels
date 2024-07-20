import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from typing import Set
from community import community_louvain


def get_transitive_triads(G: nx.DiGraph, filter_transitive=False) -> Set:
    """Returns set of transitive triads from networkx DiGraph
    A transitive triad in the output set is defined as:
    {A, B, C} := {(A, B), (B, C), (A, C)}
    """
    transitive_triads = set()
    for n in G.nodes():
        for nbr in G.neighbors(n):
            for nbr2 in G.neighbors(nbr):
                if G.has_edge(n, nbr2) and n != nbr2:  # Check for transitivity and avoid self-loops
                    transitive_triads.add((int(n), int(nbr), int(nbr2)))
    transitive_triads_names = {'030T', '120D', '120U', '300'}
    if filter_transitive:
        transitive_triads_to_exclude = set()
        for triad in transitive_triads:
            triad_type = nx.triad_type(G.subgraph(triad))
            if triad_type not in transitive_triads_names:
                transitive_triads_to_exclude.add(triad)
        transitive_triads = transitive_triads.difference(transitive_triads_to_exclude)
    return transitive_triads


get_triad_edges = lambda x: [(x[0], x[1]), (x[1], x[2]), (x[0], x[2])]


# noinspection PyShadowingNames
def filter_transitive_triads(transitive_triads: Set,
                             labels: np.array,
                             edge_index: np.array) -> Set:
    """Returns the known_edgeindex, known_labels, unknown_edgeindex
      containing triads with only one missing sign
    """

    known_edgeindex = []
    known_labels = []
    unknown_edgeindex = []
    for triad in tqdm(transitive_triads, desc='Iter on transitive triads'):
        edges = get_triad_edges(triad)

        num_missing_signs = 0
        curr_existing_edges = []
        curr_labels = []
        curr_unknown_edge = None
        for e in edges:
            e_index = np.where(np.logical_and(edge_index[0] == e[0], edge_index[1] == e[1]))[0][0]
            if labels[e_index] == 0:
                num_missing_signs += 1
                curr_unknown_edge = edge_index[:, e_index]
            else:
                curr_existing_edges.append(edge_index[:, e_index])
                curr_labels.append(labels[e_index])

        if num_missing_signs == 1:
            known_edgeindex.append(curr_existing_edges)
            known_labels.append(curr_labels)
            unknown_edgeindex.append(curr_unknown_edge)

    return np.array(known_edgeindex), np.array(known_labels), np.array(unknown_edgeindex)


def get_unknown_label(known_labels):
    """
    This function returns the expected sign of the unknown edge according to the social balance theory
    Exemplary cases:
        (-1, -1) -> 1
        (1, 1) -> 1
        (1, -1) -> -1
    :param known_labels: labels of the two edges with known sign composing each triad
    :return:
    """
    return known_labels[:, 0] * known_labels[:, 1]


def get_data_for_triad_social_balance(dataset, filter_transitive, device):
    """
    This function puts together all the previous function to output the necessary data
    to compute the self-supervised loss based on micro/triad-level social balance
    :return: unknown edges (pos, neg, labels) and known edges [edges, labels]
    """
    edge_index = torch.vstack([dataset['edges'],
                               dataset['masked_edges']]).detach().cpu().numpy().T
    train_known_labels = torch.clone(dataset['label']).to(device)
    train_known_labels[train_known_labels == 0] = -1.
    labels = torch.cat([train_known_labels, torch.Tensor(
        [0] * dataset['masked_edges'].shape[0]).float().to(device)]).detach().cpu().numpy()

    # Create directed graph from training edges
    graph = nx.from_edgelist(edge_index.T, create_using=nx.DiGraph())

    print('Getting transitive triangles')
    transitive_triads = get_transitive_triads(graph, filter_transitive)

    print('Filter transitive triangles with exactly one unknown edge')
    known_edgeindex, known_labels, unknown_edgeindex = filter_transitive_triads(transitive_triads, labels, edge_index)

    assert known_edgeindex.shape[0] == known_labels.shape[0] == unknown_edgeindex.shape[0]

    unknown_labels = get_unknown_label(known_labels)
    pos_unknown_edges = torch.Tensor(unknown_edgeindex.T[:, unknown_labels == 1]).long().to(device)
    neg_unknown_edges = torch.Tensor(unknown_edgeindex.T[:, unknown_labels == -1]).long().to(device)
    known_edgeindex = torch.Tensor(known_edgeindex).long().to(device)
    known_labels = torch.Tensor(known_labels).float().to(device)
    unknown_labels = torch.Tensor(unknown_labels).float().to(device)

    return [pos_unknown_edges, neg_unknown_edges, unknown_labels], [known_edgeindex, known_labels]

