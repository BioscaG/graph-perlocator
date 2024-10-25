import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import multiprocessing
import pandas as pd
from datetime import datetime
import itertools	
from pathlib import Path
import configparser

AVG_FRIENDS = 5

def percolate_graph(G, p, perl_vertex=True):
    H = G.copy()
    if not perl_vertex:
        for edge in list(H.edges):
            if random.random() > p:
                H.remove_edge(*edge)
    else:
        for node in list(H.nodes):
            if random.random() > p:
                H.remove_node(node)
    return nx.number_connected_components(H)

def calculate_mitj(G, k, p, perl_vertex):
    mitj = 0
    for i in range(k):
        mitj += percolate_graph(G, p, perl_vertex)
    mitj = mitj/k
    print(p,mitj)
    return mitj, p

def evaluate(G, k, q, perl_vertex=True):
    p_values = np.linspace(0, 1, q)
    args_list = [(G, k, p, perl_vertex) for p in p_values]
    connected_fractions = []
    with multiprocessing.Pool() as pool:
        connected_fractions = pool.starmap(calculate_mitj, args_list)
    return connected_fractions

def average_cc(vectors):
    return np.mean(vectors, axis=0)

def plot_and_save(mediaandprob, graph_type, n, k, q, perl_vertex, show_plot=True):
    media, prob = zip(*mediaandprob)
    plt.figure(figsize=(10, 6))
    plt.plot(prob, media, marker='o', linestyle='-', color='b')
    title = 'Transición de Fase ' + graph_type
    plt.title(title)
    plt.xlabel('Probabilidad de Percolación (p)')
    plt.ylabel('Media Grafos Conexos')
    plt.legend()
    plt.grid()

    main_folder = Path('data')
    folder_name = '{}_n{}_k{}_q{}_pvert{}'.format(graph_type, n, k, q, perl_vertex)
    folder = main_folder / graph_type / folder_name
    main_folder.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)

    date = datetime.now()
    date = date.strftime("%Y-%m-%d_%H:%M")
    file_name = str(date) + '.csv'
    df = pd.DataFrame({
        'prob': prob,
        'media': media
    })
    df.to_csv(str(folder) + '/' + file_name, index=False)
    plot_name = str(date) + '.png'
    plt.savefig(str(folder) + '/' + plot_name)
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_connected_rgg(n, radius):
    G = None
    while G is None or not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius)
    return G

def generate_connected_grid(n):
    G = None
    while G is None or not nx.is_connected(G):
        G = nx.grid_2d_graph(n, n)
    return G

def generate_connected_3dgrid(n):
    G = None
    while G is None or not nx.is_connected(G):
        G = nx.Graph()
        for x, y, z in itertools.product(range(n), range(n), range(n)):
            G.add_node((x, y, z))
        for x, y, z in itertools.product(range(n), range(n), range(n)):
            if x + 1 < n:
                G.add_edge((x, y, z), (x + 1, y, z))
            if y + 1 < n:
                G.add_edge((x, y, z), (x, y + 1, z))
            if z + 1 < n:
                G.add_edge((x, y, z), (x, y, z + 1))
    return G

def generate_connected_caveman_graph(l, k):
    G = nx.connected_caveman_graph(l, k)
    return G

def input_data():
    graph_type = input('Introdueix tipus de graf (g2d, rgg, g3d, ccg): ')
    if graph_type == 'g3d':
        print('ALERTA: Consumeix molta memoria (posar n < 100)')
    if graph_type != 'ccg':
        n = int(input('Introdueix la mida del graf n{}: '.format(
            'xn' if graph_type == 'g2d' else 'xnxn' if graph_type == 'g3d' else '')))
    elif graph_type == 'ccg':
        n = int(input('Introdueix el nombre de coves: '))
    k = int(input('Nombre de proves per probabilitat: '))
    q = int(input('Quants valors de probabilitat vols estudiar: '))
    perl_vertex = False
    if graph_type != 'rgg':
        perl_vertex = int(input('Perlocar vertexs(1) o arestes(0)? '))
    return graph_type, n, k, q, perl_vertex


def generate_percolate_save(graph_type, n, k, q, perl_vertex,show_plot=True):
    mediaandprob = None
    if graph_type == 'g2d' or graph_type == 'g3d':
        G = generate_connected_grid(n) if graph_type == 'g2d' else generate_connected_3dgrid(n)
        mediaandprob = evaluate(G, k, q, perl_vertex)

    elif graph_type == 'rgg':
        radius = np.sqrt(np.log(n) / (np.pi * n))
        means_and_probs = []
        for _ in range(100):
            G = generate_connected_rgg(n, radius) 
            m_i_p_de_G = evaluate(G, k, q, perl_vertex=False) 
            means_and_probs.append(m_i_p_de_G)
            break
        mediaandprob = average_cc(means_and_probs) 
    
    elif graph_type == 'ccg':
        G = generate_connected_caveman_graph(n, AVG_FRIENDS)
        mediaandprob =evaluate(G, k, q, perl_vertex)

    plot_and_save(mediaandprob, graph_type, n, k, q, perl_vertex, show_plot)

def parameters_reader():
    config = configparser.ConfigParser()
    config_name = input('config file name: ')
    config.read(config_name)
    lista = config.get('config', 'n').split(',')
    n = [int(i) for i in lista]
    k = config.getint('config', 'k')
    q = config.getint('config', 'q')
    perl_vertex = config.getint('config', 'perl_vertex')
    graph_type = config.get('config', 'graph_type')
    return n, k, q, perl_vertex, graph_type


def main():
    automatic_generation = input('Generacio automatica o manual (a, m): ')
    if automatic_generation == 'm':
        graph_type, n, k, q, perl_vertex = input_data()
        generate_percolate_save(graph_type, n, k, q, perl_vertex)
    else:
        n, k, q, perl_vertex, graph_type = parameters_reader()
        for n_in in n:
            print('VALOR ', n_in, '////////')
            generate_percolate_save(graph_type, n_in, k, q, perl_vertex, False)



if __name__ == "__main__":
    main()





