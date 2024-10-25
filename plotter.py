import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


def load_data(graph_type=None, n_range=None, k_range=None, q_range=None, perl_vertex=None):
    base_dir = Path('data')

    all_data = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folder = Path(root)
                folder_parts = folder.parts

                try:
                    graph_type_folder = folder_parts[-2]
                    params = folder_parts[-1].split('_')

                    graph_type_curr = graph_type_folder
                    n_curr = int(params[1][1:]) if params[1][1:].isdigit() else None
                    k_curr = int(params[2][1:]) if params[2][1:].isdigit() else None
                    q_curr = int(params[3][1:]) if params[3][1:].isdigit() else None
                    perl_vertex_curr = int(params[4][5]) if params[4][5].isdigit() else None

                    if (graph_type is None or graph_type == graph_type_curr) and \
                            (n_range is None or (n_curr is not None and n_range[0] <= n_curr <= n_range[1])) and \
                            (k_range is None or (k_curr is not None and k_range[0] <= k_curr <= k_range[1])) and \
                            (q_range is None or (q_curr is not None and q_range[0] <= q_curr <= q_range[1])) and \
                            (perl_vertex is None or (perl_vertex_curr is not None and perl_vertex == perl_vertex_curr)):
                        df = pd.read_csv(file_path)
                        df['graph_type'] = graph_type_curr
                        df['n'] = n_curr
                        df['k'] = k_curr
                        df['q'] = q_curr
                        df['perl_vertex'] = perl_vertex_curr
                        all_data.append(df)
                except Exception as e:
                    print(f"Error al procesar {file_path}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_data(df):
    if df.empty:
        print("No se encontraron datos con los filtros seleccionados.")
        return

    n_values = sorted(df['n'].unique())
    max_values = {}

    for n in n_values:
        df_n = df[df['n'] == n]
        grouped = df_n.groupby('prob')['media'].mean()
        max_values[n] = grouped.max()

    sorted_n_values = sorted(max_values, key=max_values.get)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_n_values)))

    lower_bound = np.zeros_like(df.groupby('prob')['media'].mean().values)

    for i, n in enumerate(sorted_n_values):
        df_n = df[df['n'] == n]
        grouped = df_n.groupby('prob')['media'].mean()

        plt.plot(grouped.index, grouped.values, linestyle='-', color=colors[i], label=f'n={n}')

        plt.fill_between(grouped.index, lower_bound, grouped.values, color=colors[i], alpha=0.7)

        lower_bound = grouped.values

    plt.title('Número medio de componentes conexas para diferentes valores de n en un ' + graph_type)
    plt.xlabel('Probabilidad de percolación (p)')
    plt.ylabel('Media de componentes conexas')
    plt.legend(title='Valor de n')
    plt.grid(True)
    plt.show()

graph_type = input('graph type (g2d, rgg, g3d, ccg): ')
n_range_a = int(input('n_range a: '))
n_range_b = int(input('n_range b: '))
prob = int(input('number of prob: '))
inst_per_prob = int(input('intancies per prob: '))
perl_vertex = int(input('perlocar vertex (1 o 0): '))
data = load_data(graph_type=graph_type, n_range=(n_range_a, n_range_b), k_range=(inst_per_prob, inst_per_prob),q_range=(prob, prob),perl_vertex=perl_vertex)
plot_data(data)
