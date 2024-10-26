#!/usr/bin/env python
# coding: utf-8
"""
4clusters_V6_26OUT_OK.py
Simulação de uma Rede de Sensores Inteligentes em clusters, baseada no método K-Means e na Lei de Friis, com controle de eficiência energética.
@author: Maurício Brigato
Mestrado em Engenharia Elétrica FESJ ICTS
Unesp Universidade Estadual Paulista "Júlio de Mesquita Filho"
https://www2.unesp.br/
https://www.sorocaba.unesp.br/#!/pos-graduacao/--engenharia-eletrica-local/
date: 25/10/2024
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def minkowski_distance(point, center, lamb):
    return sum(abs(point[k] - center[k]) ** lamb for k in range(len(point))) ** (1 / lamb)

def generate_points(tot_sensors, limit):
    return [(random.uniform(0, limit), random.uniform(0, limit)) for _ in range(tot_sensors)]

def calculate_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def friis_transmission_power(pt, gt, gr, lambda_, d):
    if d == 0:
        return float('inf')  # Evita divisão por zero
    return pt * (gt * gr * (lambda_ / (4 * math.pi * d)) ** 2)

def find_cluster_head(cluster, centroid):
    closest_point = min(cluster, key=lambda point: calculate_distance(point[0], centroid[0], point[1], centroid[1]))
    return closest_point

def main():
    print('------------------------4CLUSTERS - INÍCIO: --->')
    inicio = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    print("Início:", inicio)

    # Definir a seed para garantir a reprodutibilidade dos resultados
    random_seed = 42
    random.seed(random_seed)

    # Variáveis de inicialização
    FREQ = 2.4 * 10**9
    PrdBm = -80
    Pt_transceiver = 0.001
    Pr = 10 ** (PrdBm / 10) / 1000
    LAMBDA = 299792458 / FREQ
    GT = 1.0
    GR = 1.0
    area_limit = 100
    tolerance = 1e-3  # Tolerância para parar o loop
    max_iterations = 1000  # Número máximo de iterações

    # Caminho para salvar os gráficos
    save_path = r'C:\Users\mauri\Documents\Mestrado_OFICIAL\DEFESA MESTRADO'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Coordenada da estação base
    sink_position = (100, 50)

    totSensors = 128
    points = generate_points(totSensors, area_limit)

    # Inicialização dos centroides
    num_clusters = 4  # Número de clusters fixo como 4 para este exemplo
    centroids = [(random.uniform(0, area_limit), random.uniform(0, area_limit)) for _ in range(num_clusters)]
    print("Centroides iniciais:", centroids)

    # Para armazenar a trajetória de cada centroide
    trajectory = [[] for _ in range(num_clusters)]

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        clusters = [[] for _ in range(num_clusters)]

        # Atribuir cada ponto ao cluster mais próximo
        for x, y in points:
            distances = [minkowski_distance((x, y), centroid, 1) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append((x, y))

        # Atualizar os centroides e registrar a trajetória
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(x for x, _ in cluster) / len(cluster)
                mean_y = sum(y for _, y in cluster) / len(cluster)
                new_centroids.append((mean_x, mean_y))
                trajectory[i].append((mean_x, mean_y))
            else:
                new_centroids.append((random.uniform(0, area_limit), random.uniform(0, area_limit)))
                trajectory[i].append(centroids[i]) 

        # Verificar se os centroides pararam de mover
        if all(math.isclose(new_centroids[i][0], centroids[i][0], abs_tol=tolerance) and
               math.isclose(new_centroids[i][1], centroids[i][1], abs_tol=tolerance) for i in range(num_clusters)):
            print(f"Iterações interrompidas na iteração {iteration} devido à convergência dos centroides.")
            break

        centroids = new_centroids

    # Calcular as potências de transmissão dos cluster heads para a estação base
    cluster_powers = []
    for i, cluster in enumerate(clusters):
        if cluster:
            centroid = centroids[i]
            cluster_head = find_cluster_head(cluster, centroid)
            total_transmission_power = 0

            for sensor in cluster:
                d = calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1])
                if d > 0:
                    transmission_power = friis_transmission_power(Pt_transceiver, GT, GR, LAMBDA, d)
                    total_transmission_power += transmission_power

            # Calcular a potência do cluster head para a estação base
            d_to_sink = calculate_distance(cluster_head[0], sink_position[0], cluster_head[1], sink_position[1])
            ch_to_sink_power = friis_transmission_power(Pt_transceiver, GT, GR, LAMBDA, d_to_sink)

            cluster_powers.append({
                'cluster_index': i + 1,
                'cluster_head': cluster_head,
                'total_transmission_power': total_transmission_power,
                'ch_to_sink_power': ch_to_sink_power,
                'distance_to_sink': d_to_sink,
                'total_sensors': len(cluster)
            })

    # Gráfico 1: Trajetória dos centroides e identificação dos cluster heads
    plt.figure(figsize=(10, 10))
    for i, (cluster, traj) in enumerate(zip(clusters, trajectory)):
        x_vals = [x for x, y in cluster]
        y_vals = [y for x, y in cluster]
        plt.scatter(x_vals, y_vals, label=f'Cluster {i+1}')

        traj_x = [x for x, y in traj]
        traj_y = [y for x, y in traj]
        plt.plot(traj_x, traj_y, '--', label=f'Trajetória Centroide {i+1}')

        plt.scatter(traj_x[-1], traj_y[-1], color='black', marker='x', s=100)

        cluster_head = find_cluster_head(cluster, (traj_x[-1], traj_y[-1]))
        plt.scatter(cluster_head[0], cluster_head[1], color='cyan', edgecolor='black', marker='*', s=150, label=f'Cluster Head {i+1}')
        plt.text(cluster_head[0], cluster_head[1], f'CH{i+1}', fontsize=12, verticalalignment='bottom')

    plt.scatter(sink_position[0], sink_position[1], color='red', marker='s', s=150, label='Base Station (Sink)')
    plt.title(f"Gráfico 1: Trajetória dos Centroides e Cluster Heads ({num_clusters} Clusters, Iterações: {iteration})")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.legend()
    # Salvando o gráfico corretamente
    plt.savefig(os.path.join(save_path, f"Grafico1_Trajetoria_Centroides_{num_clusters}clusters_{iteration}iteracoes.png"), bbox_inches='tight')
    plt.show()

    # Gráficos 2, 3, 4 e 5 continuam como antes, usando o valor real de "iteration" e sendo salvos corretamente
    # Gráfico 2: Distribuição dos clusters com cluster heads
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, cluster in enumerate(clusters):
        x_vals = [x for x, y in cluster]
        y_vals = [y for x, y in cluster]
        ax.scatter(x_vals, y_vals, label=f'Cluster {i+1}')

        cx, cy = centroids[i]
        ax.scatter(cx, cy, color='black', marker='x', s=100)

        cluster_head = cluster_powers[i]['cluster_head']
        ax.scatter(cluster_head[0], cluster_head[1], color='cyan', edgecolor='black', marker='*', s=150, label=f'Cluster Head {i+1}')
        ax.text(cluster_head[0], cluster_head[1], f'CH{i+1}', fontsize=12, verticalalignment='bottom')

    ax.scatter(sink_position[0], sink_position[1], color='red', marker='s', s=150, label='Base Station (Sink)')
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_title(f"Gráfico 2: Clusters com Alcance Máximo e Cluster Heads ({num_clusters} Clusters, Iterações: {iteration})")
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"Grafico2_Distribuicao_Clusters_{num_clusters}clusters_{iteration}iteracoes.png"), bbox_inches='tight')
    plt.show()

    # Gráfico 3: Potência de transmissão dos cluster heads para a base station
    ch_to_sink_powers = [data['ch_to_sink_power'] for data in cluster_powers]
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, num_clusters+1), ch_to_sink_powers, color='blue')
    plt.xlabel("Cluster")
    plt.ylabel("Potência de Transmissão para Base Station (W)")
    plt.title(f"Gráfico 3: Potência de Transmissão dos Cluster Heads para Base Station ({num_clusters} Clusters, Iterações: {iteration})")
    plt.xticks(range(1, num_clusters+1), [f'CH{i}' for i in range(1, num_clusters+1)])
    plt.savefig(os.path.join(save_path, f"Grafico3_Potencia_Cluster_Heads_{num_clusters}clusters_{iteration}iteracoes.png"), bbox_inches='tight')
    plt.show()

    # Gráfico 4: Potência Total Gasta por Cluster (em ordem crescente)
    df_clusters_sorted = pd.DataFrame(cluster_powers).sort_values("total_transmission_power").reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.bar(df_clusters_sorted.index + 1, df_clusters_sorted["total_transmission_power"], color='blue', alpha=0.7)
    plt.xlabel("Cluster (Ordenado por Potência Total)")
    plt.ylabel("Potência Total Gasta (W)")
    plt.title(f"Gráfico 4: Potência Total Gasta por Cluster (Ordem Crescente) ({num_clusters} Clusters, Iterações: {iteration})")
    plt.xticks(df_clusters_sorted.index + 1, [f'Cluster {i}' for i in df_clusters_sorted["cluster_index"]])
    plt.savefig(os.path.join(save_path, f"Grafico4_Potencia_Total_Clusters_{num_clusters}clusters_{iteration}iteracoes.png"), bbox_inches='tight')
    plt.show()

    # Gráfico 5: Comparação de distâncias dos cluster heads ao sink
    distances_to_sink = [data['distance_to_sink'] for data in cluster_powers]
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, num_clusters+1), distances_to_sink, color='green')
    plt.xlabel("Cluster")
    plt.ylabel("Distância do CH ao Sink (m)")
    plt.title(f"Gráfico 5: Distância dos Cluster Heads ao Sink ({num_clusters} Clusters, Iterações: {iteration})")
    plt.xticks(range(1, num_clusters+1), [f'CH{i}' for i in range(1, num_clusters+1)])
    plt.savefig(os.path.join(save_path, f"Grafico5_Distancia_Cluster_Heads_{num_clusters}clusters_{iteration}iteracoes.png"), bbox_inches='tight')
    plt.show()

    print('------------------------4CLUSTERS - FINAL: --->')
    fim = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    print("Início:", inicio)
    print("Final:", fim)

if __name__ == "__main__":
    main()
