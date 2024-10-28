#!/usr/bin/env python
# coding: utf-8
"""
Programa: comparaclusters.py
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
import os
from datetime import datetime

# Definir caminho para salvar os gráficos
save_path = r'C:\Users\mauri\Documents\Mestrado_OFICIAL\DEFESA MESTRADO\comparaclusters'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Funções auxiliares e de cálculo
def minkowski_distance(point, center, lamb):
    return sum(abs(point[k] - center[k]) ** lamb for k in range(len(point))) ** (1 / lamb)

def generate_points(tot_sensors, limit):
    return [(random.uniform(0, limit), random.uniform(0, limit)) for _ in range(tot_sensors)]

def calculate_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def friis_transmission_power(pt, gt, gr, lambda_, d):
    return pt * (gt * gr * (lambda_ / (4 * math.pi * d)) ** 2) if d > 0 else float('inf')

def find_cluster_head(cluster, centroid):
    return min(cluster, key=lambda point: calculate_distance(point[0], centroid[0], point[1], centroid[1]))

def simulate_clusters(num_clusters, points, sink_position, save_path, lambda_, gt, gr, pt_transceiver):
    centroids = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_clusters)]
    iteration = 0
    tolerance = 1e-3
    trajectory = [[] for _ in range(num_clusters)]
    clusters = [[] for _ in range(num_clusters)]

    # Iterar até a convergência dos centroides
    while True:
        iteration += 1
        clusters = [[] for _ in range(num_clusters)]
        for x, y in points:
            distances = [minkowski_distance((x, y), centroid, 1) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append((x, y))

        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(x for x, y in cluster) / len(cluster)
                mean_y = sum(y for x, y in cluster) / len(cluster)
                new_centroids.append((mean_x, mean_y))
                trajectory[i].append((mean_x, mean_y))
            else:
                new_centroids.append(centroids[i])

        # Parar o loop se os centroides pararem de mover
        if all(math.isclose(new_centroids[i][0], centroids[i][0], abs_tol=tolerance) and
               math.isclose(new_centroids[i][1], centroids[i][1], abs_tol=tolerance) for i in range(num_clusters)):
            break
        centroids = new_centroids

    # Cálculo das potências
    cluster_powers = []
    for i, cluster in enumerate(clusters):
        if cluster:
            centroid = centroids[i]
            cluster_head = find_cluster_head(cluster, centroid)
            total_transmission_power = sum(friis_transmission_power(pt_transceiver, gt, gr, lambda_, calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1])) for sensor in cluster if calculate_distance(sensor[0], cluster_head[0], sensor[1], cluster_head[1]) > 0)
            d_to_sink = calculate_distance(cluster_head[0], sink_position[0], cluster_head[1], sink_position[1])
            ch_to_sink_power = friis_transmission_power(pt_transceiver, gt, gr, lambda_, d_to_sink)
            cluster_powers.append({
                'cluster_index': i + 1,
                'cluster_head': cluster_head,
                'total_transmission_power': total_transmission_power,
                'ch_to_sink_power': ch_to_sink_power,
                'distance_to_sink': d_to_sink,
                'total_sensors': len(cluster)
            })

    return cluster_powers, iteration

# Código principal para gerar gráficos e tabela
def main():
    inicio = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    print("Início:", inicio)

    FREQ = 2.4 * 10**9
    PrdBm = -80
    Pt_transceiver = 0.001
    Pr = 10 ** (PrdBm / 10) / 1000
    LAMBDA = 299792458 / FREQ
    GT = 1.0
    GR = 1.0
    sink_position = (100, 50)
    totSensors = 128
    points = generate_points(totSensors, 100)
    cluster_sizes = [4, 8, 16, 32, 64, 128, 256, 512]

    all_cluster_data = {}
    for num_clusters in cluster_sizes:
        cluster_data, iteration = simulate_clusters(num_clusters, points, sink_position, save_path, LAMBDA, GT, GR, Pt_transceiver)
        all_cluster_data[num_clusters] = {'data': cluster_data, 'iteration': iteration}

    df_all_data = pd.concat([pd.DataFrame(all_cluster_data[nc]['data']).assign(Clusters=nc, Iterações=all_cluster_data[nc]['iteration']) for nc in cluster_sizes], ignore_index=True)
    df_all_data['Relacao_CH_Sink_Potencia_Total'] = df_all_data['ch_to_sink_power'] / df_all_data['total_transmission_power']
    df_all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all_data.dropna(subset=['Relacao_CH_Sink_Potencia_Total'], inplace=True)

    # Tabela de eficiência energética
    efficiency_data = df_all_data[['Clusters', 'total_transmission_power', 'ch_to_sink_power', 'Relacao_CH_Sink_Potencia_Total']].sort_values('Relacao_CH_Sink_Potencia_Total')
    efficiency_data.to_csv(os.path.join(save_path, 'Tabela_Eficiencia_Energetica.csv'), index=False)

    # Gerar e salvar gráficos de 1 a 10
    for i, num_clusters in enumerate(cluster_sizes, start=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        subset = df_all_data[df_all_data['Clusters'] == num_clusters]
        ax.bar(subset['cluster_index'], subset['total_transmission_power'], label="Total Transmissão", color='blue')
        ax.bar(subset['cluster_index'], subset['ch_to_sink_power'], label="CH-Sink", color='red', alpha=0.6)
        ax.set_title(f"Potências por Cluster ({num_clusters} Clusters)")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Potência (W)")
        ax.legend(fontsize=8)
        plt.savefig(os.path.join(save_path, f"Potencias_{num_clusters}Clusters.png"), bbox_inches='tight')
        plt.show()

    # Gráfico 6: Potência média de transmissão dos sensores para o Cluster Head
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_power_per_sensor = df_all_data.groupby('Clusters')['total_transmission_power'].mean()
    ax.plot(avg_power_per_sensor.index, avg_power_per_sensor.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Potência Média dos Sensores para Cluster Head por Número de Clusters")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Média de Transmissão (W)")
    plt.savefig(os.path.join(save_path, "Potencia_Media_Sensores_CH.png"), bbox_inches='tight')
    plt.show()

    # Gráfico 7: Potências CH-Sink de cada cluster
    fig, ax = plt.subplots()
    ax.bar(df_all_data['Clusters'], df_all_data['ch_to_sink_power'], color='cyan')
    ax.set_title("Potência CH-Sink de cada Cluster")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência CH-Sink (W)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Potencia_CH_Sink_por_Cluster.png"))
    plt.show()

    # Gráfico 8: Relação Potência CH-Sink / Potência Total
    fig, ax = plt.subplots()
    ax.bar(df_all_data['Clusters'], df_all_data['Relacao_CH_Sink_Potencia_Total'], color='blue')
    ax.set_title("Relação Potência CH-Sink / Potência Total")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Relação Potência CH-Sink / Potência Total")
    plt.xticks(rotation=45)
    ax.set_ylim(0, df_all_data['Relacao_CH_Sink_Potencia_Total'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Relacao_Potencia_CH_Sink_Total.png"))
    plt.show()

    # Gráfico 9: Eficiência Energética de Transmissão de Sinal Wireless com K-Means
    # Gráfico 9: Eficiência Energética de Transmissão de Sinal Wireless com K-Means
    fig, ax = plt.subplots()
    avg_power_per_sensor = df_all_data.groupby('Clusters')['total_transmission_power'].mean() / df_all_data.groupby('Clusters')['total_transmission_power'].sum()
    ax.plot(avg_power_per_sensor.index, avg_power_per_sensor.values, marker='o', linestyle='-', color='purple')
    ax.set_title("Eficiência Energética de Transmissão de Sinal Wireless com K-Means")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Med-Sens-CH / Potência Transm.Total")
    plt.savefig(os.path.join(save_path, "Eficiencia_Energetica.png"), bbox_inches='tight')
    plt.show()

    # Gráfico 10: Potência total de transmissão do CH-Sink por cluster (4 a 512)
    fig, ax = plt.subplots()
    ch_sink_power_total = df_all_data.groupby('Clusters')['ch_to_sink_power'].sum()
    ax.plot(ch_sink_power_total.index, ch_sink_power_total.values, marker='s', linestyle='--', color='red')
    ax.set_title("Potência Total CH-Sink por Número de Clusters")
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Potência Total CH-Sink (W)")
    plt.savefig(os.path.join(save_path, "Potencia_Total_CH_Sink.png"), bbox_inches='tight')
    plt.show()

    print("Final:", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

if __name__ == "__main__":
    main()

