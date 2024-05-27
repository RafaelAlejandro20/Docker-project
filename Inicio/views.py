from django.http import HttpResponse
from django.shortcuts import render
import osmnx as ox
import networkx as nx
import random
import math
import folium
import json

G = ox.graph_from_place('Jilotepec, Estado de Mexico, Mexico',network_type='drive')

def tupla(texto):
    lat, lon = map(float, texto.split(','))
    return (lat, lon)

def nodo_cercano(ruta):
    nodo_cercano = ox.nearest_nodes(G, ruta[1], ruta[0])
    return nodo_cercano

def distancia(nodo_uno, nodo_dos):
    long_metros = nx.shortest_path_length(G, nodo_uno, nodo_dos, weight='length')
    long_kilometros = long_metros / 1000
    return long_kilometros

gas_reg = 23.45
gas_pre = 25.28
die = 25.07
kil = 12

def precioGasolina(hid, kil):
    costo = hid / kil
    return costo

def gasto(kil):
    gasto_hid = 1 / kil
    return gasto_hid

def index(request):
    return render(request, 'index.html', {})

def ruta(request):
    txt_res1 = request.GET['partida']
    txt_res2 = request.GET['destino']

    txtres1 = tupla(txt_res1)
    txtres2 = tupla(txt_res2)

    lista1 = list(txtres1)
    lista2 = list(txtres2)

    res1 = nodo_cercano(txtres1)
    res2 = nodo_cercano(txtres2)
    total = distancia(res1, res2)

    shortest_path = ox.shortest_path(G, res1, res2, weight='length')
    route_coordinates = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_path]

    gas = total * precioGasolina(gas_reg, kil)

    gas_hid = total * gasto(kil)

    return render(request, 'ruta.html', {'kilometros':total, 'route_coordinates': route_coordinates, 'nodo_partida': lista1, 'nodo_destino': lista2, 'np': txt_res1, 'nd': txt_res2, 'gas':gas, 'gasto':gas_hid})

def genetica(request):
    return render(request, 'crear.html', {})

def resolucion(request):
    nod_lat = request.GET['lat[]']
    nod_lon = request.GET['lon[]']
    tarifa = request.GET['tarifa']

    lat_list = [float(lat) for lat in nod_lat.split(',')]
    lon_list = [float(lon) for lon in nod_lon.split(',')]
    nodes_list = list(zip(lat_list, lon_list))

    def calcular_distancias(nodes):
        distancias = {}
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    distancias[(node1, node2)] = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        return distancias

    distancias = calcular_distancias(nodes_list)

    def fitness(individual):
        distance = 0
        for i in range(len(individual) - 1):
            distance += distancias[(individual[i], individual[i+1])]
        distance += distancias[(individual[-1], individual[0])]
        return distance

    def create_individual():
        return random.sample(nodes_list, len(nodes_list))

    def select_parents(population, fitnesses, tournament_size=3):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        tournament.sort(key=lambda x: x[1])
        return tournament[0][0]

    def crossover(parent1, parent2):
        child = [-1] * len(parent1)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child[start:end] = parent1[start:end]
        p2_index = 0
        for i in range(len(child)):
            if child[i] == -1:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]
        return child

    def mutate(individual):
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def create_next_generation(population, fitnesses, elitism=True):
        new_population = []
        if elitism:
            elites = sorted(list(zip(population, fitnesses)), key=lambda x: x[1])[:2]
            new_population.extend([elite[0] for elite in elites])
        while len(new_population) < len(population):
            parent1 = select_parents(population, fitnesses)
            parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        return new_population

    population_size = 20
    generations = 400
    crossover_rate = 0.7
    mutation_rate = 0.2

    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        fitnesses = [fitness(individual) for individual in population]
        population = create_next_generation(population, fitnesses)

    best_individual = min(population, key=fitness)

    best_individual_indices = [nodes_list.index(node) for node in best_individual]

    mymap = folium.Map(location=nodes_list[0], zoom_start=12)

    for node_id, (lat, lon) in enumerate(nodes_list):
        folium.Marker(location=(lat, lon), popup=f'Nodo {node_id}: {lat}, {lon}').add_to(mymap)

    folium.PolyLine([(nodes_list[node_idx][0], nodes_list[node_idx][1]) for node_idx in best_individual_indices],
                    color="green", weight=2.5, opacity=1).add_to(mymap)

    mymap

    print(f"Mejor recorrido encontrado: {best_individual}")
    print(f"Distancia del recorrido: {fitness(best_individual):.2f}")

    def haversine_distance(coord1, coord2):
        R = 6371000
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance

    distancia_metros = sum(haversine_distance(nodes_list[best_individual_indices[i]], nodes_list[best_individual_indices[i+1]]) for i in range(len(best_individual_indices) - 1))
    distancia_metros += haversine_distance(nodes_list[best_individual_indices[-1]], nodes_list[best_individual_indices[0]])

    distancia_kilometros = distancia_metros / 1000

    print(f"Distancia del recorrido (metros): {distancia_metros:.2f}Mtrs")
    print(f"Distancia del recorrido (kilómetros): {distancia_kilometros:.2f}Km")

    ruta_mapa = json.dumps(best_individual)

    gas = distancia_kilometros * precioGasolina(gas_reg, kil)

    gas_hid = distancia_kilometros * gasto(kil)

    return render(request, 'resolucion.html', {'ruta':ruta_mapa, 'dis_metros':distancia_metros, 'dis_kilometros': distancia_kilometros, 'gasolina': gas, 'gas_hid': gas_hid, 'tar':tarifa})
