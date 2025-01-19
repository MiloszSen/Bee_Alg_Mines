from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import random
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

def load_data(file_obj):
    """
    Odczytuje dane z pliku Excel (plik w postaci obiektu).
    Zakładamy, że arkusze: Mines, Cities, Resources, Routes.
    """
    excel_data = pd.ExcelFile(file_obj)

    mines_df = excel_data.parse("Mines")
    cities_df = excel_data.parse("Cities")
    resources_df = excel_data.parse("Resources")
    routes_df = excel_data.parse("Routes")

    mines = mines_df.set_index("Mine").to_dict("index")
    cities = cities_df.set_index("City").to_dict("index")
    resources = resources_df.set_index("Resource").to_dict("index")

    for city, data in cities.items():
        if data["MinDemand"] > data["MaxDemand"]:
            cities[city]["MinDemand"] = data["MaxDemand"]

    return mines, cities, resources, routes_df

def create_graph(routes_df):
    G = nx.Graph()
    for _, row in routes_df.iterrows():
        G.add_edge(row["From"], row["To"], weight=row["Distance"])
    return G

def generate_initial_population(mines, cities, truck_capacity, population_size=10):
    population = []
    for _ in range(population_size):
        route = []
        city_deliveries = {city: 0 for city in cities}
        for mine, mine_data in mines.items():
            resource = mine_data["Resource"]
            deliveries = []
            for city, city_data in cities.items():
                max_additional_demand = max(0, city_data["MaxDemand"] - city_deliveries[city])
                remaining_demand = random.randint(city_data["MinDemand"], city_data["MaxDemand"])
                remaining_demand = min(remaining_demand, max_additional_demand)
                while remaining_demand > 0:
                    amount = min(truck_capacity, remaining_demand)
                    deliveries.append((city, amount))
                    city_deliveries[city] += amount
                    remaining_demand -= amount
            if deliveries:
                route.append((mine, deliveries, resource))
        if route:
            population.append(route)
    return population

def evaluate_route(route, resources, graph, max_co2_emission=500, co2_penalty=10,
                   transportation_cost_per_km=1, truck_capacity=30,
                   cities=None, truck_start_cost=100, max_days=5,
                   num_trucks=1, truck_speed=60):
    total_profit = 0
    total_co2 = 0
    city_deliveries = {city: 0 for city in cities}
    max_daily_distance = 12 * truck_speed
    day = 1
    daily_distance = 0
    trucks_used = 0

    for mine, deliveries, resource in route:
        for city, amount in deliveries:
            if amount < 5:
                continue
            if amount > truck_capacity:
                amount = truck_capacity
            if city_deliveries[city] + amount > cities[city]["MaxDemand"]:
                amount = max(0, cities[city]["MaxDemand"] - city_deliveries[city])
            if amount <= 0:
                continue

            city_deliveries[city] += amount

            extraction_cost = resources[resource]["ExtractionCost"] * amount
            co2_emission = resources[resource]["CO2Emission"] * amount
            selling_price = resources[resource]["Price"] * amount

            distance = None
            if graph.has_edge(mine, city):
                distance = graph[mine][city]['weight']
            else:
                try:
                    path = nx.shortest_path(graph, mine, city, weight='weight')
                    distance = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                except nx.NetworkXNoPath:
                    continue

            if distance is not None:
                if daily_distance + distance > max_daily_distance:
                    trucks_used += 1
                    if trucks_used >= num_trucks:
                        day += 1
                        trucks_used = 0
                        daily_distance = 0
                        if day > max_days:
                            continue
                daily_distance += distance

                transport_cost = (transportation_cost_per_km * distance * amount) + truck_start_cost
                profit = selling_price - extraction_cost - transport_cost
                if profit > 0:
                    total_profit += profit

                total_co2 += co2_emission
    if total_co2 > max_co2_emission:
        penalty = (total_co2 - max_co2_emission) * co2_penalty
        total_profit -= penalty

    return total_profit

def mutate_route(route, cities, truck_capacity):
    new_route = []
    for mine, deliveries, resource in route:
        mutated_deliveries = []
        for city, amount in deliveries:
            if random.random() < 0.3:
                amount = max(1, min(truck_capacity, amount + random.randint(-5, 5)))
            mutated_deliveries.append((city, amount))
        new_route.append((mine, mutated_deliveries, resource))
    return new_route

def bee_algorithm(mines, cities, resources, graph,
                  truck_capacity=30, population_size=10, max_iterations=10,
                  max_co2=500, num_trucks=1, max_days=5, elite_ratio=0.2):
    population = generate_initial_population(mines, cities, truck_capacity, population_size)
    profit_scores = [
        evaluate_route(
            route,
            resources,
            graph,
            max_co2_emission=max_co2,
            truck_capacity=truck_capacity,
            cities=cities,
            num_trucks=num_trucks,
            max_days=max_days
        ) for route in population
    ]

    elite_count = max(1, int(population_size * elite_ratio))
    iteration_data = [] 

    for iteration in range(max_iterations):
        start_time = time.time()

        elite_indices = sorted(range(len(population)), key=lambda i: profit_scores[i], reverse=True)[:elite_count]
        elite_population = [population[i] for i in elite_indices]

        new_population = []

        for elite_route in elite_population:
            new_population.append(mutate_route(elite_route, cities, truck_capacity))

        total_profit = sum(profit_scores)
        if total_profit != 0:
            probabilities = [score / total_profit for score in profit_scores]
        else:
            probabilities = [0] * len(profit_scores)

        for _ in range(int(population_size * 0.5)): 
            selected_index = random.choices(range(len(population)), weights=probabilities, k=1)[0]
            selected_route = population[selected_index]
            new_population.append(mutate_route(selected_route, cities, truck_capacity))

        for _ in range(int(population_size * 0.2)): 
            rand_route = generate_initial_population(mines, cities, truck_capacity, population_size=1)[0]
            new_population.append(rand_route)

        population = new_population
        profit_scores = [
            evaluate_route(
                route,
                resources,
                graph,
                max_co2_emission=max_co2,
                truck_capacity=truck_capacity,
                cities=cities,
                num_trucks=num_trucks,
                max_days=max_days
            ) for route in population
        ]

        iteration_time = time.time() - start_time
        current_best_profit = max(profit_scores) if profit_scores else 0
        iteration_data.append({
            "iteration": iteration + 1,
            "profit": current_best_profit,
            "time": iteration_time
        })

    best_index = profit_scores.index(max(profit_scores))
    return population[best_index], profit_scores[best_index], iteration_data


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run-algorithm", methods=["POST"])
def run_algorithm():
    num_trucks = int(request.form.get("numTrucks", 1))
    truck_capacity = int(request.form.get("truckCapacity", 30))
    max_days = int(request.form.get("maxDays", 5))
    pop_size = int(request.form.get("popSize", 10))
    max_iter = int(request.form.get("maxIter", 10))
    max_co2 = float(request.form.get("maxCO2", 500))

    excel_file = request.files.get("excelFile")
    if not excel_file or excel_file.filename == "":
        return jsonify({"error": "Nie wybrano pliku Excel."})

    try:
        mines, cities, resources, routes_df = load_data(excel_file)
        graph = create_graph(routes_df)
        alg_time_st = time.time()
        best_route, best_profit, iteration_data = bee_algorithm(
            mines=mines,
            cities=cities,
            resources=resources,
            graph=graph,
            truck_capacity=truck_capacity,
            population_size=pop_size,
            max_iterations=max_iter,
            max_co2=max_co2,
            num_trucks=num_trucks,
            max_days=max_days
        )
        alg_time_all = time.time() - alg_time_st

        city_deliveries_final = {city: 0 for city in cities} 

        deliveries_list = []
        for mine, deliveries, resource in best_route:
            for city, raw_amount in deliveries:
                if raw_amount < 5:
                    continue
                
                already_delivered = city_deliveries_final[city]
                max_can_deliver = cities[city]["MaxDemand"] - already_delivered
                if max_can_deliver <= 0:
                    continue

                final_amount = min(raw_amount, max_can_deliver)
                if final_amount < 5:
                   
                    continue

                
                city_deliveries_final[city] += final_amount

               
                try:
                    forward_path = nx.shortest_path(graph, mine, city, weight='weight')
                except nx.NetworkXNoPath:
                    
                    continue

                
                backward_path = []
                try:
                    backward_path = nx.shortest_path(graph, city, mine, weight='weight')
                except nx.NetworkXNoPath:
                    pass

                deliveries_list.append((forward_path, backward_path, resource, final_amount))


        deliveries_chunks = []
        edges_list_table = []

        for (forward_path, backward_path, resource, amount) in deliveries_list:
           
            for i in range(len(forward_path) - 1):
                edges_list_table.append({
                    "from": forward_path[i],
                    "to": forward_path[i+1],
                    "amount": amount,
                    "resource": resource
                })

            
            chunk_edges = []
           
            for i in range(len(forward_path) - 1):
                chunk_edges.append({
                    "from": forward_path[i],
                    "to": forward_path[i+1],
                    "amount": amount,
                    "resource": resource
                })
           
            if backward_path:
                for i in range(len(backward_path) - 1):
                    chunk_edges.append({
                        "from": backward_path[i],
                        "to": backward_path[i+1],
                        "amount": 0,  
                        "resource": resource
                    })

            if chunk_edges:
                deliveries_chunks.append(chunk_edges)

        
        assigned_anim_edges = []
        truck_index = 0
        for chunk in deliveries_chunks:
            for edge in chunk:
                edge["truckId"] = truck_index
                assigned_anim_edges.append(edge)
            truck_index = (truck_index + 1) % num_trucks

        
        all_routes = []
        for idx, row in routes_df.iterrows():
            all_routes.append({
                "from": row["From"],
                "to": row["To"],
                "distance": row["Distance"]
            })

        
        return jsonify({
            "bestProfit": best_profit,
            "routeEdges": edges_list_table,        
            "routeEdgesAnim": assigned_anim_edges, 
            "numTrucks": num_trucks,
            "allRoutes": all_routes,
            "iterationData": iteration_data,
            "algTime": alg_time_all
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
