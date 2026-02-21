import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import random
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "http://localhost:5000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", # Keep for CRA compatibility
        "http://localhost:5173", # Standard Vite port
        "http://localhost:5174",
        "https://pathfinderlk.netlify.app/"  # Your current Vite port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class Location(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    open_time: int 
    close_time: int
    original_index: Optional[int] = None 
    arrival_time: Optional[str] = None
    violation: Optional[bool] = False

class RouteRequest(BaseModel):
    mode: str
    locations: List[Location]

# --- 1. REAL OSRM MATRIX ---
def get_osrm_matrix(locations):
    coords = ";".join([f"{loc.lng},{loc.lat}" for loc in locations])
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coords}?annotations=duration"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "durations" in data:
                # OSRM returns seconds. Convert to HOURS.
                matrix_seconds = np.array(data["durations"])
                return matrix_seconds / 3600.0
    except Exception as e:
        print(f"OSRM Matrix Error: {e}")
    
    # Fallback to Euclidean
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.sqrt((locations[i].lat - locations[j].lat)**2 + 
                           (locations[i].lng - locations[j].lng)**2) * 111.0
            matrix[i][j] = dist / 30.0 
    return matrix

# --- 2. OSRM VISUALS ---
def get_road_geometry(sorted_locations):
    coords = ";".join([f"{loc.lng},{loc.lat}" for loc in sorted_locations])
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coords}?overview=full&geometries=geojson"
    try:
        response = requests.get(url)
        if response.status_code == 200 and "routes" in response.json():
            data = response.json()
            if len(data["routes"]) > 0:
                geometry = data["routes"][0]["geometry"]["coordinates"]
                return [[lat, lng] for lng, lat in geometry]
    except Exception as e:
        print(f"OSRM Route Error: {e}")
    return []

# --- 3. FITNESS LOGIC ---
def calculate_cost(route, matrix):
    current_time = 8.0 
    total_cost = 0.0
    service_time = 0.5 

    for i in range(len(route) - 1):
        u_idx = route[i].original_index
        v_idx = route[i+1].original_index
        
        # Travel Cost
        travel_time = matrix[u_idx][v_idx]
        total_cost += travel_time # Minimize Travel Time is Priority #1
        current_time += travel_time
        
        # Time Window Penalties
        v = route[i+1]
        open_t = float(v.open_time)
        close_t = float(v.close_time)
        
        if current_time > close_t:
            penalty = (current_time - close_t) * 5000 
            total_cost += penalty
        elif current_time < open_t:
            wait_time = open_t - current_time
            total_cost += wait_time # Wait is bad, but better than late
            current_time = open_t
            
        current_time += service_time

    return total_cost

# --- 4. HEURISTIC: NEAREST NEIGHBOR (The Fix) ---
def create_nearest_neighbor_route(locations, matrix):
    # Start at Depot (Index 0)
    route = [locations[0]]
    unvisited = set(range(1, len(locations))) # Indices of other stops
    current_idx = 0
    
    while unvisited:
        # Find closest unvisited node to current node
        nearest_idx = min(unvisited, key=lambda x: matrix[current_idx][x])
        route.append(locations[nearest_idx])
        unvisited.remove(nearest_idx)
        current_idx = nearest_idx
        
    return route

# --- 5. SCHEDULE APPLIER ---
def apply_schedule(route, matrix):
    current_time = 8.0
    service_time = 0.5
    final_schedule = []
    
    start_node = route[0].model_copy()
    start_node.arrival_time = "08:00"
    final_schedule.append(start_node)

    for i in range(len(route) - 1):
        u_idx = route[i].original_index
        v_idx = route[i+1].original_index
        
        travel_time = matrix[u_idx][v_idx]
        current_time += travel_time
        
        v = route[i+1]
        open_t = float(v.open_time)
        close_t = float(v.close_time)
        violation = False
        
        if current_time > close_t:
            violation = True
        elif current_time < open_t:
            current_time = open_t
            
        hours = int(current_time)
        minutes = int((current_time - hours) * 60)
        time_str = f"{hours:02d}:{minutes:02d}"
        
        new_loc = v.model_copy()
        new_loc.arrival_time = time_str
        new_loc.violation = violation
        final_schedule.append(new_loc)
        current_time += service_time
        
    return final_schedule

# --- 6. GENETIC ALGORITHM ---
def solve_ga(locations):
    # Set Indices
    for idx, loc in enumerate(locations):
        loc.original_index = idx
        
    matrix = get_osrm_matrix(locations)
    
    pop_size = 100
    generations = 500
    
    start_node = locations[0]
    other_nodes = locations[1:]
    
    population = []
    
    # A. Inject "Smart" Routes (Nearest Neighbor)
    # This guarantees we start with a logical path, not random mess
    nn_route = create_nearest_neighbor_route(locations, matrix)
    population.append(nn_route)
    
    # B. Fill rest with Random Routes
    for _ in range(pop_size - 1):
        shuffled = random.sample(other_nodes, len(other_nodes))
        route = [start_node] + shuffled
        population.append(route)
    
    # Evolution
    for _ in range(generations):
        population.sort(key=lambda r: calculate_cost(r, matrix))
        
        survivors = population[:pop_size//2]
        next_gen = survivors[:]
        
        while len(next_gen) < pop_size:
            parent = random.choice(survivors)
            child = parent[:] 
            
            # Mix of mutations
            if random.random() < 0.3 and len(child) > 2: # Swap
                idx1, idx2 = random.sample(range(1, len(child)), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            elif random.random() < 0.3 and len(child) > 3: # Reverse (Untangle)
                idx1, idx2 = sorted(random.sample(range(1, len(child)), 2))
                child[idx1:idx2+1] = reversed(child[idx1:idx2+1])
                
            next_gen.append(child)
        population = next_gen

    best_route = population[0]
    return apply_schedule(best_route, matrix)

# --- ENDPOINTS ---
@app.post("/api/optimize")
def optimize_route(request: RouteRequest):
    optimized_stops = solve_ga(request.locations)
    route_shape = get_road_geometry(optimized_stops)
    return {
        "optimized_stops": optimized_stops,
        "route_shape": route_shape
    }

@app.get("/api/sos")
def trigger_sos():
    return {
        "message": "SOS Signal Received",
        "nearest_provider": "Kumara Auto Works (Rank 1 - TOPSIS Score 0.92)",
        "eta": "15 mins"
    }



@app.get("/api/geocoder/reverse")
def reverse_geocode(lat: float, lon: float):
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
    headers = {
        'User-Agent': 'Pathfinder_University_Project_Yasas'
    }
    try:
        response = requests.get(url, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)