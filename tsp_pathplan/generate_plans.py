from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from save_animations import animate_bidirectional_path_on_image, animate_path_on_image
from PIL import Image
from coordinates_list import coordinates_from_json
from python_tsp.heuristics import solve_tsp_local_search
from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image, draw_lines_from_coordinates
from recalculate_to_latlon import recalculate_coordinates, read_coordinates_from_csv, coords_to_percentage
from time import time
import os
import json
import numpy as np
import Astar
import Dijkstra
import Bi_Astar
import RRT_Connect
import shutil

files_to_delete = [
    'fly_around_coordinates.txt',
    'avoid_coordinates.txt',
    'A*_coordinates.txt',
    'Dijkstra_coordinates.txt',
    'Bi_A*_coordinates.txt',
    'RRT_Connect_coordinates.txt',
    'pathDict.txt'
]

for file_name in files_to_delete:
    if os.path.exists(file_name):
        os.remove(file_name)

# Function to delete and recreate a directory
def recreate_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  
    os.makedirs(directory)  

# Mission directory
mission_directory = 'created_missions'
recreate_directory(mission_directory)
for directory in ['fly_around', 'avoid', 'A*', 'Dijkstra', 'RRT_Connect', 'Bi_A*']:
    # Identified new data directories for fly_around, avoid, and path
    recreate_directory('identified_new_data_'+directory)

list_of_the_resulted_coordinates_percentage_fly_around = []
list_of_the_resulted_coordinates_lat_lon_fly_around = []
list_of_the_resulted_coordinates_percentage_avoid = []
list_of_the_resulted_coordinates_lat_lon_avoid = []
list_of_optimized_coordinates = []

samples_number = 30
print('Number of samples',samples_number)

# load the processor
processor = AutoProcessor.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# 1. Step 1: Extract object types from the user's input command using the LLM

# llm = ChatOpenAI(api_key='', model_name='gpt-4o', temperature=0)
#
# step_1_template = """
# Extract all types of objects the drone needs to find from the following mission description:
# "{command}"
#
# Output the result in JSON format with a list of object types.
# Example output:
# {{
#     "object_types_fly_around": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"],
#     "object_types_avoid": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
# }}
# """
#
# step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)
#
# step_1_chain = step_1_prompt | llm
#
# example_objects = """
# {
#     "village_1": {"type": "village", "coordinates": [1.5, 3.5]},
#     "village_2": {"type": "village", "coordinates": [2.5, 6.0]},
#     "airfield": {"type": "airfield", "coordinates": [8.0, 6.5]}
# }
# """

def find_objects(json_input, samples_number):
    """
    Process the mission description to find object coordinates on the map and return optimized coordinates using TSP.
    """
    result_coordinates_fly_around = ""
    result_coordinates_avoid = ""
    find_objects_json_input = json.loads(json_input.replace("`", "").replace("json",""))
    pathDict = {}

    for i in range(len(find_objects_json_input["object_types_fly_around"])):
        sample_fly_around = find_objects_json_input["object_types_fly_around"][i]
    
    for i in range(len(find_objects_json_input["object_types_avoid"])):
        sample_avoid = find_objects_json_input["object_types_avoid"][i]
    
    for num in range(1, samples_number - 28 + 1):

        avoid_pixels = []
        parsed_points_avoid = {}

        start_VLM = time()
        # Open file for writing
        with open('fly_around_coordinates.txt', 'a') as file:
            
            print('-------------------------------------------------------------------')
            print(f"Processing image {num}")
            # Process the image and text
            inputs = processor.process(
                images=[Image.open(f'benchmark-UAV-VLPA-nano-30/images/{num}.jpg')],
                text=f'''
                Point all the {sample_fly_around}
                '''
            )
    
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
            # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
    
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text_fly_around = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
            if "There are none." not in generated_text_fly_around:

                parsed_points_fly_around = parse_points(generated_text_fly_around)
                parsed_points_fly_around = {'home_position': {'type': 'home', 'coordinates': [10, 10]}} | parsed_points_fly_around
                print('Parsed points fly_around:\n',parsed_points_fly_around)

                csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
                coordinates_dict = read_coordinates_from_csv(csv_file_path)

                optimized_coordinates = tsp_optimized_coordinates(parsed_points_fly_around)

                pts, width, height = image_discretization(f'benchmark-UAV-VLPA-nano-30/images/{num}.jpg', discretization_step=5)
                fly_around_pixels = coordinates_from_json(optimized_coordinates, width, height)
                result_coordinates_fly_around = recalculate_coordinates(optimized_coordinates, num, coordinates_dict)

                image_to_draw = f'identified_new_data_fly_around/identified{num}.png'
                draw_dots_and_lines_on_image(f'benchmark-UAV-VLPA-nano-30/images/{num}.jpg', optimized_coordinates, output_path=image_to_draw, obstacle=False)

                # Writing the coordinates to the file in the required format
                file.write(f"Image {num}:\n")
                for building, data in result_coordinates_fly_around.items():
                    file.write(f"  {building.replace('_', ' ').title()}:\n")
                    file.write(f"    Latitude: {data['coordinates'][0]}\n")
                    file.write(f"    Longitude: {data['coordinates'][1]}\n")
                
                file.write("\n")
    
        with open('avoid_coordinates.txt', 'a') as file:
            # Process the image and text
            inputs = processor.process(
                images=[Image.open(f'benchmark-UAV-VLPA-nano-30/images/{num}.jpg')],
                text=f'''
                Point all the {sample_avoid}
                '''
            )
    
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
            # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
    
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text_avoid = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if "There are none." not in generated_text_avoid:

                parsed_points_avoid = parse_points(generated_text_avoid)
                print('\nParsed points avoid:\n',parsed_points_avoid)
    
                csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
                coordinates_dict = read_coordinates_from_csv(csv_file_path)
    
                result_coordinates_avoid = recalculate_coordinates(parsed_points_avoid, num, coordinates_dict)
                
                # print(result_coordinates_avoid)

                avoid_pixels = coordinates_from_json(parsed_points_avoid, width, height)
                if num == 11:
                    avoid_pixels.pop(0)
                    parsed_points_avoid.pop('lakes_1', None)

                draw_dots_and_lines_on_image(image_to_draw, parsed_points_avoid, output_path=f'identified_new_data_avoid/identified{num}.png', obstacle=True)

                # Writing the coordinates to the file in the required format
                file.write(f"Image {num}:\n")
                for lake, data in result_coordinates_avoid.items():
                    file.write(f"  {lake.replace('_', ' ').title()}:\n")
                    file.write(f"    Latitude: {data['coordinates'][0]}\n")
                    file.write(f"    Longitude: {data['coordinates'][1]}\n")
                
                file.write("\n")

            else:
                print('\nParsed points avoid:\n',parsed_points_avoid)
                draw_dots_and_lines_on_image(image_to_draw, parsed_points_avoid, output_path=f'identified_new_data_avoid/identified{num}.png', obstacle=True)

            print('\nVLM execution took:', time() - start_VLM, 's')

        start_grid = time()

        for xy in fly_around_pixels:
            pts.append(xy)

        graph, coordinates, N = graphCreation(pts, avoid_pixels)
        adjacency_list = adjacencyListCreation(graph)  # pts and x,y of start and final points

        print('\nGrid creation took:', time() - start_grid, 's')

        image_to_draw = f'identified_new_data_avoid/identified{num}.png'

        algo_paths = [
            {
                "name": "A*",
                "gif": "GIFs/A*.gif",
                "path_list": pathplanningAstar
            },
            {
                "name": "Dijkstra",
                "gif": "GIFs/Dijkstra.gif",
                "path_list": pathplanningDijkstra
            },
            {
                "name": "RRT_Connect",
                "gif": "GIFs/RRTConnect.gif",
                "path_list": pathplanningRRTConnect
            },
            {
                "name": "Bi_A*",
                "gif": "GIFs/Bi_A*.gif",
                "path_list": pathplanningBiAstar
            }
        ]

        for algo in algo_paths:
            flyAround = []
            last_vertex = []
            startTime = time()

            print('\nFinding the path with ' + algo["name"] + '...\n')
            for i in range(len(fly_around_pixels), 1, -1):
                pathList = algo["path_list"](adjacency_list, coordinates[:N-i+3], N-i+2)
                for v in pathList:
                    flyAround.append(coordinates[v])
                if not flyAround:
                    break
                last_vertex = flyAround.pop()

            if not last_vertex:
                continue

            flyAround.append(last_vertex)
            # print('Final Path by ' + algo["name"] + ': \n', flyAround)
            draw_lines_from_coordinates(image_to_draw, flyAround, output_path='identified_new_data_'+algo["name"]+f'/identified{num}.png', obstacle=False)

            percentage_json = coords_to_percentage(flyAround, image_to_draw)
            result_path_coordinates = recalculate_coordinates(percentage_json, num, coordinates_dict)

            # list_of_the_resulted_coordinates_percentage_avoid.append(parsed_points_avoid)
            list_of_the_resulted_coordinates_lat_lon_fly_around.append(result_path_coordinates)

            # print(result_path_coordinates)

            with open(algo["name"] + '_coordinates.txt', 'a') as file:

                file.write(f"Image {num}:\n")
                for obj, data in result_path_coordinates.items():
                    file.write(f"  {obj.replace('_', ' ').title()}:\n")
                    file.write(f"    Latitude: {data['coordinates'][0]}\n")
                    file.write(f"    Longitude: {data['coordinates'][1]}\n")

                file.write("\n")

            if algo["name"] not in pathDict:
                pathDict[algo["name"]] = 0

            pathDict[algo["name"]] += time() - startTime

    with open('pathDict.txt', 'a') as file:
        file.write(str(pathDict))
    file.close()

    return json.dumps(result_coordinates_fly_around), json.dumps(result_coordinates_avoid), list_of_the_resulted_coordinates_lat_lon_fly_around


# Full pipeline
def generate_drone_mission(command):
    # Step 1: Extract object types
    # object_types_response = step_1_chain.invoke({"command": command})
    # object_types_json = object_types_response.content  # Use 'content' to get the actual response text

    object_types_json = """
    ```json{
        "object_types_fly_around": ["building"],
        "object_types_avoid": ["lakes"]
    }
    """

    # Step 2: Find objects on the map
    t1_find_objects = time()
    objects_json_fly_around, objects_json_fly_avoid, list_of_the_resulted_coordinates_lat_lon_fly_around = find_objects(object_types_json, samples_number)

    del_t_find_objects = (time() - t1_find_objects)/60
    
    # Step 3: Generate the flight plan
    t1_generate_drone_mission = time()

    if not list_of_the_resulted_coordinates_lat_lon_fly_around:
        return del_t_find_objects, time() - t1_generate_drone_mission

    for i, image_data in enumerate(list_of_the_resulted_coordinates_lat_lon_fly_around, start=1):
        with open(f"created_missions/mission{i}.txt","w") as f:
            f.write("arm throttle\n")
            f.write("mode guided\n")
            f.write("takeoff 100\n")
            
            for obj_key, obj_info in image_data.items():
                coords = obj_info["coordinates"]
                lat, lon = coords[0], coords[1]
                
                f.write(f"guided({lat}, {lon}, 100)\n")
                f.write("mode circle\n")
            
            f.write("mode rtl\n")
            f.write("disarm\n")
    
    del_t_generate_drone_mission = (time() - t1_generate_drone_mission)/60
    
    return del_t_find_objects, del_t_generate_drone_mission  # Return the response text from AIMessage


def image_discretization(image_path, discretization_step):

    image = Image.open(image_path)
    width, height = image.size

    pts = []
    for i in range(0, width, discretization_step):
        for j in range(0, height, discretization_step):
            pts.append([i, j])
    return pts, width, height


### Here goes the creation of the graph edges ###
def graphCreation(pts, obstacles_list):
    
    coordinates = [[]]
    coordinates += pts.copy()

    N = len(coordinates) - 1
    obstacles = []
    obstacle_rad = 25
    for obstacle in obstacles_list:
        for i in range(1, N + 1):
            rad = 0.5 * 5 ** .5 * obstacle_rad
            d = ((obstacle[0] - coordinates[i][0]) ** 2 +
                 (obstacle[1] - coordinates[i][1]) ** 2) ** .5
            if d <= rad:
                obstacles.append(i)

    # region GraphConstruction
    graph = []
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            dist = ((coordinates[i][0] - coordinates[j][0]) ** 2 +
                    (coordinates[i][1] - coordinates[j][1]) ** 2) ** .5
            if dist < obstacle_rad and i not in obstacles and j not in obstacles:
                graph.append([i, j, dist])
    return graph, coordinates, N
    # endregion


def adjacencyListCreation(graph):
    # example of adjacency list (or rather map)
    # adjacency_list =
    # {'1': [('2', 1.2), ('3', 3.4), ('4', 7.9)],
    # '2': [('4', 5.5)],
    # '3': [('4', 12.6)]}

    adjacency_list = {i + 1: [] for i in range(len(graph))}

    for i in range(len(graph)):
        a, b, ll = map(float, graph[i])
        a, b = int(a), int(b)
        adjacency_list[a].append((b, ll))
        adjacency_list[b].append((a, ll))

    return adjacency_list


def pathplanningAstar(adjacency_list, coordinates, N):
    return Astar.Graph(adjacency_list).a_star_algorithm(coordinates, N)


def pathplanningDijkstra(adjacency_list, coordinates, N):
    return Dijkstra.shortestPathFastDijkstra(adjacency_list, N)


def pathplanningBiAstar(adjacency_list, coordinates, N):
    return Bi_Astar.pathplanningBidirectionalAStar(adjacency_list, coordinates, N)


def pathplanningRRTConnect(adjacency_list, coordinates, N):
    return RRT_Connect.pathplanningRRT(adjacency_list, coordinates, N)


# TSP Function: Solve TSP using Euclidean distance between coordinates
def tsp_optimized_coordinates(coordinates):
    """
    Use TSP to optimize the path for a drone using Euclidean distance between coordinates.
    The coordinates should be in a dictionary format where each key represents a building
    and each value is another dictionary with 'type' and 'coordinates' as keys.
    """
    coords = []
    names = []

    # Extract coordinates and names from the input dictionary
    for name, data in coordinates.items():
        if 'coordinates' in data:
            coords.append(data['coordinates'])
            names.append(name)

    # Calculate Euclidean distance matrix
    num_coords = len(coords)
    distance_matrix = np.zeros((num_coords, num_coords))

    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            distance_matrix[i][j] = distance_matrix[j][i] = dist

    distance_matrix[:, 0] = 0
    # Solve the TSP using local search
    optimal_order = solve_tsp_local_search(distance_matrix)

    # Ensure that the optimal_order is a list of indices, not a list of lists
    if isinstance(optimal_order[0], list):  # This checks if it's a nested list
        optimal_order = optimal_order[0]  # Flatten the list

    # Reorder the coordinates according to the optimal TSP path
    optimized_coordinates = {names[i]: coordinates[names[i]] for i in optimal_order}

    return optimized_coordinates


def animate_paths(img_path, algo_paths, coordinates, obstaclesList, img):
    for algo in algo_paths:
        if algo["name"] in ["RRT_Connect", "Bi-A*"]:
            animate_bidirectional_path_on_image(
                img_path,
                coordinates,
                obstaclesList,
                algo["path_list"][1],
                algo["path_list"][2],
                output_gif=algo["gif"],
                interval=170
            )
        else:
            animate_path_on_image(
                img_path,
                coordinates,
                obstaclesList,
                algo["path_list"],
                output_gif=algo["gif"],
                interval=200
            )

# Example usage:
command = """
    Create a flight plan for the quadcopter to fly around each of the building
    at the height 100m return to home and land at the take-off point. Avoid lakes.
    """

# Run the full pipeline
vlm_model_time, mission_generation_time = generate_drone_mission(command)
total_computational_time = vlm_model_time + mission_generation_time

# Evaluation time
print('-------------------------------------------------------------------')
print('Time to get VLM results: ', vlm_model_time, 'mins')
print('Time to get Mission Text files: ', mission_generation_time, 'mins')
print('Total Computational Time: ', total_computational_time, 'mins')
