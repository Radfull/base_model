# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pygad as pg
# import requests

# from PIL import Image
# from io import BytesIO
# from openrouteservice.directions import directions
# from openrouteservice import convert
# import contextily as ctx
# import json
# from haversine import haversine
# import pygad as pg
# from dijkstra_for_robots import Graph


# API_KEY_OPEN = "5b3ce3597851110001cf62484580fb04ffc34ec191e892a13e92a8bd"
# API_YANDEX_STATIC_MAP = '74b8473d-0114-4f09-9698-0ce984a100d1'
# API_YANDEX_GEOCODER = 'af44d231-d82a-4418-ac7e-5429ae2f732f'


# def fit_func(alg, sol, ind_sol):
#     return 0


# def get_ozon_points(addres, center_coords):
#     params_points = {
#         'll': f"{center_coords[0]},{center_coords[1]}",
#         'spn': "0.05,0.05",  # масштаб (размер области)
#         'geocode' : addres,
#         'format' : 'json',
#         'apikey': API_YANDEX_GEOCODER
#     }
#     # get_points_resp = 'https://geocode-maps.yandex.ru/v1/?apikey=YOUR_API_KEY&geocode=бул+Мухаммед+Бин+Рашид,+дом+1&format=json'
#     resp_points = requests.get('https://geocode-maps.yandex.ru/v1/', params=params_points)


#     json_data = resp_points.content.decode('utf-8')
#     data = json.loads(json_data)

#     if data["response"]["GeoObjectCollection"]["metaDataProperty"]["GeocoderResponseMetaData"]["found"] != "0":
#         first_feature = data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
#         pos = first_feature["Point"]["pos"]  # "долгота широта"
#         lon, lat = map(float, pos.split()) 
#     else:
#         print("Объект не найден.")

#     return (lon, lat)

# def make_pvz_txt():
#     df_ozon = pd.read_excel('ozon_info.xlsx',header=1, sheet_name='Список ПВЗ')
#     df_ozon.drop(['Unnamed: 0'], axis=1, inplace=True)
#     df_ozon = df_ozon[(df_ozon['Город'] == 'Казань') | (df_ozon['Город'] == 'г. Казань')]

#     print(df_ozon.head())
#     print(df_ozon.shape)
#     kazan_pvz = df_ozon['Адрес пункта'].to_list()
#     print(kazan_pvz)
#     kazan_pvz_conv = list(item.replace(', ' , '+') for item in kazan_pvz)
#     kazan_pvz_conv = list(item.replace(' ' , '+') for item in kazan_pvz_conv)
#     print(kazan_pvz_conv)

#     with open('ozon_pvz.txt', 'w+') as f:
#         for item in kazan_pvz_conv:
#             f.write('%s\n' %item)

#     print("File written successfully")
#     f.close()

# def scaler(range, coords, center_coords, img_size=[650,450]):

#     normalized_x = (coords[0] - (center_coords[0] - range)) / (2 * range)
#     normalized_y = (coords[1] - (center_coords[1] - range)) / (2 * range)
    
#     pixel_x = normalized_x * img_size[0]
#     pixel_y = img_size[1] - (normalized_y * img_size[1])
    
#     return (pixel_x, pixel_y)

# def calculate_distance_matrix(points):
#     """Создает матрицу расстояний между точками в метрах"""
#     n = len(points)
#     dist_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 dist_matrix[i][j] = haversine(
#                     (points[i][1], points[i][0]),  # (lat, lon)
#                     (points[j][1], points[j][0]),  # (lat, lon)
#                     unit='m'  # Расстояние в метрах
#                 )
#     return dist_matrix


# def get_lenght(p1,p2):
#     return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1] ** 2))

# def get_ozon_point_geocoder(center_coords):
#     query = "Казань, Ozon, пункты выдачи"
#     params_points = {
#         'll': f"{center_coords[0]},{center_coords[1]}",
#         'spn': "0.05,0.05",  # масштаб (размер области)
#         'geocode' : query,
#         'format' : 'json',
#         'apikey': API_YANDEX_GEOCODER
#     }
    
#     # url = f"https://geocode-maps.yandex.ru/1.x/?apikey={API_YANDEX_GEOCODER}&format=json&geocode={query}"
#     response = requests.get('https://geocode-maps.yandex.ru/v1/', params=params_points).json()

#     # response = requests.get(url).json()
#     points = response["response"]["GeoObjectCollection"]["featureMember"]
#     coords = []

#     for point in points:
#         coords.append(tuple(point["GeoObject"]["Point"]["pos"].split()))
    
#     return coords


# def get_shortest_route(points):
    
#     conn = {}
#     for i in range(len(points)):
#         conn[points[i]] = list()
#         for j in range(len(points)):
#             if i != j:
#                 conn[points[i]].append(points[j])
            
#     p_from = (391.5599999999557, 118.00350000000105)
#     p_to = ((279.8639999999999, 56.49750000002098))
#     graph = Graph(points, conn)
#     sh_path = graph.dijkstra_algorithm(p_from, p_to)
#     return sh_path


    
# def main():
#     # Координаты центра Казани
#     kazan_center = (49.108795, 55.796127)

#     # p = get_ozon_point_geocoder(kazan_center)
#     # print(p)
#     # make_pvz_txt()

#     # with open('ozon_pvz.txt', 'r') as f:
#     #     addresses = [line.strip() for line in f.readlines()]

#     # f.close()
#     # # print(addresses)
#     # pvz_coords = []
#     # for add in addresses:
#     #     pvz_coords.append(get_ozon_points(add, kazan_center))
    
#     # with open('ozon_pvz_points.txt', 'w') as f:
#     #     for lon, lat in pvz_coords:
#     #         f.write(f"{lon}, {lat}\n") 
#     # f.close()
#     pvz_address_points = []
#     with open('ozon_pvz_points.txt', 'r') as f:
#         for line in f.readlines():
#             lot, lat = map(float, line.split(', '))
#             if lot >= (kazan_center[0]-0.05) and lat >= (kazan_center[1] - 0.05) and lot <= (kazan_center[0]+0.05) and lat <= (kazan_center[1] + 0.05):
#                 pvz_address_points.append((lot, lat))
#             # print(lot)
#     f.close()
#     pvz_address_points_scaled = []
#     for point in pvz_address_points:
#         pvz_address_points_scaled.append(scaler(0.05, point, kazan_center))

#     # print(pvz_address_points_scaled)
#     # # print(list(i[0] for i in pvz_address_points_scaled))
#     img = Image.open('kazan_yandex_map.png')
#     plt.imshow(img)
#     plt.scatter(list(i[0] for i in pvz_address_points_scaled),list(i[1] for i in pvz_address_points_scaled), c='red')

#     sh = get_shortest_route(pvz_address_points_scaled)

#     for i in range(1,len(sh)):
#         plt.plot([sh[i-1][0], sh[i][0]], [sh[i-1][1], sh[i][1]], c='green')

#     # for point in pvz_address_points_scaled:
#     #     plt.plot(point, c='red', markersize=12, marker='o')
#     plt.show()    
#     # # Параметры запроса
#     # params = {
#     #     'll': f"{kazan_center[0]},{kazan_center[1]}",
#     #     'spn': "0.05,0.05",  # масштаб (размер области)
#     #     'size': "650,450",  # размер изображения
#     #     'l': "map",         # тип карты
#     #     'z' : '21',
#     #     'apikey': API_YANDEX_STATIC_MAP
#     # }
    
#     # # Отправляем запрос
#     # resp = requests.get('https://static-maps.yandex.ru/1.x/', params=params)
    
#     # # Проверяем успешность запроса
#     # if resp.status_code == 200:
#     #     # Открываем изображение из ответа
#     #     img = Image.open(BytesIO(resp.content))
        
#     #     # Создаем фигуру для отображения
#     #     plt.figure(figsize=(10, 8))
#     #     plt.imshow(img)
#     #     for point in pvz_coords:
#     #         plt.plot(point, c='blue')
#     #     plt.axis('off')  # Скрываем оси
#     #     plt.title('Карта Казани')
#     #     plt.show()
        
#     #     # Сохраняем изображение
#     #     img.save('kazan_yandex_map.png')
#     #     print("Карта сохранена как 'kazan_yandex_map.png'")
#     # else:
#     #     print(f"Ошибка запроса: {resp.status_code}")
#     #     print(resp.text)


# if __name__ == "__main__":
#     main()




# import time
# import openrouteservice
# import plotly.graph_objs as go
# import pandas as pd
# import plotly.express as px


# pvz_address_points_lot = []
# pvz_address_points_lat = []

# with open('ozon_pvz_points.txt', 'r') as f:
#     for line in f.readlines():
#         lot, lat = map(float, line.split(', '))
#         pvz_address_points_lot.append(lot)
#         pvz_address_points_lat.append(lat)
#         # print(lot)
# f.close()

# # Sample data
# data = {
#     'Sequence': [1, 2, 3, 4, 5, 161, 162, 163, 164],
#     'Latitude': [47.707439, 47.706535, 47.708466, 47.708198, 47.708568, 
#                  47.733784, 47.734829, 47.735678, 47.719747],
#     'Longitude': [-122.201863, -122.201832, -122.204317, -122.206690, -122.206190, 
#                   -122.191986, -122.191846, -122.190102, -122.189500],
#     'ZoneID': ['C-17.2C', 'C-17.2C', 'C-17.2C', 'C-17.3C', 'C-17.3C', 
#                'C-18.3G', 'C-18.3G', 'C-18.3G', 'C-18.2J']
# }

# pvz_data = {'Sequence' : [i for i in range(1,len(pvz_address_points_lat) + 1)],
#             'Latitude' : pvz_address_points_lot,
#             'Longitude' : pvz_address_points_lat,
#             'ZoneID' : ['C-17.2C'] * len(pvz_address_points_lot)}



# df = pd.DataFrame(pvz_data)
# print(df.head())

# # Sort by Sequence
# df = df.sort_values('Sequence')
# # Check for missing values
# print(df.isnull().sum())
# # Convert ZoneID to categorical
# df['ZoneID'] = df['ZoneID'].astype('category')
# # Get basic statistics
# print(df.describe())

# def visualize_route(master_df, api_key, route_id = None):
#     if route_id:
#         df = master_df[(master_df['RouteID']== route_id) & (master_df['Sequence']!=0)]
#     else:
#         df = master_df[master_df['Sequence']!=0]
    
#     # Create a dictionary to assign a unique color to each ZoneID using brighter colors
#     unique_zones = df['ZoneID'].unique()
#     zone_colors = {zone: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, zone in enumerate(unique_zones)}

#     # Initialize OpenRouteService client
#     client = openrouteservice.Client(key=api_key)

#     def get_route(client, start, end, retries=5, delay=10):
#         for i in range(retries):
#             try:
#                 return client.directions(coordinates=[start, end], profile='driving-car', format='geojson')
#             except openrouteservice.exceptions.ApiError as e:
#                 if 'rate limit' in str(e).lower():
#                     print(f"Rate limit exceeded. Retrying in {delay} seconds.")
#                     time.sleep(delay)
#                 else:
#                     raise e
#         raise Exception(f"Failed to get route after {retries} retries.")

#     # Create the map visualization
#     fig = go.Figure()

#     # Add the route lines
#     for i in range(len(df) - 1):
#         start = (df.iloc[i]['Longitude'], df.iloc[i]['Latitude'])
#         end = (df.iloc[i + 1]['Longitude'], df.iloc[i + 1]['Latitude'])
        
#         if pd.notnull(start[0]) and pd.notnull(start[1]) and pd.notnull(end[0]) and pd.notnull(end[1]):
#             route = get_route(client, start, end)
#             geojson = route['features'][0]['geometry']
#             coordinates = geojson['coordinates']
#             lons, lats = zip(*coordinates)
            
#             color = zone_colors[df.iloc[i]['ZoneID']] if df.iloc[i]['ZoneID'] == df.iloc[i+1]['ZoneID'] else 'black'
            
#             fig.add_trace(go.Scattermapbox(
#                 mode="lines",
#                 lon=lons,
#                 lat=lats,
#                 line=dict(width=3, color=color),
#                 hoverinfo='none',
#                 showlegend=False
#             ))

#     # Add the stops with sequence numbers
#     fig.add_trace(go.Scattermapbox(
#         lat=df['Latitude'],
#         lon=df['Longitude'],
#         mode='markers+text',
#         marker=go.scattermapbox.Marker(
#             size=10,
#             color=[zone_colors[z] for z in df['ZoneID']],
#             showscale=False
#         ),
#         text=df['Sequence'].astype(str),
#         textposition='top center',
#         textfont=dict(size=14, color='black'),
#         hoverinfo='text'
#     ))


#     # Add a legend for ZoneID colors
#     for zone, color in zone_colors.items():
#         fig.add_trace(go.Scattermapbox(
#             lat=[None], lon=[None],
#             mode='markers',
#             marker=dict(size=10, color=color),
#             legendgroup=zone,
#             showlegend=True,
#             name=f'Zone {zone}'
#         ))

#     # Add a legend for black lines
#     fig.add_trace(go.Scattermapbox(
#         lat=[None], lon=[None],
#         mode='lines',
#         line=dict(width=3, color='black'),
#         legendgroup='Transition',
#         showlegend=True,
#         name='Transition between zones'
#     ))

#     # Update the layout of the map
#     fig.update_layout(
#         mapbox=dict(
#             style="open-street-map",
#             zoom=12,
#             center=dict(lon=df['Longitude'].mean(), lat=df['Latitude'].mean())
#         ),
#         height=1000,
#         width=1200,
#         margin={"r":0,"t":0,"l":0,"b":0},
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         ),
#         title=f"Route Visualization for {route_id}"
#     )

#     # Show the plot
#     fig.show()

#     # Save the figure to an HTML file
#     filename = f"Route_{route_id}.html"
#     fig.write_html(filename)
#     print(f"Map saved to {filename}")


# visualize_route(df, '5b3ce3597851110001cf62484580fb04ffc34ec191e892a13e92a8bd')

import time
import random
import openrouteservice
from openrouteservice.exceptions import ApiError, _OverQueryLimit
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygad

# Конфигурация
API_KEY = "5b3ce3597851110001cf62484580fb04ffc34ec191e892a13e92a8bd"  # Замените на ваш реальный ключ
MAX_RETRIES = 5
DELAY_BASE = 1.5  # Базовая задержка в секундах между запросами
CACHE_FILE = "distance_matrix_cache1.npy"  # Файл для кеширования матрицы расстояний

def read_points_from_file(filename):
    """Чтение точек из файла с обработкой ошибок"""
    points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    lon, lat = map(float, line.strip().split(','))
                    points.append((lon, lat))
                except ValueError:
                    print(f"Ошибка формата в строке: {line.strip()}")
        return points
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return []

def calculate_distance_matrix(points, api_key, use_cache=True):
    """Расчет матрицы расстояний с кешированием и обработкой ошибок"""
    if use_cache:
        try:
            return np.load(CACHE_FILE)
        except:
            pass
    
    n = len(points)
    distance_matrix = np.zeros((n, n))
    client = openrouteservice.Client(key=api_key)
    
    for i in range(n):
        for j in range(i+1, n):  # Оптимизация: матрица симметричная
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    # Добавляем случайную задержку для избежания лимита
                    time.sleep(DELAY_BASE * (random.random() + 0.5))
                    
                    # Запрос маршрута
                    route = client.directions(
                        coordinates=[points[i], points[j]],
                        profile='driving-car',
                        format='json'
                    )
                    
                    # Извлечение расстояния
                    if 'routes' in route and len(route['routes']) > 0:
                        distance = route['routes'][0]['summary']['distance'] / 1000  # км
                        distance_matrix[i][j] = distance
                        distance_matrix[j][i] = distance  # Симметричная матрица
                        break
                    else:
                        print(f"Некорректный ответ API для точек {i} и {j}")
                        distance_matrix[i][j] = float('inf')
                        distance_matrix[j][i] = float('inf')
                        break
                        
                except (_OverQueryLimit, ApiError) as e:
                    retry_count += 1
                    wait_time = DELAY_BASE * (2 ** retry_count)
                    print(f"Ошибка API (попытка {retry_count}): {e}. Ждем {wait_time} сек.")
                    time.sleep(wait_time)
                except Exception as e:
                    print(f"Неизвестная ошибка для точек {i} и {j}: {e}")
                    distance_matrix[i][j] = float('inf')
                    distance_matrix[j][i] = float('inf')
                    break
            else:
                print(f"Не удалось получить расстояние для точек {i} и {j} после {MAX_RETRIES} попыток")
                distance_matrix[i][j] = float('inf')
                distance_matrix[j][i] = float('inf')
    
    # Сохраняем в кеш
    np.save(CACHE_FILE, distance_matrix)
    return distance_matrix

def genetic_algorithm_tsp(distance_matrix, num_generations=50, population_size=100):
    """Генетический алгоритм для задачи коммивояжера с исправлением ошибки индексации"""
    def fitness_func(ga_instance, solution, solution_idx):
        """Функция приспособленности с преобразованием индексов в целые числа"""
        # Преобразуем значения в целые числа для индексации
        int_solution = solution.astype(int)
        total_distance = 0
        
        # Рассчитываем расстояние для всего маршрута
        for i in range(len(int_solution)-1):
            total_distance += distance_matrix[int_solution[i]][int_solution[i+1]]
        
        # Добавляем возврат в начальную точку
        total_distance += distance_matrix[int_solution[-1]][int_solution[0]]
        
        # Инвертируем расстояние для максимизации приспособленности
        return 1.0 / (total_distance + 1e-10)  # Добавляем малое значение для избежания деления на 0
    
    # Настройки генетического алгоритма
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=population_size//2,
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=len(distance_matrix),
        gene_space=range(len(distance_matrix)),
        mutation_type="swap",
        mutation_probability=0.2,
        crossover_type="scattered",
        suppress_warnings=True,
        allow_duplicate_genes=False,  # Важно для задачи коммивояжера
        gene_type=int,  # Указываем, что гены должны быть целыми числами
        init_range_low=0,
        init_range_high=len(distance_matrix)-1)
    
    # Запуск алгоритма
    ga_instance.run()
    
    # Получение лучшего решения
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    # Преобразуем решение в целые числа
    return solution.astype(int)

def visualize_route(points, route, api_key):
    """Визуализация маршрута с обработкой ошибок"""
    client = openrouteservice.Client(key=api_key)
    fig = go.Figure()
    
    # Добавляем маркеры точек
    fig.add_trace(go.Scattermapbox(
        lat=[p[1] for p in points],
        lon=[p[0] for p in points],
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=[str(i) for i in range(len(points))],
        textposition='top center',
        name='Пункты выдачи'
    ))
    
    # Получаем координаты маршрута в правильном порядке
    route_points = [points[i] for i in route] + [points[route[0]]]
    
    try:
        # Получаем геометрию маршрута
        directions = client.directions(
            coordinates=route_points,
            profile='driving-car',
            format='geojson',
            optimize_waypoints=True
        )
        
        # Добавляем маршрут на карту
        if 'features' in directions and len(directions['features']) > 0:
            geometry = directions['features'][0]['geometry']
            lons, lats = zip(*geometry['coordinates'])
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                line=dict(width=4, color='blue'),
                name='Оптимальный маршрут'
            ))
    except Exception as e:
        print(f"Ошибка при построении маршрута: {e}")
        # Резервный вариант - соединяем точки прямыми линиями
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=[p[0] for p in route_points],
            lat=[p[1] for p in route_points],
            line=dict(width=2, color='green', dash='dot'),
            name='Приблизительный маршрут'
        ))
    
    # Настройки карты
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lon=sum(p[0] for p in points)/len(points), 
            lat=sum(p[1] for p in points)/len(points)),
            zoom=11
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=800,
        title="Оптимальный маршрут между ПВЗ Ozon в Казани"
    )
    
    fig.show()
    fig.write_html("optimal_route_kazan.html")

if __name__ == "__main__":
    # 1. Загрузка точек
    points = read_points_from_file("ozon_pvz_points.txt")
    if not points:
        print("Не удалось загрузить точки. Проверьте файл.")
        exit()
    
    # 2. Расчет матрицы расстояний
    print("Расчет матрицы расстояний...")
    try:
        distance_matrix = calculate_distance_matrix(points, API_KEY)
        print("Матрица расстояний успешно рассчитана")
    except Exception as e:
        print(f"Ошибка при расчете матрицы расстояний: {e}")
        exit()
    
    # 3. Поиск оптимального маршрута
    print("Поиск оптимального маршрута...")
    try:
        optimal_route = genetic_algorithm_tsp(distance_matrix)
        print("Найденный маршрут:", optimal_route)
    except Exception as e:
        print(f"Ошибка в генетическом алгоритме: {e}")
        exit()
    
    # 4. Визуализация
    print("Визуализация результатов...")
    visualize_route(points, optimal_route, API_KEY)