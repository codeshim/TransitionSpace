# Required Libraries
import numpy  as np
from shapely import affinity
from extractFunction import *

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, polygonList, target_function):

    dim = len(min_values)
    population     = np.random.uniform(min_values, max_values, (size, dim)) 
    fitness_values_int = target_function(polygonList, population)
    fitness_values = np.array([fitness_values_int])
    fitness_values = fitness_values[:, np.newaxis]
    
    population     = np.hstack((population, fitness_values))

    return population # variable 갯수 + final 결과값

############################################################################

# Function: Epson Vector -> 이게 뭔가 평등하게 뽑아져오지 않는데? 
def epson_vector(min_values, mu, sigma):
    #epson = np.random.normal(mu, sigma, len(min_values))
    epson = np.random.uniform(min, max, len(min_values))

    return epson.reshape(1, -1)


# Function: Update Solution
def update_solution(guess, epson, min_values, max_values, polygonList, target_function):
    
    updated_solution = guess[0,:-1] + epson
    #print(updated_solution)

    #updated_solution = np.array([guess[0,:-1]]) 
    updated_solution = np.clip(updated_solution, min_values, max_values)
    

    fitness_values_int = target_function(polygonList, updated_solution)
    fitness_values = np.array([fitness_values_int])
    fitness_values = fitness_values[:, np.newaxis]
    updated_solution = np.hstack((updated_solution, fitness_values))


    return updated_solution # variable 갯수 + final 결과값

############################################################################

# Polygon SA
def simulated_annealing(min_values = [-100,-100], max_values = [100,100], min = 0, max = 1, initial_temperature = 1.0, temperature_iterations = 1000, final_temperature = 0.0001, alpha = 0.9, polygonList = [], target_function = target_function, verbose = True, target_value = None):    
    
    
    # image recorders
    trackAll = False
    
    
    # initialize to the number of polygon
    num_polygon = len(polygonList)
    x_track = [0] * num_polygon
    y_track = [0] * num_polygon
    theta_track = [0] * num_polygon

    x_final = [0] * num_polygon
    y_final = [0] * num_polygon
    theta_final = [0] * num_polygon
    
    # 얘랑, 얘가 같아버린다 라고 말하는거같은데
    result_polygons = polygonList.copy()

    guess       = initial_variables(1, min_values, max_values, polygonList, target_function) 
    best        = np.copy(guess)
    fx_best     = guess[0,-1] # guess[0][-1]랑 똑같음
    
    for i in range(num_polygon):
        x_track[i] = 0.1 * best[0][(i*3)+0]
        y_track[i] = 0.1 * best[0][(i*3)+1]
        theta_track[i] = 1 * best[0][(i*3)+2]

    num_of_update = 0
    global_update = 0

    temperature = float(initial_temperature)
    while (temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            if (verbose == True):
                print('Temperature = ', round(temperature, 4), ' ; iteration = ', repeat, ' ; f(x) = ', best[0, -1])
            fx_old    = guess[0,-1]    
            epson     = epson_vector(min_values, min, max) 
            

            # update polygon
            for i in range(num_polygon):
                polygonList[i] = affinity.translate(polygonList[i], xoff=x_track[i], yoff=y_track[i])
                polygonList[i] = affinity.rotate(polygonList[i], theta_track[i])

            
            
            
            new_guess = update_solution(guess, epson, min_values, max_values, polygonList, target_function)
            #print(new_guess)

            for i in range(num_polygon):
                x_track[i] = 0.1 * new_guess[0][(i*3)+0]
                y_track[i] = 0.1 * new_guess[0][(i*3)+1]
                theta_track[i] = 1 * new_guess[0][(i*3)+2]
            
            
            fx_new    = new_guess[0,-1] 
            delta     = (fx_new - fx_old) 
            r         = np.random.rand()
            p         = np.exp(-delta/temperature)
            
            if (delta < 0 or r <= p):
                guess = np.copy(new_guess)
            

            if (fx_new < fx_best):
                fx_best = fx_new
                best    = np.copy(guess) 
                
                # save to polygon
                for i in range(num_polygon):
                    x_final[i] = x_track[i]
                    y_final[i] = y_track[i]
                    theta_final[i] = theta_track[i]

                    # save to image
                    tempPolygonToDraw = polygonList[i]
                    tempPolygonToDraw = affinity.translate(tempPolygonToDraw, xoff=x_final[i], yoff=y_final[i])
                    tempPolygonToDraw = affinity.rotate(tempPolygonToDraw, theta_final[i])

                # save to return
                result_polygons = polygonList.copy()
                

                draw_to_imgs(polygonList, num_of_update, 'process')
                num_of_update = num_of_update + 1
                

            if trackAll:
                for i in range(num_polygon):
                        
                    x_final[i] = x_track[i]
                    y_final[i] = y_track[i]
                    theta_final[i] = theta_track[i]

                    # save to image
                    tempPolygonToDraw = polygonList[i]
                    tempPolygonToDraw = affinity.translate(tempPolygonToDraw, xoff=x_final[i], yoff=y_final[i])
                    tempPolygonToDraw = affinity.rotate(tempPolygonToDraw, theta_final[i])

                
                draw_to_imgs(polygonList, global_update, 'process')
                global_update = global_update + 1
            
        
        temperature = alpha * temperature   

        if (target_value is not None):
            if (fx_best <= target_value):
                temperature = final_temperature 
                break
        

    # 여기서 뱉어내는거 자체가 polygon이어야 할거같은데 
    return best[0, -1], result_polygons 

############################################################################