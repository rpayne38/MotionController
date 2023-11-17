import carla
import random
import time
import sys

sys.path.append("/home/rpayne/MotionController/CARLA_0.9.15/PythonAPI/carla")

from agents.navigation.global_route_planner import GlobalRoutePlanner

nTicks = 5

def GetWayPoints(world, waypoints, road_id=None, life_time=50.0):
    filtered_waypoints = []
    for waypoint in waypoints:
        if(waypoint.road_id == road_id and waypoint.lane_id < 0):
            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                        persistent_lines=True)
            filtered_waypoints.append(waypoint)
        
    return filtered_waypoints
      
def main():                                   
    # Connect to server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Change world to Town04
    world = client.get_world()

    map = world.get_map()
    if "Town04" not in map.name:
        world = client.load_world("Town04")
        map = world.get_map()

    # Add a random vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.find("vehicle.nissan.micra")

    # Give the vehicle a random colour
    if vehicle_blueprint.has_attribute("color"):
        color = random.choice(vehicle_blueprint.get_attribute("color").recommended_values)
        vehicle_blueprint.set_attribute("color", color)

    waypoints = world.get_map().generate_waypoints(distance=1.0)
    waypoints = GetWayPoints(world, waypoints, road_id=3, life_time=20)

    spawn_point = waypoints[0].transform
    spawn_point.location.z += 2.0

    # Spawn in the vehicle
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)

    target_waypoint = waypoints[-1]
    world.debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), life_time=20,
                            persistent_lines=True)
    
    time.sleep(1.0)
    
    for i in range(nTicks):
        # negative is left, positive is right
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        world_snapshot = world.wait_for_tick()
        vehicle_snapshot = world_snapshot.find(vehicle.id)
        if vehicle_snapshot:
            print(vehicle_snapshot.get_velocity())
        else:
            print("No vehicle")

    # Destroy the vehicle
    time.sleep(3.0)
    vehicle.destroy()

main()