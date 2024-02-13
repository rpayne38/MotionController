import carla
import random
import time
import sys
import numpy as np
import math
import StanleyController
from Common import Waypoint

sys.path.append("/home/rpayne/MotionController/CARLA_0.9.15/PythonAPI/carla")

from agents.navigation.global_route_planner import GlobalRoutePlanner

dt = 0.1
nTicks = 1e6
planResolution = 10
targetSpeed = 10

# CARLA uses the left-hand rule with Z as the up vector and y pointing North
# N* uses the right-hand rule with Z as the up vector and y pointing North

def ConvertAngle(angle):
    # some angles are greater than 360 for some reason
    while angle > 360:
        angle -= 360

    if angle > 180:
        angle -= 180

    return angle

def DrawTransform(world, transform, colour, lifetime):
    rads = math.radians(transform.rotation.yaw)
    start = transform.location
    end = start + carla.Location(math.cos(rads), math.sin(rads), 0.0)
    world.debug.draw_arrow(start, end, 0.2, 0.2, color=colour, life_time=lifetime)

def DisplayClosestPtToSpectator(world, pts):
    while True:
        spectator = world.get_spectator()
        spectatorPos = spectator.get_transform().location

        idx = 0
        minDist = 1e9
        for i, pt in enumerate(pts):
            dist = pt.location.distance(spectatorPos)
            if dist < minDist:
                idx = i
                minDist = dist

        DrawTransform(world, pts[idx], carla.Color(r=0, g=255, b=0), 1)
        print(idx)


def GeneratePlan(routePlanner, start, end):
    waypoints = routePlanner.trace_route(start, end)
    for i, waypoint in enumerate(waypoints):
        waypoints[i] = waypoint[0].transform
    
    return waypoints
      
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

    blueprint_library = world.get_blueprint_library()

    # delete all vehicles
    world_snapshot = world.wait_for_tick()
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    for vehicle in vehicles:
        vehicle.destroy()

    # Add a random vehicle
    vehicle_blueprint = blueprint_library.find("vehicle.nissan.micra")

    # Give the vehicle a random colour
    if vehicle_blueprint.has_attribute("color"):
        color = random.choice(vehicle_blueprint.get_attribute("color").recommended_values)
        vehicle_blueprint.set_attribute("color", color)

    # Setup path planner
    grp = GlobalRoutePlanner(map, planResolution)
    spawn_points = world.get_map().get_spawn_points()

    a = carla.Location(spawn_points[342].location)
    b = carla.Location(spawn_points[231].location)
    c = carla.Location(spawn_points[253].location)
    d = carla.Location(spawn_points[157].location)

    waypoints = GeneratePlan(grp, a, b)
    waypoints.extend(GeneratePlan(grp, b, c))
    #waypoints.extend(GeneratePlan(grp, c, d))
    
    # Draw plan
    for waypoint in waypoints:
        DrawTransform(world, waypoint, carla.Color(r=0, g=255, b=0), 100)

    # Get vehicle spawn location
    spawn_point = waypoints[1]
    spawn_point.location.y -= 3.0
    spawn_point.location.z += 2.0

    # Spawn in the vehicle
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    
    time.sleep(1.0)

    # Convert from left-hand to right-hand rule
    plan = []
    for waypoint in waypoints:
        pt = Waypoint(-waypoint.location.x, waypoint.location.y, targetSpeed, StanleyController.NormaliseAngle(math.radians(180.0 - waypoint.rotation.yaw)))
        plan.append(pt)

    # Setup Controller
    controller = StanleyController.StanleyController(plan, 1.0, 0.6, targetSpeed, np.radians(70.0), 2.5)

    tick = 0
    while tick < nTicks:
        world_snapshot = world.wait_for_tick()
        vehicle_snapshot = world_snapshot.find(vehicle.id)

        transform = vehicle_snapshot.get_transform()
        position = transform.location
        yaw = transform.rotation.yaw
        speed = vehicle_snapshot.get_velocity().length()

        # Update controller. Making sure to convert between left and right-hand rule
        steerCmd, throttleCmd, targetIdx = controller.Update(-position.x, position.y, StanleyController.NormaliseAngle(np.radians(180.0 - yaw)), speed)
        DrawTransform(world, waypoints[targetIdx], carla.Color(r=255, g=0, b=0), dt)

        # Normalise commands and convert to left-hand rule
        steerCmd /= np.pi
        steerCmd *= -1
        print(f"steerCmd: {steerCmd:.2f} throttleCmd: {throttleCmd:.2f} currentSpeed: {speed:.2f} m/s", end="\r")
        
        # negative is left, positive is right
        vehicle.apply_control(carla.VehicleControl(throttle=throttleCmd, steer=steerCmd))

        tick += 1
        time.sleep(dt)


    # Destroy the vehicle
    time.sleep(3.0)
    vehicle.destroy()

main()