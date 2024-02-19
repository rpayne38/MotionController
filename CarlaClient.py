import carla
import random
import time
import sys
import numpy as np
import math

import NMPC
from Common import Waypoint, State, NormaliseAngle
from Models import KinematicBicycle

sys.path.append("/home/rpayne/MotionController/CARLA_0.9.15/PythonAPI/carla")

from agents.navigation.global_route_planner import GlobalRoutePlanner

DT = 0.05          # secs
N_TICKS = 1e4
PLAN_RES = 2.0     # metres
TARGET_SPEED = 4.0    # m/s

MAX_SPEED = 12.0
MAX_STEERING_ANGLE = np.radians(70.0)
WHEEL_BASE = 2.5

# CARLA uses the left-hand rule with Z as the up vector and y pointing North
# N* uses the right-hand rule with Z as the up vector and y pointing North

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

def ProportionalController(gain, error):
        return np.clip(gain * error, 0.0, 1.0)
      
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
    grp = GlobalRoutePlanner(map, PLAN_RES)
    spawn_points = world.get_map().get_spawn_points()

    townA = carla.Location(spawn_points[342].location)
    townB = carla.Location(spawn_points[231].location)
    townC = carla.Location(spawn_points[253].location)
    townD = carla.Location(spawn_points[157].location)

    motorwayA = carla.Location(spawn_points[50].location)
    motorwayB = carla.Location(spawn_points[100].location)

    waypoints = GeneratePlan(grp, townA, townB)
    waypoints.extend(GeneratePlan(grp, townB, townC))
    #waypoints.extend(GeneratePlan(grp, c, d))
    
    # Draw plan
    for waypoint in waypoints:
        DrawTransform(world, waypoint, carla.Color(r=0, g=255, b=0), 1000 / TARGET_SPEED)

    # Get vehicle spawn location
    spawn_point = waypoints[1]
    spawn_point.location.z += 2.0

    # Spawn in the vehicle
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    
    time.sleep(1.0)

    # Convert from left-hand to right-hand rule
    plan = []
    for waypoint in waypoints:
        pt = Waypoint(-waypoint.location.x, waypoint.location.y, TARGET_SPEED, NormaliseAngle(math.radians(180.0 - waypoint.rotation.yaw)))
        plan.append(pt)

    # Setup Controller
    model = KinematicBicycle(WHEEL_BASE, MAX_SPEED, MAX_STEERING_ANGLE)
    controller = NMPC.ModelPredictiveController(plan, model)

    tick = 0
    prevSpeed = 0.0
    while tick < N_TICKS:
        t0 = time.time()
        world_snapshot = world.wait_for_tick()
        vehicle_snapshot = world_snapshot.find(vehicle.id)

        transform = vehicle_snapshot.get_transform()
        position = transform.location
        yaw = transform.rotation.yaw
        speed = vehicle_snapshot.get_velocity().length()

        # Update controller. Making sure to convert between left and right-hand rule
        state = State(-position.x, position.y, speed, NormaliseAngle(np.radians(180.0 - yaw)))
        steerAngle, accel, targetIdx = controller.tick(state)

        # Draw target waypoint
        for i in range(NMPC.PRED_HORIZON):
            DrawTransform(world, waypoints[targetIdx + i], carla.Color(r=255, g=0, b=0), DT)

        # Normalise commands and convert to left-hand rule
        steerCmd = steerAngle * -1
        steerCmd /= MAX_STEERING_ANGLE

        throttleCmd = ProportionalController(0.4, accel - ((speed - prevSpeed) / DT))
        prevSpeed = speed

        # negative is left, positive is right
        vehicle.apply_control(carla.VehicleControl(steer = steerCmd, throttle = throttleCmd))

        tick += 1

        dur = time.time() - t0
        if dur < DT:
            time.sleep(DT - dur)

        print(f"Steer: {steerAngle:.2f}\t Throttle: {throttleCmd:.2f}\t currentSpeed: {speed:.2f} m/s\t Rate: {1 / (time.time() - t0):.1f} Hz", end="\r")


    # Destroy the vehicle
    time.sleep(3.0)
    vehicle.destroy()

main()