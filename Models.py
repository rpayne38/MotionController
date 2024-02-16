import numpy as np
import math
import copy

from Common import State, NormaliseAngle

class KinematicBicycle():
    def __init__(self, wheelBase, maxSpeed , maxSteeringAngle):
        self.wheelBase = wheelBase
        self.maxSpeed = maxSpeed
        self.maxSteeringAngle = maxSteeringAngle

    def update(self, state : State, accel, steeringAngle, dt):
        newState = copy.deepcopy(state)

        steeringAngle = np.clip(steeringAngle, -self.maxSteeringAngle, self.maxSteeringAngle)
        
        newState.x += newState.speed * math.cos(newState.yaw) * dt
        newState.y += newState.speed * math.sin(newState.yaw) * dt
        newState.speed += accel * dt
        newState.yaw += (newState.speed * math.tan(steeringAngle) / self.wheelBase) * dt

        newState.yaw = NormaliseAngle(newState.yaw)

        #TODO(RP) Clip speed?
        
        return newState
