from dataclasses import dataclass
import numpy as np
import math

@dataclass
class Waypoint:
    x : float
    y : float
    targetSpeed : float
    dir : float # about z, positive is North

    def pos(self):
        return np.array([self.x, self.y])
    
    def numpyArray(self):
        return np.array([self.x, self.y, self.targetSpeed, self.dir])
    
@dataclass
class State:
    x : float       # metres
    y : float       # metres
    speed : float   # m/s
    yaw : float     # rads

    def numpyArray(self):
        return np.array([self.x, self.y, self.speed, self.yaw])
    
def NormaliseAngle(angle):
    # normalise angle between -pi and pi
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle