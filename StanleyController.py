import numpy as np
import matplotlib.pyplot as plt
from Common import Waypoint

ksoft = 1.0 # low speed gain
    
def NormaliseAngle(angle):
    # normalise angle between -pi and pi
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def TargetAheadOfVehicle(vehiclePos, vehicleYaw, targetPos):
    # https://stackoverflow.com/questions/65794490/unity3d-check-if-a-point-is-to-the-left-or-right-of-a-vector
    error = targetPos - vehiclePos
    normalisedError = error / np.hypot(error[0], error[1])

    forwardVec = np.array([np.cos(vehicleYaw), np.sin(vehicleYaw)])
    forwardVec /= np.hypot(forwardVec[0], forwardVec[1])
    rightVec = np.array([forwardVec[1], -forwardVec[0]])
    crossProduct = np.cross(normalisedError, rightVec)

    # If True target is on top of or behind vehicle
    return crossProduct >= 0.0

class StanleyController():
    def __init__(self, plan : list[Waypoint], speedGain, steerGain, targetSpeed, maxSteer, length):
        self.targetSpeed = targetSpeed
        self.maxSteer = maxSteer
        self.length = length

        self.targetIdx = 0

        self.plan = plan

        self.kp = speedGain
        self.k = steerGain

    def Update(self, x, y, yaw, velocity):

        # Calculate speed control
        throttleCmd = self.__ProportionalController(self.targetSpeed, velocity)

        # Calculate front axle position
        frontAxleX = x + self.length * 0.5 * np.cos(yaw)
        frontAxleY = y + self.length * 0.5 * np.sin(yaw)

        # Update target point on path and calculate cross track error
        self.targetIdx = self.__UpdateTargetIdx([frontAxleX, frontAxleY], yaw)
        crossTrackErr = self.__CalculateCrossTrackError([frontAxleX, frontAxleY], yaw)

        # Calcaulte the angle to steer towards, in the local frame
        steeringAngle = self.__CalcualteSteeringAngle(self.plan[self.targetIdx].dir, crossTrackErr, yaw, velocity)

        return steeringAngle, throttleCmd ,self.targetIdx

    def __CalcualteSteeringAngle(self, pathHeading, crossTrackError, yaw, velocity):
        # Apply Stanley Control
        headingCorrection = NormaliseAngle(pathHeading - yaw)
        crossTrackErrorCorrection = np.arctan2(self.k * crossTrackError, ksoft + velocity)
        steeringAngle = headingCorrection + crossTrackErrorCorrection
        steeringAngle = np.clip(steeringAngle, -self.maxSteer, self.maxSteer)
        return steeringAngle
    
    def __UpdateTargetIdx(self, frontAxlePosition, yaw):
        # Find the index of the point closest to the front axle
        errorXY = np.zeros((len(self.plan), 2))
        for i, coords in enumerate(self.plan):
            errorXY[i] = coords.pos() - frontAxlePosition
        
        distances = np.hypot(errorXY[:,0], errorXY[:,1])
        targetIdx = np.argmin(distances)
        
        # Ensure tagetIdx is ahead of us
        while TargetAheadOfVehicle(frontAxlePosition, yaw, self.plan[targetIdx].pos()):
            targetIdx += 1

        return targetIdx

    def __CalculateCrossTrackError(self, frontAxlePosition, yaw):
        # Cross Track Error is the distance in y to the target, in the vehicle's frame of reference
        globalErr = self.plan[self.targetIdx].pos() - frontAxlePosition
        localErr = np.array([globalErr[0] * np.cos(-yaw) - globalErr[1] * np.sin(-yaw),
                             globalErr[0] * np.sin(-yaw) + globalErr[1] * np.cos(-yaw)])
        return localErr[1]

    def __ProportionalController(self, target, current):
        return np.clip(self.kp * (target - current), -1.0, 1.0)