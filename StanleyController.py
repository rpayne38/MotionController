import numpy as np
import matplotlib.pyplot as plt
from PythonRobotics.PathPlanning.CubicSpline import cubic_spline_planner

kp = 1.0
k = 0.6    # steering gain
ksoft = 1.0 # low speed gain
max_steer = np.radians(80.0) # max steering angle
dt = 0.05 # timestep
length = 2.5 # distance between front and rear axle
targetSpeed = 10.0 # m/s

class StanleyController():
    def __init__(self, plan, x = 0.0, y = 0.0, yaw = 0.0, velocity = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.targetIdx = 0
        self.frontAxle = np.array([0.0, 0.0])
        self.crossTrackErr = np.zeros(plan.size)
        self.plan = plan

    def Update(self):
        accel = self.ProportionalController(targetSpeed, self.velocity)

        # Calculate front axle position
        self.frontAxle[0] = self.x + length * np.cos(self.yaw)
        self.frontAxle[1] = self.y + length * np.sin(self.yaw)

        crossTrackErr, valid = self.CalculateCrossTrackError(self.frontAxle)

        if (not valid):
            return True
        
        self.crossTrackErr = np.append(self.crossTrackErr, crossTrackErr)

        targetHeading = self.plan[self.targetIdx + 1] - self.plan[self.targetIdx]
        targetHeading = np.arctan2(targetHeading[1], targetHeading[0])
        steeringAngle = self.CalcualteSteeringAngle(targetHeading, crossTrackErr)

        # Uses bicycle model of vehicle
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += self.velocity / length * np.tan(steeringAngle) * dt
        self.yaw = self.NormaliseAngle(self.yaw)
        self.velocity += accel * dt
        print(self.velocity)

        return False

    def CalcualteSteeringAngle(self, pathHeading, crossTrackError):
        # Apply Stanley Control
        headingCorrection = self.NormaliseAngle(pathHeading - self.yaw)
        crossTrackErrorCorrection = np.arctan2(k * crossTrackError, ksoft * self.velocity)
        steeringAngle = headingCorrection + crossTrackErrorCorrection
        steeringAngle = np.clip(steeringAngle, -max_steer, max_steer)
        return steeringAngle

    def CalculateCrossTrackError(self, frontAxlePosition):
        # Find the index of the point closest to the front axle
        errorXY = np.zeros(self.plan.shape)
        for i, coords in enumerate(self.plan):
            errorXY[i] = coords - frontAxlePosition
        
        distances = np.hypot(errorXY[:,0], errorXY[:,1])
        targetIdx = np.argmin(distances)
        if self.targetIdx < targetIdx:
            self.targetIdx = targetIdx
        crossTrackErr = distances[self.targetIdx]

        if self.targetIdx == self.plan.shape[0] - 1:
            return crossTrackErr, False

        pathYaw = self.plan[self.targetIdx + 1] - self.plan[self.targetIdx]
        pathYaw = np.arctan2(pathYaw[1], pathYaw[0]) 

        # The angle we are approaching the path at determines if the cross track error should be positive or negative
        angleToTarget = np.arctan2(frontAxlePosition[1] - self.plan[self.targetIdx][1], frontAxlePosition[0] - self.plan[self.targetIdx][0])
        yawDiff = self.NormaliseAngle(pathYaw - angleToTarget)
        if yawDiff < 0:
            crossTrackErr *= -1

        return crossTrackErr, True

    def ProportionalController(self, target, current):
        return kp * (target - current)
    
    def NormaliseAngle(self, angle):
        # normalise angle between -pi and pi
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
    
    def Plot(self):
        plt.figure(1)
        plt.cla()
        plt.plot(self.plan[:,0], self.plan[:,1])
        plt.plot(self.x, self.y, color='red', marker='o')
        plt.plot(self.frontAxle[0], self.frontAxle[1], color='red', marker='x')
        plt.grid(True)
        plt.pause(dt)

if  __name__ == '__main__':
    
    # Define plan to follow
    #  target course
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    cx, cy, _, _, _ = cubic_spline_planner.calc_spline_course(
    ax, ay, ds=0.1)
    plan = np.array([cx, cy]).transpose()

    finished = False
    controller = StanleyController(plan)
    while not finished:
        finished = controller.Update()
        controller.Plot()
    
    plt.subplots(1)
    plt.plot(controller.crossTrackErr)
    plt.grid(True)

    plt.show()