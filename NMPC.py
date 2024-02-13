import matplotlib.pyplot as plt
import numpy as np
import math
import acado
import sys

sys.path.append("./CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise

from Common import Waypoint, State
from Models import KinematicBicycle

WHEEL_BASE = 1.32
MAX_SPEED = 20.0
MAX_STEERING_ANGLE = np.radians(50.0)
TARGET_SPEED = 5.0

TIMESTEP = 0.05
MAX_TIMESTEPS = 450
PRED_HORIZON = 15

NX = 4  # [x, y, v, yaw]
NY = 4  # reference state variables
NYN = 4  # reference terminal state variables
NU = 2  # [accel, delta]

# mpc parameters
Q = np.diag([1.0, 1.0, 0.1, 0.75])  # state cost matrix
Qf = Q  # state final matrix

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def waypointsToNumpy(waypoints : list[Waypoint]):
    # Output format is [x0, y0, v0, yaw0], [x1, y1, v1, yaw1], ...
    ret = np.zeros((len(waypoints), NY))
    for i, waypoint in enumerate(waypoints):
        ret[i][0] = waypoint.x
        ret[i][1] = waypoint.y
        ret[i][2] = waypoint.targetSpeed
        ret[i][3] = waypoint.dir
    
    return ret


class ModelPredictiveController():
    def __init__(self, waypoints : list[Waypoint], model : KinematicBicycle, state : State):
        self.waypoints = waypoints
        self.model = model
        self.state = state

    def _predict(self, accel: list[float], steeringAngle: list[float]):
        ret = [self.model.update(self.state, accel[0], steeringAngle[0], TIMESTEP)]
        for i in range(PRED_HORIZON):
            ret.append(self.model.update(ret[i], accel[i], steeringAngle[i], TIMESTEP))

        return ret
    
    def _solve(self, xPred: list[State]):
        # see acado.c for parameter details
        # xPred = predicted state at each timestep

        _x0 = np.zeros((1,NX))                  # initial state
        X = np.zeros((PRED_HORIZON + 1, NX))    # predicted state
        U = np.zeros((PRED_HORIZON, NU))        # optimal control input 
        Y = np.zeros((PRED_HORIZON, NY))        # reference state
        yN = np.zeros((1,NYN))                  # final reference state

        nearestWaypointIdx, dist = self.nearestWaypoint()

        for t in range(PRED_HORIZON):
            Y[t,:] = self.waypoints[nearestWaypointIdx + t].numpyArray()  # reference state
            
            X[t][0] = xPred[t].x
            X[t][1] = xPred[t].y
            X[t][2] = xPred[t].speed
            X[t][3] = xPred[t].yaw

        X[-1:] = X[-2,:]
        _x0[0,:] = self.state.numpyArray()
        yN[0,:] = Y[-1,:NYN]         # reference terminal state

        # acado.mpc(forceInit, doFeedback, x0, X, U, Y, yN, W, WN, verbose)
        #   forceInit: force solver initialization? (0/1)
        #   doFeedback: do feedback step? (0/1)
        #   x0: initial state feedback
        #   X: differential variable vectors.
        #   U: control variable vectors.
        #   Y: reference/measurement vectors of first N nodes.
        #   yN: Reference/measurement vector for N+1 node.
        #   W: weight matrix
        #   WN: weight matrix
        #   verbose: verbosity level (0-2)
        X, U = acado.mpc(0, 1, _x0, X,U,Y,yN, np.transpose(np.tile(Q,PRED_HORIZON)), Qf, 0)
            
        ox = get_nparray_from_matrix(X[:,0])
        oy = get_nparray_from_matrix(X[:,1])
        ov = get_nparray_from_matrix(X[:,2])
        oyaw = get_nparray_from_matrix(X[:,3])
        oa = get_nparray_from_matrix(U[:,0])
        odelta = get_nparray_from_matrix(U[:,1])

        return oa, odelta, State(ox, oy, ov, oyaw), nearestWaypointIdx
    
    def nearestWaypoint(self):
        minDist = float("inf")
        idx = len(self.waypoints) + 1
        for i, waypoint in enumerate(self.waypoints):
            dist = math.dist([self.state.x, self.state.y], waypoint.pos())
            if dist < minDist:
                minDist = dist
                idx = i

        return idx, minDist

    
    def tick(self, accel, steeringAngle):
        # Predict next states based on current steering angle and acceleration
        predictedStates = self._predict(accel, steeringAngle)
                
        # solve cost func to get new state
        accel, steeringAngle, newState, targetWaypoint = self._solve(predictedStates)

        steeringCmd = steeringAngle[0]
        throttleCmd = accel[0]

        # Update platform state with MPC Estimates
        self.state = self.model.update(self.state, throttleCmd, steeringCmd, TIMESTEP)
        print(f"X: {self.state.x} Y: {self.state.y} Speed: {self.state.speed} Yaw: {self.state.yaw}")

        return throttleCmd, steeringCmd, targetWaypoint

def get_switch_back_course(targetSpeed, dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    waypoints = []
    for i in range(len(cx)):
        waypoints.append(Waypoint(cx[i], cy[i], targetSpeed, cyaw[i]))

    waypoints[-1].targetSpeed = 0.0

    return waypoints

def main():
    waypoints = get_switch_back_course(TARGET_SPEED, TIMESTEP)

    initialState = State(waypoints[0].x, waypoints[0].y, 0.1, 0.0)
    model = KinematicBicycle(WHEEL_BASE, MAX_SPEED, MAX_STEERING_ANGLE)

    nmpc = ModelPredictiveController(waypoints, model, initialState)

    cx = np.zeros((len(waypoints), 1))
    cy = np.zeros((len(waypoints), 1))
    for i in range(len(waypoints)):
        cx[i] = waypoints[i].x
        cy[i] = waypoints[i].y

    x = np.zeros((MAX_TIMESTEPS, 1))
    y = np.zeros((MAX_TIMESTEPS, 1))
    speed = np.zeros((MAX_TIMESTEPS, 1))
    
    t = 0
    accel = [0.0] * PRED_HORIZON
    steeringAngle = [0.0] * PRED_HORIZON
    
    while t < MAX_TIMESTEPS:
        throttleCmd, steeringCmd, targetIdx = nmpc.tick(accel, steeringAngle)
        accel = [throttleCmd] * PRED_HORIZON
        steeringAngle = [steeringCmd] * PRED_HORIZON

        x[t] = nmpc.state.x
        y[t] = nmpc.state.y
        speed[t] = nmpc.state.speed

        targetx = np.zeros((PRED_HORIZON, 1))
        targety = np.zeros((PRED_HORIZON, 1))
        for i in range(PRED_HORIZON):
            targetx[i] = nmpc.waypoints[targetIdx + i].x
            targety[i] = nmpc.waypoints[targetIdx + i].y

        #print(t / MAX_TIMESTEPS * 100, end ='\r')

        t += 1

        if True:  # pragma: no cover
            plt.cla()
            plt.plot(cx, cy, "-r")
            plt.plot(x, y, "*g",)
            plt.plot(targetx, targety, "x")
            plt.grid(True)
            plt.pause(0.1)

    plt.show()

    plt.plot(speed,"-r")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()