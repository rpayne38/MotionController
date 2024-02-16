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

WHEEL_BASE = 2.5
MAX_SPEED = 12.0
MAX_STEERING_ANGLE = 0.8
TARGET_SPEED = 5.0

TIMESTEP = 0.05
MAX_TIMESTEPS = 2000
PRED_HORIZON = 15
MAX_ITER = 5
U_CONVERGENCE = 0.1

NX = 4  # [x, y, v, yaw]
NY = 4  # reference state variables
NYN = 4  # reference terminal state variables
NU = 2  # [accel, delta]

# mpc parameters
Q = np.diag([1.0, 1.0, 0.1, 0.8])  # state cost matrix
Qf = Q  # state final matrix

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

class ModelPredictiveController():
    def __init__(self, waypoints : list[Waypoint], model : KinematicBicycle):
        self.waypoints = waypoints
        self.model = model

        self.steeringCmds = np.zeros((PRED_HORIZON))
        self.throttleCmds = np.zeros((PRED_HORIZON))

    def _predict(self, accel: list[float], steeringAngle: list[float], state : State):
        ret = [self.model.update(state, accel[0], steeringAngle[0], TIMESTEP)]
        for i in range(PRED_HORIZON):
            ret.append(self.model.update(ret[i], accel[i], steeringAngle[i], TIMESTEP))

        return ret
    
    def _solve(self, xPred: list[State], state):
        # see acado.c for parameter details
        # xPred = predicted state at each timestep

        _x0 = np.zeros((1,NX))                  # initial state
        X = np.zeros((PRED_HORIZON + 1, NX))    # predicted state
        U = np.zeros((PRED_HORIZON, NU))        # optimal control input 
        Y = np.zeros((PRED_HORIZON, NY))        # reference state
        yN = np.zeros((1,NYN))                  # final reference state

        nearestWaypointIdx, dist = self.nearestWaypoint(state)

        if (nearestWaypointIdx + PRED_HORIZON) > len(self.waypoints):
            return None, None, None, None

        for t in range(PRED_HORIZON):
            Y[t,:] = self.waypoints[nearestWaypointIdx + t].numpyArray()  # reference state
            X[t,:] = xPred[t].numpyArray()                                # predicted state

        X[-1:] = X[-2,:]
        _x0[0,:] = state.numpyArray()
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
    
    def nearestWaypoint(self, state : State):
        pos = np.array([state.x, state.y])

        minDist = float("inf")
        idx = len(self.waypoints) + 1
        for i, waypoint in enumerate(self.waypoints):
            dist = math.dist(pos, waypoint.pos())
            if dist < minDist:
                minDist = dist
                idx = i

        return idx, minDist

    
    def tick(self, state : State):

        accel = self.throttleCmds
        steeringAngle = self.steeringCmds

        for i in range(MAX_ITER):
            # Predict next states based on current steering angle and acceleration
            predictedStates = self._predict(accel, steeringAngle, state)

            predictedAccel = np.copy(accel)
            predictedSteeringAngle = np.copy(steeringAngle)
                    
            # solve cost func to get new state
            accel, steeringAngle, newState, targetWaypoint = self._solve(predictedStates, state)

            if targetWaypoint is None:
                return None, None, None
            
            du = np.sum(abs(accel - predictedAccel) + abs(steeringAngle - predictedSteeringAngle))
            if du <= U_CONVERGENCE:
                break

        self.steeringCmds = steeringAngle
        self.throttleCmds = accel

        #print(f"X: {self.state.x} Y: {self.state.y} Speed: {self.state.speed} Yaw: {self.state.yaw}")

        return self.steeringCmds[0], self.throttleCmds[0], targetWaypoint

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

    state = State(waypoints[0].x, waypoints[0].y, 0.1, -1.57)
    model = KinematicBicycle(WHEEL_BASE, MAX_SPEED, MAX_STEERING_ANGLE)

    nmpc = ModelPredictiveController(waypoints, model)

    cx = np.zeros((len(waypoints), 1))
    cy = np.zeros((len(waypoints), 1))
    for i in range(len(waypoints)):
        cx[i] = waypoints[i].x
        cy[i] = waypoints[i].y

    x = np.array([state.x])
    y = np.array([state.y])
    speed = np.array([state.speed])
    
    t = 0
    while t < MAX_TIMESTEPS:
        steeringCmd, throttleCmd, targetIdx = nmpc.tick(state)
        if targetIdx is None:
            break
        
        state = model.update(state, throttleCmd, steeringCmd, TIMESTEP)

        x = np.append(x,  state.x)
        y = np.append(y, state.y)
        speed = np.append(speed, state.speed)

        targetx = np.zeros((PRED_HORIZON, 1))
        targety = np.zeros((PRED_HORIZON, 1))
        for i in range(PRED_HORIZON):
            targetx[i] = nmpc.waypoints[targetIdx + i].x
            targety[i] = nmpc.waypoints[targetIdx + i].y

        print(f"{(t / MAX_TIMESTEPS * 100):.1f}", end ='\r')

        t += 1

    if True:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, "-r")
        plt.plot(x, y, "*g",)
        plt.plot(targetx, targety, "x")
        plt.grid(True)
        #plt.pause(0.001)

    plt.show()

    plt.plot(speed,"-r")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()