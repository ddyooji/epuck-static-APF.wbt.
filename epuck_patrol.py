"""E-puck room patrol controller.

Visits WAYPOINTS in a loop. Moves toward the current waypoint with an
artificial potential field (attraction to the goal + repulsion from the
8 IR distance sensors). Position comes from wheel odometry, so waypoints
are in meters RELATIVE TO THE ROBOT'S START POSE (x = forward, y = left).
Edit WAYPOINTS to match your room.
"""
import math
import random

from controller import Robot

# ---- Patrol route (meters, relative to start pose; x forward, y left) ----
# Route for "epuck static APF.wbt": 4x4 m arena, robot starts at world
# (X=-1.8, Z=-0.9) facing +X (NUE coordinates). Relative to the start:
#   x = X + 1.8,   y = -(Z + 0.9)   (left of the robot is -Z)
# The route keeps >= 0.25 m from every box and cylinder in that world.
WAYPOINTS = [
    (0.00, 0.70),
    (1.00, 0.70),
    (2.60, 0.70),
    (3.50, 0.70),
    (3.50, -1.40),
    (3.50, -2.60),
    (2.30, -2.60),
    (2.30, -1.30),
    (0.80, -1.10),
    (0.00, -0.90),
    (0.00, 0.00),
]

# ---- Robot constants (e-puck) ----
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
MAX_SPEED = 6.28  # rad/s

# IR sensor directions in the robot frame (rad, CCW positive): ps0..ps7
SENSOR_ANGLES = [-0.30, -0.80, -1.57, -2.64, 2.64, 1.57, 0.80, 0.30]

# ---- APF / control parameters ----
REACH_DIST = 0.04      # waypoint reached within this distance (m)
K_ATT = 1.0            # attraction gain
K_REP = 2.5            # repulsion gain
SENSOR_THRESHOLD = 100.0
SENSOR_SCALE = 1000.0
K_TURN = 3.0
STUCK_WINDOW = 3.0     # s
STUCK_MIN_MOVE = 0.02  # m moved in the window, else considered stuck
ESCAPE_TIME = 1.5      # s


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

left_enc = left_motor.getPositionSensor()
right_enc = right_motor.getPositionSensor()
left_enc.enable(timestep)
right_enc.enable(timestep)

ps = []
for i in range(8):
    s = robot.getDevice("ps%d" % i)
    s.enable(timestep)
    ps.append(s)

x = y = theta = 0.0
prev_l = prev_r = None
wp_index = 0
stuck_ref = (0.0, 0.0)
stuck_timer = 0.0
escape_timer = 0.0
escape_dir = 1

while robot.step(timestep) != -1:
    # ---- Odometry ----
    l, r = left_enc.getValue(), right_enc.getValue()
    if prev_l is None:
        prev_l, prev_r = l, r
        continue
    dl = (l - prev_l) * WHEEL_RADIUS
    dr = (r - prev_r) * WHEEL_RADIUS
    prev_l, prev_r = l, r
    d = (dl + dr) / 2.0
    dth = (dr - dl) / AXLE_LENGTH
    x += d * math.cos(theta + dth / 2.0)
    y += d * math.sin(theta + dth / 2.0)
    theta = wrap(theta + dth)

    # ---- Waypoint handling ----
    wx, wy = WAYPOINTS[wp_index]
    ex, ey = wx - x, wy - y
    dist = math.hypot(ex, ey)
    if dist < REACH_DIST:
        wp_index = (wp_index + 1) % len(WAYPOINTS)
        continue

    # ---- Stuck detection ----
    stuck_timer += dt
    if stuck_timer >= STUCK_WINDOW:
        if math.hypot(x - stuck_ref[0], y - stuck_ref[1]) < STUCK_MIN_MOVE:
            escape_timer = ESCAPE_TIME
            escape_dir = random.choice((-1, 1))
        stuck_ref = (x, y)
        stuck_timer = 0.0

    if escape_timer > 0.0:
        escape_timer -= dt
        left_motor.setVelocity(clamp(0.6 * MAX_SPEED * (1 - 0.8 * escape_dir), -MAX_SPEED, MAX_SPEED))
        right_motor.setVelocity(clamp(0.6 * MAX_SPEED * (1 + 0.8 * escape_dir), -MAX_SPEED, MAX_SPEED))
        continue

    # ---- Potential field in the robot frame ----
    gx = math.cos(theta) * ex + math.sin(theta) * ey
    gy = -math.sin(theta) * ex + math.cos(theta) * ey
    fx = K_ATT * gx / dist
    fy = K_ATT * gy / dist

    closest = 0.0
    for a, sensor in zip(SENSOR_ANGLES, ps):
        v = sensor.getValue()
        if v > SENSOR_THRESHOLD:
            strength = K_REP * (v - SENSOR_THRESHOLD) / SENSOR_SCALE
            fx -= strength * math.cos(a)
            fy -= strength * math.sin(a)
            closest = max(closest, v)

    err = math.atan2(fy, fx)
    w = clamp(K_TURN * err, -MAX_SPEED, MAX_SPEED)
    slow = 1.0 - clamp((closest - SENSOR_THRESHOLD) / 2000.0, 0.0, 0.7)
    v = MAX_SPEED * 0.8 * max(0.0, math.cos(err)) * slow

    left_motor.setVelocity(clamp(v - w, -MAX_SPEED, MAX_SPEED))
    right_motor.setVelocity(clamp(v + w, -MAX_SPEED, MAX_SPEED))
