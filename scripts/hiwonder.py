# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
import numpy as np
from board_controller import BoardController
from servo_bus_controller import ServoBusController
import utils as ut
from math import sin, cos, atan, acos, asin, sqrt, atan2

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters
BASE_SCALE = 100

PI = 3.1415926535897932384
K2 = [1, 0, 0]
K_VEC = [0, 0, 1]  # Used in isolating the Z component of transformation matrices
DET_J_THRESH = 3 * 10 & -5  # Threshold for when determinant is "close to 0"
VEL_SCALE = 0.25  # Scale the EE velocity by this factor when close to a singularity

IK_POINTS = [
    ut.EndEffector(x=14.795, y=6.026, z=5.876, rotx=-2.755, roty=0.014, rotz=3.142),
    ut.EndEffector(x=-1.155, y=-10.183, z=10.257, rotx=1.458, roty=0.214, rotz=3.142),
]
soln = 0


class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees

        self.joint_limits = [
            [-120, 120],
            [-90, 90],
            [-120, 120],
            [-100, 100],
            [-90, 90],
            [-120, 30],
        ]

        # Max joint velocities to avoid dangerous behavior
        self.vel_limits = [
            [-30, 30],
            [-30, 30],
            [-30, 30],
            [-30, 30],
            [-30, 30],
        ]
        self.joint_control_delay = 0.2  # secs
        self.speed_control_delay = 0.2

        # Link lengths (cm)
        self.l1, self.l2, self.l3, self.l4, self.l5 = 15.5, 9.9, 9.5, 5.5, 10.5

        self.dh_params = np.zeros((5, 4))
        self.T = np.zeros((5, 4, 4))
        self.T_cumulative = [np.eye(4)]
        self.jacobian = np.zeros((3, 5))
        self.inv_jacobian = np.zeros((5, 3))

        self.ik_iterator = -1

        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f"---------------------------------------------------------------------")

        self.set_base_velocity(cmd)

        ######################################################################

        position = [0] * 3

        theta = np.deg2rad(self.joint_values.copy())

        # Initialize DH parameters [theta, d, a, alpha]
        self.dh_params = [
            [theta[0], self.l1, 0, (np.pi / 2)],
            [(np.pi / 2) + theta[1], 0, self.l2, 0],
            [-theta[2], 0, self.l3, 0],
            [theta[3], 0, self.l4 + self.l5, theta[4]],
            [(-np.pi / 2), 0, 0, (-np.pi / 2)],
        ]

        # Calculate transformation matrices from DH table
        for index, dh_item in enumerate(self.dh_params):
            self.T[index] = ut.dh_to_matrix(dh_item)

        # Set T0_0 to the identity matrix
        self.T_cumulative = [np.eye(4)]

        # Calculate remaining cumulative transformation matrices
        for i in range(len(self.T)):
            self.T_cumulative.append(np.array(self.T_cumulative[-1] @ self.T[i]))

        # Extract end-effector position from final transformation matrix
        position = [
            self.T_cumulative[-1][0, 3],
            self.T_cumulative[-1][1, 3],
            self.T_cumulative[-1][2, 3],
        ]

        roll, pitch, yaw = ut.rotm_to_euler(self.T_cumulative[-1][:3, :3])

        ######################################################################

        print(
            f"[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n\
            [DEBUG] Euler Angles: Roll: {round(roll, 3)}, Pitch: {round(pitch, 3)}, Yaw: {round(yaw, 3)} \n"
        )

        if cmd.utility_btn:
            self.analytical_ik()
        else:
            self.set_arm_velocity(cmd)

    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """Computes wheel speeds based on joystick input and sends them to the board"""
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        
        speed = np.zeros(4)

        vx, vy, w = cmd.base_vx, cmd.base_vy, cmd.base_w

        v_ang = atan2(vy, vx)
        v_mag = sqrt(vx**2 + vy**2)

        speed[0] = (sin(v_ang - (PI / 4)) * v_mag) + w
        speed[1] = (-sin(v_ang + (PI / 4)) * v_mag) - w
        speed[2] = (-sin(v_ang + (PI / 4)) * v_mag) + w
        speed[3] = (sin(v_ang - (PI / 4)) * v_mag) - w

        if max(speed) > 1 or max(speed) < -1:
            speed /= abs(max(speed))

        speed = speed * BASE_SCALE

        ######################################################################

        print(speed)
        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def analytical_ik(self):
        self.ik_iterator += 1
        if self.ik_iterator == len(IK_POINTS):
            self.ik_iterator = 0
        EE = IK_POINTS[self.ik_iterator]

        x, y, z = EE.x, EE.y, EE.z

        R_05 = ut.euler_to_rotm((EE.rotx, EE.roty, EE.rotz))
        R_05_K = R_05 @ K_VEC

        p_wrist = [x, y, z] - ((self.l4 + self.l5) * (R_05_K))

        wx = p_wrist[0]
        wy = p_wrist[1]
        wz = p_wrist[2]

        r = sqrt((wx**2 + wy**2) + ((wz - self.l1) ** 2))

        # Theta 1 standard solve
        j1 = ut.wraptopi(atan2(y, x) + np.pi)  # seems to give desired - 180

        # Theta 2 standard solve
        alpha = acos((r**2 + self.l2**2 - self.l3**2) / (2 * r * self.l2))
        phi = acos((wz - self.l1) / (r))
        j2 = alpha + phi

        # Theta 3 standard solve
        j3 = acos((r**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3))

        # Set of 4 potential solutions
        #! @EarlJr53 interested in adding more solutions perhaps?
        solns = np.array(
            [
                # Standard configuration
                [j1, j2, j3, 0, 0],
                # Flipped elbow
                [j1, -alpha + phi, -j3, 0, 0],
                # Mirrored base
                [ut.wraptopi(j1 + PI), -alpha - phi, -j3, 0, 0],
                # Mirrored base, flipped elbow
                [ut.wraptopi(j1 + PI), alpha - phi, j3, 0, 0],
            ]
        )

        # Keep track of how many valid solutions have been "skipped"
        valid_solns_count = 0

        new_thetalist = [0.0] * 6

        for angs in solns:
            # Temporary DH matrix
            mini_DH = [
                [angs[0], self.l1, 0, np.pi / 2],
                [(np.pi / 2) + angs[1], 0, self.l2, 0],
                [-angs[2], 0, self.l3, 0],
            ]

            # Translation matrix from 0 to 3
            t_03 = np.eye(4)
            for dh_item in mini_DH:
                t_temp = ut.dh_to_matrix(dh_item)
                t_03 = t_03 @ t_temp

            # Rotation matrix from 0 to 3
            R_03 = t_03[:3, :3]

            # Rotation matrix from 3 to 5
            R_35 = np.transpose(R_03) @ R_05

            # Solve for Theta 4 and Theta 5
            angs[3] = atan2(R_35[1, 2], R_35[0, 2])
            angs[4] = ut.wraptopi(atan2(R_35[2, 0], R_35[2, 1]) + PI)

            # Check if the current configuration is valid
            if ut.check_joint_limits(angs, np.deg2rad(self.joint_limits)):
                # Check if we've reached either the requested `soln` or the last valid solution
                if soln is valid_solns_count or valid_solns_count is len(solns) - 1:
                    new_thetalist[0:5] = angs
                    break
                # "Skip" valid solutions until the requested `soln`
                else:
                    last_valid = angs  # Hang onto the most recent valid solution
                    valid_solns_count += 1
            # If we've made it to the end and don't have another valid solution, use the most recent valid one
            elif valid_solns_count is len(solns) - 1:
                new_thetalist[0:5] = last_valid

        new_thetalist[5] = self.joint_values[5]
        self.set_joint_values(new_thetalist, duration=1000, radians=True)

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities.

        Args:
            cmd (GamepadCmds): Contains linear velocities for the arm.
        """
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
        vel = np.array(vel)

        ######################################################################

        thetalist_dot = np.zeros((5))

        # Calculate Jacobian from cumulative transformation matrices
        for index, transform in enumerate(self.T_cumulative):
            if index == 5:  # Don't need the T4_5 matrix for this step
                break
            else:
                # For each column of the Jacobian,
                self.jacobian[:, index] = np.cross(
                    transform[:3, :3] @ K_VEC,
                    (self.T_cumulative[5][:3, 3] - transform[:3, 3]),
                )

        # Invert speed for flipped motor
        self.jacobian[:, 2] = -self.jacobian[:, 2]

        # Generate pseudoinverse of Jacobian
        self.inv_jacobian = np.linalg.pinv(self.jacobian)

        # Calculate determinant of Jacobian for singularity avoidance
        det_J = np.linalg.det(np.dot(self.jacobian, np.transpose(self.jacobian)))

        # Scale EE velocity down if close to a singularity
        if abs(det_J) < DET_J_THRESH:
            vel = vel * VEL_SCALE

        # Get desired joint velocities from inverse Jacobian and EE velocity
        thetalist_dot = np.rad2deg(self.inv_jacobian @ vel)

        # Clip joint velocities to present max velocities
        for joint, limits in enumerate(self.vel_limits):
            if thetalist_dot[joint] < limits[0]:
                thetalist_dot[joint] = limits[0]
            elif thetalist_dot[joint] > limits[1]:
                thetalist_dot[joint] = limits[1]

        ######################################################################

        print(f"[DEBUG] Current thetalist (deg) = {self.joint_values}")
        print(
            f"[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}"
        )
        print(f"[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}")

        # Update joint angles
        dt = 0.5  # Fixed time step
        K = 500  # mapping gain for individual joint control
        new_thetalist = [0.0] * 6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * thetalist_dot[i]
        # individual joint control
        new_thetalist[0] += dt * K * cmd.arm_j1
        new_thetalist[1] += dt * K * cmd.arm_j2
        new_thetalist[2] += dt * K * cmd.arm_j3
        new_thetalist[3] += dt * K * cmd.arm_j4
        new_thetalist[4] += dt * K * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K * cmd.arm_ee

        new_thetalist = [round(theta, 2) for theta in new_thetalist]
        print(f"[DEBUG] Commanded thetalist (deg) = {new_thetalist}")

        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)

    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """Moves a single joint to a specified angle"""
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.rad2deg(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)

        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)

    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles.

        Args:
            thetalist (list): Target joint angles in degrees.
            duration (int): Movement duration in milliseconds.
        """
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.rad2deg(theta) for theta in thetalist]

        thetalist = self.enforce_joint_limits(thetalist)
        self.joint_values = thetalist  # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(
            thetalist
        )  # remap the joint values from software to hardware

        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            self.servo_bus.move_servo(joint_id, pulse, duration)

    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [
            np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)
        ]

    def move_to_home_position(self):
        print(f"Moving to home position...")
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f"Arrived at home position: {self.joint_values} \n")
        time.sleep(1.0)
        print(f"------------------- System is now ready!------------------- \n")

    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """Converts degrees to servo pulse value"""
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int(
            (x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min
        )

    def pulse_to_angle(self, x: float):
        """Converts servo pulse value to degrees"""
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round(
            (x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2
        )

    def stop_motors(self):
        """Stops all motors safely"""
        self.board.set_motor_speed([0] * 4)
        print("[INFO] Motors stopped.")

    def remap_joints(self, thetalist: list):
        """Reorders angles to match hardware configuration.

        Args:
            thetalist (list): Software joint order.

        Returns:
            list: Hardware-mapped joint angles.

        Note: Joint mapping for hardware
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5]
            joint[2] = joint[4]
            joint[3] = joint[3]
            joint[4] = joint[2]
            joint[5] = joint[1]
        """
        return thetalist[::-1]
