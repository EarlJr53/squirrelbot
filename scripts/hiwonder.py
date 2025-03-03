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

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters

K_VEC = [0, 0, 1]
DET_J_THRESH = 3 * 10 & -5
VEL_SCALE = 0.25

class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees
        # self.home_position = [20, 20, 20, 20, 20, 0]  # degrees

        self.joint_limits = [
            [-120, 120], [-90, 90], [-120, 120],
            [-100, 100], [-90, 90], [-120, 30]
        ]
        self.vel_limits = [
            [-30, 30],
            [-30, 30],
            [-30, 30],
            [-30, 30],
            [-30, 30],
        ]
        self.joint_control_delay = 0.2 # secs
        self.speed_control_delay = 0.2

        # Link lengths
        # self.l1, self.l2, self.l3, self.l4, self.l5 = .155, .099, .095, .055, .105
        self.l1, self.l2, self.l3, self.l4, self.l5 = 15.5, 9.9, 9.5, 5.5, 10.5


        self.dh_params = np.zeros((5,4))
        # self.T = np.array([np.eye(4) for _ in range(5)])
        self.T = np.zeros((5, 4, 4))
        # self.T_cumulative = np.array([np.eye(4) for _ in range(5)])
        self.T_cumulative = [np.eye(4)]
        self.jacobian = np.zeros((3, 5))
        self.inv_jacobian = np.zeros((5, 3))

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

        print(f'---------------------------------------------------------------------')
        
        # self.set_base_velocity(cmd)
        

        ######################################################################

        position = [0]*3

        theta = np.deg2rad(self.joint_values.copy())

        # Initialize DH parameters [theta, d, a, alpha]
        self.dh_params = [
            [theta[0], self.l1, 0, (np.pi / 2)],
            [(np.pi / 2) + theta[1], 0, self.l2, 0],
            [-theta[2], 0, self.l3, 0],
            [theta[3], 0, self.l4 + self.l5, theta[4]],
            # [theta[3], 0, self.l4, (np.pi / 2)],
            [(-np.pi / 2), 0, 0, (-np.pi / 2)],
            # [0, 0, self.l5, theta[4]],
        ]
        # self.dh_params = [
        #     [theta[0], self.l1, 0, (-np.pi / 2)],
        #     [theta[1] - (np.pi/2), 0, self.l2, np.pi],
        #     [theta[2], 0, self.l3, np.pi],
        #     [theta[3] + (np.pi/2), 0, 0, (np.pi)],
        #     [theta[4], self.l4+self.l5, 0, 0]
        # ]

        for index, dh_item in enumerate(self.dh_params):
            self.T[index] = ut.dh_to_matrix(dh_item)

        # print(len(self.T))

        self.T_cumulative = [np.eye(4)]
        # self.T_cumulative = np.array(self.T_cumulative)
        for i in range(len(self.T)):
            self.T_cumulative.append(np.array(self.T_cumulative[-1] @ self.T[i]))

        # print(self.T_cumulative)

        # Extract end-effector position from final transformation matrix
        position = [
            self.T_cumulative[-1][0,3],
            self.T_cumulative[-1][1,3],
            self.T_cumulative[-1][2,3]
        ]
        
        ######################################################################

        print(f'[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n')

        self.set_arm_velocity(cmd)


    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """ Computes wheel speeds based on joystick input and sends them to the board """
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        # insert your code for finding "speed"

        speed = [0]*4
        
        ######################################################################

        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def compute_Jacobian(self):

        # Compute Jacobian matrix for the robot arm
        angles_rad = np.deg2rad(self.joint_values)

        sigma1 = np.cos(angles_rad[0] - angles_rad[1] + angles_rad[2] - angles_rad[3])
        sigma2 = np.cos(angles_rad[0] + angles_rad[1] - angles_rad[2] + angles_rad[3])
        sigma6 = np.sin(angles_rad[1] - angles_rad[2] + angles_rad[3])
        sigma8 = np.cos(angles_rad[1] - angles_rad[2] + angles_rad[3])
        sigma7 = self.l3 * np.cos(angles_rad[1] - angles_rad[2])
        sigma5 = self.l3 * np.sin(angles_rad[1] - angles_rad[2])
        sigma3 = self.l4 * sigma6 + self.l5 * sigma6 + self.l2 * np.sin(angles_rad[1]) + sigma5
        sigma4 = self.l4 * sigma8 + self.l5 * sigma8 + self.l2 * np.cos(angles_rad[1]) + sigma7
        
        J_v = np.array([
            [np.sin(angles_rad[0]) * sigma3, -np.cos(angles_rad[0]) * sigma4, 
            (self.l4 * sigma1 / 2) + (self.l5 * sigma1 / 2) + (self.l3 * np.cos(angles_rad[0] + angles_rad[1] - angles_rad[2]) / 2) + 
            (self.l3 * np.cos(angles_rad[0] - angles_rad[1] + angles_rad[2]) / 2) + (self.l4 * sigma2 / 2) + (self.l5 * sigma2 / 2),
            -((self.l4 + self.l5) * (sigma1 + sigma2) / 2), 0],
            [-np.cos(angles_rad[0]) * sigma3, -np.sin(angles_rad[0]) * sigma4, 
            np.sin(angles_rad[0]) * (self.l4 * sigma8 + self.l5 * sigma8 + sigma7),
            -((self.l4 + self.l5) * (np.sin(angles_rad[0] + angles_rad[1] - angles_rad[2] + angles_rad[3]) + np.sin(angles_rad[0] - angles_rad[1] + angles_rad[2] - angles_rad[3])) / 2), 0],
            [0, -self.l4 * sigma6 - self.l5 * sigma6 - self.l2 * np.sin(angles_rad[1]) - sigma5, self.l4 * sigma6 + self.l5 * sigma6 + sigma5, -sigma6 * (self.l4 + self.l5), 0]
        ])

        return J_v

    def jacobian_other(self):


        t_cumul = self.T[0] @ self.T[1] @ self.T[2] @ self.T[3] @ self.T[4]

        J = np.empty((3, 5))

        t1 = t_cumul[0:3, 3]
        r1 = np.array([0, 0, 1])
        j1 = np.cross(r1, t1)
        J[0:3, 0] = j1

        t2 = t_cumul[0:3, 3] - self.T[0][0:3, 3]
        r2 = self.T[0][0:3, 2]
        j2 = np.cross(r2, t2)
        J[0:3, 1] = j2

        t3 = t_cumul[0:3, 3] - (self.T[0] @ self.T[1])[0:3, 3]
        r3 = (self.T[0] @ self.T[1])[0:3, 2]
        j3 = np.cross(r3, t3)
        J[0:3, 2] = j3

        t4 = t_cumul[0:3, 3] - (self.T[0] @ self.T[1] @ self.T[2])[0:3, 3]
        r4 = (self.T[0] @ self.T[1] @ self.T[2])[0:3, 2]
        j4 = np.cross(r4, t4)
        J[0:3, 3] = j4

        t5 = t_cumul[0:3, 3] - (self.T[0] @ self.T[1] @ self.T[2] @ self.T[3])[0:3, 3]
        r5 = (self.T[0] @ self.T[1] @ self.T[2] @ self.T[3])[0:3, 2]
        j5 = np.cross(r5, t5)
        J[0:3, 4] = j5

        return J

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities.

        Args:
            cmd (GamepadCmds): Contains linear velocities for the arm.
        """
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
        vel = np.array(vel)

        ######################################################################
        # insert your code for finding "thetalist_dot"

        thetalist_dot = np.zeros((5))

        # print(len(self.T_cumulative))
        # self.T_cumulative = np.array(self.T_cumulative)
        # print(self.T_cumulative.shape)

        # J = np.zeros((5, 3))
        # # Extracts the translational component to get from frame 0 to the end-effector frame
        # d = self.T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # # Calculates the jacobian by crossing the distance to the end-effector and the z component of rotation
        # for i in range(0, 5):
        #     T_i = self.T_cumulative[i]
        #     z = T_i @ np.vstack([0, 0, 1, 0])
        #     d1 = T_i @ np.vstack([0, 0, 0, 1])
        #     r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
        #     J[i] = np.cross(z[:3].flatten(), r.flatten())

        # self.jacobian = np.transpose(J)
        for index, transform in enumerate(self.T_cumulative):
            if index == 5:
                break
            else:
                self.jacobian[:, index] = np.cross(
                    transform[:3, :3] @ K_VEC,
                    (self.T_cumulative[5][:3, 3] - transform[:3, 3]),
                )
        # self.jacobian = self.jacobian_other()
        # self.jacobian = self.compute_Jacobian()

        # print(self.jacobian)

        self.inv_jacobian = np.linalg.pinv(self.jacobian)

        det_J = np.linalg.det(np.dot(self.jacobian, np.transpose(self.jacobian)))

        if abs(det_J) < DET_J_THRESH:
            vel = vel * VEL_SCALE

        thetalist_dot = np.rad2deg(self.inv_jacobian @ vel)
        # thetalist_dot = self.inv_jacobian @ vel

        # thetalist_unbounded = thetalist_dot.copy()

        for joint, limits in enumerate(self.vel_limits):
            if thetalist_dot[joint] < limits[0]:
                thetalist_dot[joint] = limits[0]
            elif thetalist_dot[joint] > limits[1]:
                thetalist_dot[joint] = limits[1]

        ######################################################################


        print(f'[DEBUG] Current thetalist (deg) = {self.joint_values}') 
        print(f'[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}')
        print(f'[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}')
        # print(f'[DEBUG] thetadot_unbounded (deg/s) = {[round(td,2) for td in thetalist_unbounded]}')


        # Update joint angles
        dt = 0.5 # Fixed time step
        K = 10 # mapping gain for individual joint control
        new_thetalist = [0.0]*6

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

        new_thetalist = [round(theta,2) for theta in new_thetalist]
        print(f'[DEBUG] Commanded thetalist (deg) = {new_thetalist}')       
        
        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)


    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle """
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
        self.joint_values = thetalist # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(thetalist) # remap the joint values from software to hardware

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
        return [np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)]


    def move_to_home_position(self):
        print(f'Moving to home position...')
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f'Arrived at home position: {self.joint_values} \n')
        time.sleep(1.0)
        print(f'------------------- System is now ready!------------------- \n')


    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """ Converts degrees to servo pulse value """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int((x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min)


    def pulse_to_angle(self, x: float):
        """ Converts servo pulse value to degrees """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round((x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2)


    def stop_motors(self):
        """ Stops all motors safely """
        self.board.set_motor_speed([0]*4)
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
