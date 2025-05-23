# gamepad_control.py
"""
Gamepad Control Module
----------------------
This module initializes a connected gamepad and translates its inputs into robot commands.
"""

import inputs
import time
from utils import GamepadCmds


class GamepadControl:
    """Handles gamepad input and converts it into robot commands."""

    def __init__(self):
        """Initialize the gamepad and internal state variables."""
        self.initialize_gamepad()
        self.gamepad_cmds_prev = GamepadCmds()
        # Default centered values
        self.abs_x = self.abs_rx = 128
        self.abs_y = self.abs_ry = -128

        # Control flags
        self.MANUAL_ARM_FLAG = False
        self.ARM_FLAG = False
        self.ARM_J1_FLAG = False
        self.ARM_J2_FLAG = False
        self.ARM_J3_FLAG = False
        self.ARM_J4_FLAG = False
        self.ARM_J5_FLAG = False
        self.ARM_EE_FLAG = False
        self.ARM_HOME = False
        self.COLLECT_BTN = False

        # # Add these lines to existing init
        # self.last_command_time = time.time()
        # self.min_command_interval = 0.1  # 100ms between commands

    def initialize_gamepad(self):
        """Attempts to initialize the first connected gamepad."""
        for attempt in range(10):
            gamepads = inputs.devices.gamepads
            if gamepads:
                self.gamepad = gamepads[0]
                print(f"[INFO] Using gamepad: {self.gamepad}")
                return True
            print(f"[WARNING] No gamepads detected. Retry [{attempt + 1}/10]...")
            time.sleep(1)

        raise RuntimeError("Failed to detect gamepad. Please check the USB connection.")

    def get_gamepad_cmds(self):
        """Fetches and maps gamepad events to robot commands.

        Returns:
            GamepadCmds: Updated command object reflecting current gamepad input.
        """

        # current_time = time.time()

        # # Check if enough time has passed since last command
        # if current_time - self.last_command_time < self.min_command_interval:
        #     return self.gamepad_cmds_prev

        gamepad_cmds = GamepadCmds()
        events = self.gamepad._do_iter()

        if events is None:
            return (
                self.gamepad_cmds_prev
            )  # Return previous commands if no events are detected

        for event in events:
            if event.ev_type != "Sync":
                self._handle_event(event)

                # print(f'type: {event.code}, state: {event.state}')

        gamepad_cmds.arm_home = int(self.ARM_HOME)
        gamepad_cmds.collect_btn = int(self.COLLECT_BTN)

        if self.ARM_FLAG:
            gamepad_cmds.arm_vx = self.map_value(
                self.abs_x, [-32767, 32767], [-0.1, 0.1]
            )
            gamepad_cmds.arm_vy = self.map_value(
                self.abs_y, [32767, -32767], [-0.1, 0.1]
            )
            gamepad_cmds.arm_vz = self.map_value(
                self.abs_rx, [-32767, 32767], [-0.1, 0.1]
            )

        elif self.MANUAL_ARM_FLAG:
            gamepad_cmds.arm_j1 = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_J1_FLAG
                else 0.0
            )
            gamepad_cmds.arm_j2 = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_J2_FLAG
                else 0.0
            )
            gamepad_cmds.arm_j3 = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_J3_FLAG
                else 0.0
            )
            gamepad_cmds.arm_j4 = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_J4_FLAG
                else 0.0
            )
            gamepad_cmds.arm_j5 = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_J5_FLAG
                else 0.0
            )
            gamepad_cmds.arm_ee = (
                self.map_value(self.abs_x, [-32767, 32767], [-0.1, 0.1])
                if self.ARM_EE_FLAG
                else 0.0
            )

        else:
            gamepad_cmds.base_vx = self.map_value(
                self.abs_x, [-32767, 32767], [-0.5, 0.5]
            )
            gamepad_cmds.base_vy = self.map_value(
                self.abs_y, [32767, -32767], [-0.5, 0.5]
            )
            gamepad_cmds.base_w = self.map_value(
                self.abs_rx, [32767, -32767], [-0.5, 0.5]
            )

        # self.last_command_time = current_time  # Update last command time

        self.gamepad_cmds_prev = gamepad_cmds
        return gamepad_cmds

    def _handle_event(self, event):
        """Handles individual gamepad events and updates internal state."""
        code_map = {
            "ABS_X": ("abs_x", event.state),
            "ABS_Y": ("abs_y", event.state),
            "ABS_RX": ("abs_rx", event.state),
            "ABS_RY": ("abs_ry", event.state),
            "BTN_TR": ("COLLECT_BTN", bool(event.state)),
            "BTN_TL": ("MANUAL_ARM_FLAG", bool(event.state)),
            # "BTN_TR": ("ARM_FLAG", bool(event.state)),
            "BTN_WEST": ("ARM_J1_FLAG", bool(event.state)),
            "BTN_EAST": ("ARM_J2_FLAG", bool(event.state)),
            "BTN_SOUTH": ("ARM_J3_FLAG", bool(event.state)),
            "BTN_NORTH": ("ARM_J4_FLAG", bool(event.state)),
            "ABS_RZ": ("ARM_J5_FLAG", bool(event.state)),
            "ABS_Z": ("ARM_EE_FLAG", bool(event.state)),
            "BTN_SELECT": ("ARM_HOME", bool(event.state)),
        }

        if event.code in code_map:
            setattr(self, code_map[event.code][0], code_map[event.code][1])

    @staticmethod
    def map_value(x: float, in_range: list, out_range: list) -> float:
        """Maps an input value from hardware range (0-255) to a desired output range.

        Args:
            x (float): Input value (0 to 255).
            in_range (list): [in_min, in_max]
            out_range (list): [out_min, out_max]

        Returns:
            float: Mapped value.
        """
        # val = (x - joint_min) * (out_max - out_min) / (joint_max - joint_min) + out_min
        val = (x - in_range[0]) * (out_range[1] - out_range[0]) / (
            in_range[1] - in_range[0]
        ) + out_range[0]
        return val if abs(val) > 0.005 else 0.0
