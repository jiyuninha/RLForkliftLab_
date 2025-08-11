from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import ackermann_actions


@configclass
class AckermannActionCfg(ActionTermCfg):
    """Configuration for the Ackermann steering action term.

    This class configures the parameters for Ackermann steering,
    typically used in wheeled robots or vehicles.
    """

    class_type: type[ActionTerm] = ackermann_actions.AckermannAction
    """The specific action class type for Ackermann steering."""

    scale: tuple[float, float] = (1.0, 1.0)
    """The scale of the action term."""

    offset: tuple[float, float] = 0.0
    """The offset of the action term."""

    wheelbase_length: float = 1.5
    """The distance between the front and rear wheels."""

    middle_wheel_distance: float = 0.8
    """The distance between the middle wheels."""

    rear_and_front_wheel_distance: float = 1.6
    """The distance between the rear and front wheels."""

    wheel_radius: float = MISSING
    """The radius of the wheels."""

    min_steering_radius: float = MISSING
    """The minimum steering angle of the vehicle, if lower than this value, the vehicle will turn on the spot."""

    steering_joint_names: list[str] = MISSING

    drive_joint_names: list[str] = MISSING

    steering_order = ["back_wheel_swivel"]
    """ Name of the steering joints in the following order ["FL", "FR", "RL", "RR"],
    e.g. if the front left joint is named "front_left", change "FL" to "front_left" """

    drive_order = ["back_wheel_drive"]
    """ Name of the drive joints in the following order ["FL", "FR", "CL", "CR", "RL", "RR"],
    e.g. if the front left joint is named "front_left", change "FL" to "front_left" """
