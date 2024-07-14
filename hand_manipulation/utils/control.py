"""Some implemented controllers."""

import numpy as np


class PIDController:
    """Class to control the Hand in the MuJoCo environment."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_clip: float = 1.0,
    ) -> None:
        """Initialize the PID Controller class.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            integral_clip: Maximum value of the integral error.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_clip = integral_clip

        self.error = 0.0
        self.previous_error = 0.0
        self.intregral_error = 0.0

    def control(self, target_position: np.ndarray, current_position: np.ndarray) -> np.ndarray:
        """Compute the control signal for the PID controller."""
        self.error = target_position - current_position
        self.intregral_error += self.error
        derivative_error = self.error - self.previous_error

        proportional_signal = self.kp * self.error
        integral_signal = np.clip(
            self.ki * self.intregral_error,
            -self.integral_clip,
            self.integral_clip,
        )
        derivative_signal = self.kd * derivative_error

        control_signal = proportional_signal + integral_signal + derivative_signal

        self.previous_error = self.error

        return control_signal
