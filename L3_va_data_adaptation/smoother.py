from collections import deque


class DataSmoother:
    def __init__(self, method: str, params: dict):
        self.method = method.lower()
        self.params = params

        # Method-specific initialization
        if self.method == "rolling_buffer":
            self.window_size = self.params["window_size"]
            self.history_v = deque(maxlen=self.window_size)
            self.history_a = deque(maxlen=self.window_size)
            self.v_sum = 0.0
            self.a_sum = 0.0
        elif self.method == "ema":
            self.alpha = self.params["alpha"]
            self.last_v = None
            self.last_a = None
        elif self.method == "steps":
            # For "steps", we will round to nearest integer and only allow adjacent moves
            self.max_step_delta = self.params["max_step_delta"]
            self.last_v = None
            self.last_a = None
        elif self.method == "none":
            pass  # No initialization needed for no smoothing
        else:
            raise ValueError(f"Unsupported smoothing method: {self.method}")

    def smooth(self, v: float, a: float) -> tuple[float, float]:
        """Applies the configured smoothing algorithm to the valence and arousal values."""
        if self.method == "rolling_buffer":
            if len(self.history_v) == self.window_size:
                self.v_sum -= self.history_v[0]
                self.a_sum -= self.history_a[0]

            self.history_v.append(v)
            self.history_a.append(a)
            self.v_sum += v
            self.a_sum += a
            smoothed_v = self.v_sum / len(self.history_v)
            smoothed_a = self.a_sum / len(self.history_a)
            return smoothed_v, smoothed_a

        elif self.method == "ema":
            if self.last_v is None:
                self.last_v, self.last_a = v, a
            else:
                self.last_v = self.alpha * v + (1 - self.alpha) * self.last_v
                self.last_a = self.alpha * a + (1 - self.alpha) * self.last_a
            return self.last_v, self.last_a

        elif self.method == "steps":
            if self.last_v is None or self.last_a is None:
                self.last_v, self.last_a = v, a

            else:
                # Only allow adjacent moves
                if abs(v - self.last_v) > self.max_step_delta:
                    v = self.last_v + self.max_step_delta * (
                        1 if v > self.last_v else -1
                    )
                if abs(a - self.last_a) > self.max_step_delta:
                    a = self.last_a + self.max_step_delta * (
                        1 if a > self.last_a else -1
                    )

                self.last_v, self.last_a = v, a
            return float(self.last_v), float(self.last_a)

        # No smoothing fallback
        return v, a
