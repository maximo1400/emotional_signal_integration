import queue
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


class EmotionSimulator:
    file_path = None
    emotion_range = None
    emot_states_area = {}
    sequences = {}
    sequence = []
    sub_id = None
    data_frec = None
    data: pandas.DataFrame | None = None
    out_queue: queue.Queue | None = None
    output_df: pandas.DataFrame | None = None
    output_rows = []
    emotiv_columns = []
    emot_states = []
    pow_by_state = {}
    pow_read = {}
    transition_duration = 0

    def __init__(self, queue: queue.Queue, starting_timestamp: float):
        self.starting_timestamp = starting_timestamp
        self.out_queue = queue
        self.load_yml_config()
        self.load_pow_data()
        self.add_emot_states()
        self.get_emotion_pow(verbose=self.verbose)
        print("Running L1 as Emotion Simulator")

    def load_yml_config(self):
        """Parse config file and return the Python object it represents."""
        config = get_config([
            "feather_file_path",
            "POW_COLUMNS",
            "emotional_states_areas",
            "sequences",
            "sub_id",
            "emotion_range",
            "transition_duration",
            "emotiv_pow_frec",
            "verbose",
        ])

        self.file_path = config["feather_file_path"]
        self.emotiv_columns = config["POW_COLUMNS"]
        self.data_frec = config["emotiv_pow_frec"]
        self.verbose = config["verbose"]

        emot_states = config["emotional_states_areas"]
        for state in emot_states:
            id = state["id"]
            label = state["label"]
            emot_range = state["range"]
            va = (emot_range["valence_min"], emot_range["valence_max"])
            ar = (emot_range["arousal_min"], emot_range["arousal_max"])
            self.emot_states_area[id] = {"label": label, "va": va, "ar": ar}

        self.sequences = config["sequences"]
        self.sub_id = config["sub_id"]
        self.emotion_range = config["emotion_range"]
        self.transition_duration = config["transition_duration"]

    def load_pow_data(self):
        "Loads the power data from the feather file, normalizes the valence and arousal values to the defined emotion range and stores it in self.data."
        self.data = feather.read_feather(self.file_path)

        # normalize valence and arousal to self.emotion_range
        if self.emotion_range is None:
            raise ValueError("emotion_range is not set")
        emot_min, emot_max = self.emotion_range

        val = self.data["valence"]
        val_min = val.min()
        val_max = val.max()
        self.data["valence"] = ((val - val_min) / (val_max - val_min)) * (
            emot_max - emot_min
        ) + emot_min

        ar = self.data["arousal"]
        ar_min = ar.min()
        ar_max = ar.max()
        self.data["arousal"] = ((ar - ar_min) / (ar_max - ar_min)) * (
            emot_max - emot_min
        ) + emot_min

    def add_emot_states(self):
        """Assigns an emotional state to each row in self.data based on the valence and arousal values
        and the defined emotional state areas."""
        if self.data is None:
            raise ValueError("Data not loaded")
        self.data["state"] = "NA"  # Default state
        for state, ranges in self.emot_states_area.items():
            va_min, va_max = ranges["va"]
            ar_min, ar_max = ranges["ar"]

            condition = (
                (self.data["state"] == "NA")  # Only unassigned rows
                & (self.data["valence"] >= va_min)
                & (self.data["valence"] <= va_max)
                & (self.data["arousal"] >= ar_min)
                & (self.data["arousal"] <= ar_max)
            )
            self.data.loc[condition, "state"] = state
            self.emot_states.append(state)
            # print(self.data[condition].shape[0], "rows assigned to state", state)

    def get_emotion_pow(self, verbose=False):
        "Organizes the power data by emotional state and initializes the read counters."
        if self.data is None:
            raise ValueError("Data not loaded")
        for emot in self.emot_states:
            emot_state_mask = self.data["state"] == emot
            subject_mask = self.data["subject_id"] == self.sub_id
            mask = emot_state_mask & subject_mask
            if self.sub_id == -1:  # If sub_id is -1, use all subjects data
                mask = emot_state_mask
            data = self.data[mask].copy()
            # Keep valence and arousal along with power columns
            cols_to_keep = self.emotiv_columns + ["valence", "arousal"]
            data = data[cols_to_keep]
            self.pow_by_state[emot] = data
            self.pow_read[emot] = 0
            if verbose:
                print(f"State '{emot}': {data.shape[0]} rows loaded.")

    def zero_pow_read(self):
        """Resets the read count for each emotion state."""
        for emot in self.emot_states:
            self.pow_read[emot] = 0

    def imput_msg(self):
        "returns the input message to select the sequence to simulate"
        msg = "Select sequence to simulate:\n"
        i = 0
        for key in self.sequences.keys():
            msg += f"{i} - {key}\n"
            i += 1
        msg += "or 'q' to save and quit:\n"
        return msg

    def decode_msg_num(self, msg_num):
        "decodes the input number to the corresponding sequence key"
        i = 0
        for key in self.sequences.keys():
            if i == msg_num:
                return key
            i += 1

    def main_loop(self):
        """Main loop of the simulator, shows a menu to select the sequence to simulate and outputs
        the pow data to the queue according to the selected sequence."""
        while True:
            msg = self.imput_msg()
            state = input(msg)
            if state == "q":
                self.finalize_output_df()
                if self.out_queue is None:
                    raise ValueError("Queue not initialized")
                self.out_queue.put(None)  # Signal to any consumer that we're done
                break
            msg = self.decode_msg_num(int(state))
            if msg in self.sequences.keys():
                self.sequence = self.sequences[msg]
                print(f"Selected sequence: {msg}")
                self.output_loop()
                self.zero_pow_read()

    def output_loop(self):
        "outputs pow data to the queue according to self.sequence"
        # print(self.data.columns)
        # print(self.data.head())
        # print(self.data["state"].value_counts())

        for idx in range(len(self.sequence)):
            state, duration = self.sequence[idx]
            next_state = (
                self.sequence[idx + 1][0] if idx + 1 < len(self.sequence) else state
            )

            t_start = time.time()
            t_cur = t_start
            while t_cur - t_start < duration:
                # row = states[idx].sample(n=1).iloc[0] # Get a random row
                row = self.pow_by_state[state].iloc[
                    self.pow_read[state]
                ]  # Get row in sequence
                pow_data = row[self.emotiv_columns].values.tolist()
                valence = row["valence"]
                arousal = row["arousal"]
                smoothed = False

                # Check if we're in transition zone
                t_in_state = t_cur - t_start
                t_start_transition = duration - self.transition_duration
                if t_in_state > t_start_transition and idx + 1 < len(self.sequence):
                    # Blend with next state
                    next_row = self.pow_by_state[next_state].iloc[
                        self.pow_read[next_state]
                    ]
                    next_pow = next_row[self.emotiv_columns].values.tolist()

                    # Linear interpolation weight
                    blend = (t_in_state - t_start_transition) / self.transition_duration

                    pow_data = [
                        curr * (1 - blend) + next_val * blend
                        for curr, next_val in zip(pow_data, next_pow)
                    ]
                    smoothed = True

                t_cur = time.time()
                self.update_queue(pow_data, t_cur)
                self.update_output_rows(
                    pow_data, state, valence, arousal, smoothed, t_cur
                )
                self.pow_read[state] += 1
                if self.pow_read[state] >= self.pow_by_state[state].shape[0]:
                    self.pow_read[state] = 0

                if self.data_frec is None:
                    raise ValueError("data_frec is not set")
                time.sleep(1 / self.data_frec)

    def update_queue(
        self,
        pow,
        t_cur,
    ) -> None:
        "adds the new pow data to the output queue"
        if self.out_queue is None:
            raise ValueError("Queue not initialized")
        self.out_queue.put({
            "pow": pow,
            "timestamp": t_cur,
        })
        print(f"new data put in L1 queue, mean: {sum(pow) / len(pow):.4f}")

    def update_output_rows(self, pow, emot_state, valence, arousal, smoothed, t_cur):
        "adds a new row to the output with the pow data, emotional state, valence, arousal, smoothed flag and timestamp"
        new_row = pow + [
            valence,
            arousal,
            emot_state,
            smoothed,
            t_cur,
        ]
        self.output_rows.append(new_row)

    def finalize_output_df(self):
        "finalizes the output dataframe by converting the output rows to a dataframe with the correct columns"
        columns = self.emotiv_columns + [
            "valence",
            "arousal",
            "emot_state",
            "smoothed",
            "timestamp",
        ]
        self.output_df = pandas.DataFrame(self.output_rows, columns=columns)

    def plot_emotion_distribution(self):
        if self.data is None:
            raise ValueError("Data not loaded")
        plt.figure(figsize=(10, 6))

        # Define colors for each state
        cmap = plt.get_cmap("tab10")
        state_colors = {
            state: cmap(i % 10) for i, state in enumerate(self.emot_states_area.keys())
        }

        for state, ranges in self.emot_states_area.items():
            va_min, va_max = ranges["va"]
            ar_min, ar_max = ranges["ar"]

            # Plot the filled emotional state area
            rectangle = plt.Rectangle(
                (va_min, ar_min),
                va_max - va_min,
                ar_max - ar_min,
                linewidth=2,
                edgecolor="black",
                facecolor=state_colors[state],
                alpha=0.3,
                label=state,
            )
            plt.gca().add_patch(rectangle)

            # Add label in center of rectangle
            plt.text(
                (va_min + va_max) / 2,
                (ar_min + ar_max) / 2,
                state,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Plot scatter points on top
        # Always plot all data to show full distribution across all subjects
        plot_data = self.data

        plt.scatter(
            plot_data["valence"],
            plot_data["arousal"],
            color="red",
            alpha=0.6,
            s=20,
        )

        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        plt.title("Valence vs Arousal with Emotional State Areas")
        # plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        margin = 0.1
        if self.emotion_range is None:
            raise ValueError("emotion_range is not set")
        plt.xlim(self.emotion_range[0] - margin, self.emotion_range[1] + margin)
        plt.ylim(self.emotion_range[0] - margin, self.emotion_range[1] + margin)
        plt.show()
