import time
import pyarrow.feather as feather
import queue
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# fmt: off
FILE_PATH = "emotion_data/Dreamer/dreamer_bandpower_frames.feather"
YAML_PATH = "config.yml"
POW_COLUMNS = [
        "AF3/theta", "AF3/alpha", "AF3/betaL", "AF3/betaH", "AF3/gamma",
        "F7/theta",  "F7/alpha",  "F7/betaL",  "F7/betaH",  "F7/gamma",
        "F3/theta",  "F3/alpha",  "F3/betaL",  "F3/betaH",  "F3/gamma",
        "FC5/theta", "FC5/alpha", "FC5/betaL", "FC5/betaH", "FC5/gamma",
        "T7/theta",  "T7/alpha",  "T7/betaL",  "T7/betaH",  "T7/gamma",
        "P7/theta",  "P7/alpha",  "P7/betaL",  "P7/betaH",  "P7/gamma",
        "O1/theta",  "O1/alpha",  "O1/betaL",  "O1/betaH",  "O1/gamma",
        "O2/theta",  "O2/alpha",  "O2/betaL",  "O2/betaH",  "O2/gamma",
        "P8/theta",  "P8/alpha",  "P8/betaL",  "P8/betaH",  "P8/gamma",
        "T8/theta",  "T8/alpha",  "T8/betaL",  "T8/betaH",  "T8/gamma",
        "FC6/theta", "FC6/alpha", "FC6/betaL", "FC6/betaH", "FC6/gamma",
        "F4/theta",  "F4/alpha",  "F4/betaL",  "F4/betaH",  "F4/gamma",
        "F8/theta",  "F8/alpha",  "F8/betaL",  "F8/betaH",  "F8/gamma",
        "AF4/theta", "AF4/alpha", "AF4/betaL", "AF4/betaH", "AF4/gamma"]
EMOTIV_POW_FREC = 8  # Hz

# fmt: on


class EmotionSimulator:
    file_path = FILE_PATH
    emotion_range = None
    emot_states_area = {}
    sequences = {}
    sequence = []
    sub_id = None
    data_frec = EMOTIV_POW_FREC  # Hz
    data = None
    out_queue = None
    emotiv_columns = POW_COLUMNS
    emot_states = list(emot_states_area.keys())
    pow_by_state = {}
    pow_read = {}

    def __init__(self, queue: queue.Queue):
        self.out_queue = queue
        self.load_yml_config()
        self.load_pow_data()
        self.add_emot_states()
        self.get_emotion_pow()
        print("Emotion Simulator initialized.")

    def load_yml_config(self, path: str | Path = YAML_PATH) -> dict | list:
        """Parse a YAML file and return the Python object it represents."""
        path = Path(path).expanduser()
        try:
            yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
            emot_states = yml_data["emotional_states_areas"]
            for state in emot_states:
                id = state["id"]
                label = state["label"]
                emot_range = state["range"]
                va = (emot_range["valence_min"], emot_range["valence_max"])
                ar = (emot_range["arousal_min"], emot_range["arousal_max"])
                self.emot_states_area[id] = {"label": label, "va": va, "ar": ar}
            self.sequences = yml_data["sequences"]
            self.sub_id = yml_data["sub_id"]
            self.emotion_range = yml_data["emotion_range"]
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML in {path}: {e}")

    def load_pow_data(self):
        self.data = feather.read_feather(self.file_path)

        # normalize valence and arousal to self.emotion_range
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

    def get_emotion_pow(self):
        for emot in self.emot_states:
            s_state = self.data["state"] == emot
            s_sub = self.data["subject_id"] == self.sub_id
            mask = s_state & s_sub
            if self.sub_id == -1:
                mask = s_state
            data = self.data[mask].reindex(columns=self.emotiv_columns)
            self.pow_by_state[emot] = data
            self.pow_read[emot] = 0
            print(f"State '{emot}': {data.shape[0]} rows loaded.")

    def zero_pow_read(self):
        for emot in self.emot_states:
            self.pow_read[emot] = 0

    def imput_msg(self):
        msg = "Select sequence to simulate:\n"
        i = 0
        for key in self.sequences.keys():
            msg += f"{i} - {key}\n"
            i += 1
        msg += "or 'q' to quit:\n"
        return msg

    def decode_msg_num(self, msg_num):
        i = 0
        for key in self.sequences.keys():
            if i == msg_num:
                return key
            i += 1

    def main_loop(self):
        while True:
            msg = self.imput_msg()
            state = input(msg)
            if state == "q":
                break
            msg = self.decode_msg_num(int(state))
            if msg in self.sequences.keys():
                self.sequence = self.sequences[msg]
                print(f"Selected sequence: {msg}")
                self.output_loop()
                self.zero_pow_read()

    def output_loop(self):
        # print(self.data.columns)
        # print(self.data.head())
        # print(self.data["state"].value_counts())

        for idx in range(len(self.sequence)):
            state, duration = self.sequence[idx]
            t_start = time.time()
            t_cur = t_start
            while t_cur - t_start < duration:
                # row = states[idx].sample(n=1).iloc[0] # Get a random row
                row = self.pow_by_state[state].iloc[
                    self.pow_read[state]
                ]  # Get row in sequence

                # print(row.values.tolist())
                self.update_queue(row.values.tolist())
                time.sleep(1 / self.data_frec)
                self.pow_read[state] += 1
                if self.pow_read[state] >= self.pow_by_state[state].shape[0]:
                    self.pow_read[state] = 0
                t_cur = time.time()

    def update_queue(self, pow) -> None:
        # while not self.out_queue.empty():
        #     time.sleep(0.1)

        self.out_queue.put(pow)
        print(f"new data put in queue, mean: {sum(pow) / len(pow):.4f}")

        # while not self.out_queue.empty():
        #     time.sleep(0.1)

    def plot_emotion_distribution(self):
        plt.figure(figsize=(10, 6))

        # Define colors for each state
        colors = plt.cm.tab10.colors
        state_colors = {
            state: colors[i % len(colors)]
            for i, state in enumerate(self.emot_states_area.keys())
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
        for state, ranges in self.emot_states_area.items():
            state_data = self.data[self.data["state"] == state]
            plt.scatter(
                state_data["valence"],
                state_data["arousal"],
                # color=state_colors[state],
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
        plt.xlim(self.emotion_range[0] - margin, self.emotion_range[1] + margin)
        plt.ylim(self.emotion_range[0] - margin, self.emotion_range[1] + margin)
        plt.show()


def main():
    out = queue.Queue()
    simulator = EmotionSimulator(out)
    simulator.plot_emotion_distribution()
    simulator.main_loop()


if __name__ == "__main__":
    main()
