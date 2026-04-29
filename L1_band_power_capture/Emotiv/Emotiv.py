import time
from . import cortex
from L1_band_power_capture.Emotiv.cortex import Cortex
import pandas as pd
import queue


class Subcribe:
    """
    A class to subscribe data stream.

    Attributes
    ----------
    c : Cortex
        Cortex communicate with Emotiv Cortex Service

    Methods
    -------
    start():
        start data subscribing process.
    sub(streams):
        To subscribe to one or more data streams.
    on_new_data_labels(*args, **kwargs):
        To handle data labels of subscribed data
    on_new_eeg_data(*args, **kwargs):
        To handle eeg data emitted from Cortex
    on_new_mot_data(*args, **kwargs):
        To handle motion data emitted from Cortex
    on_new_dev_data(*args, **kwargs):
        To handle device information data emitted from Cortex
    on_new_met_data(*args, **kwargs):
        To handle performance metrics data emitted from Cortex
    on_new_pow_data(*args, **kwargs):
        To handle band power data emitted from Cortex
    """

    def __init__(
        self,
        app_client_id,
        app_client_secret,
        verbose=True,
        **kwargs,
    ):
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=False, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(query_profile_done=self.on_query_profile_done)
        self.c.bind(load_unload_profile_done=self.on_load_unload_profile_done)
        self.c.bind(save_profile_done=self.on_save_profile_done)
        self.c.bind(new_eeg_data=self.on_new_eeg_data)
        self.c.bind(new_mot_data=self.on_new_mot_data)
        self.c.bind(new_dev_data=self.on_new_dev_data)
        self.c.bind(new_met_data=self.on_new_met_data)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(new_com_data=self.on_new_com_data)
        self.c.bind(new_fe_data=self.on_new_fe_data)
        self.c.bind(inform_error=self.on_inform_error)
        self.data = {}
        self.verbose = verbose

    def start(self, profile_name, streams, queue: queue, headsetId=""):
        """
        To start training process as below workflow
        (1) check access right -> authorize -> connect headset->create session
        (2) query profile -> get current profile -> load/create profile -> subscribe sys
        (3) start and accept MC action training in the action list one by one
        Parameters
        ----------
        profile_name : string, required
            name of profile
        actions : list, required
            list of actions which will be trained
        headsetId: string , optional
             id of wanted headet which you want to work with it.
             If the headsetId is empty, the first headset in list will be set as wanted headset
        Returns
        -------
        Nonenew datasself.headset_id
        """
        if profile_name == "":
            raise ValueError("Empty profile_name. The profile_name cannot be empty.")

        self.profile_name = profile_name
        self.streams = streams
        self.queue = queue
        self.c.set_wanted_profile(profile_name)

        if headsetId != "":
            self.c.set_wanted_headset(headsetId)

        self.c.open()

    def subscribe_data(self, streams):
        """
        To subscribe to one or more data streams
        'com': Mental command
        'fac' : Facial expression
        'sys': training event

        Parameters
        ----------
        streams : list, required
            list of streams. For example, ['sys']

        Returns
        -------
        None
        """
        self.c.sub_request(streams)

    def load_profile(self, profile_name):
        """
        To load an existed profile or create new profile for training

        Parameters
        ----------
        profile_name : str, required
            profile name

        Returns
        -------
        None
        """

        status = "load"
        self.c.setup_profile(profile_name, status)

    def unload_profile(self, profile_name):
        """
        To unload an existed profile or create new profile for training

        Parameters
        ----------
        profile_name : str, required
            profile name

        Returns
        -------
        None
        """
        self.c.setup_profile(profile_name, "unload")

    def save_profile(self, profile_name):
        """
        To save a profile

        Parameters
        ----------
        profile_name : str, required
            profile name

        Returns
        -------
        None
        """
        self.c.setup_profile(profile_name, "save")

    def get_active_action(self, profile_name):
        self.c.get_mental_command_active_action(profile_name)

    def get_command_brain_map(self, profile_name):
        self.c.get_mental_command_brain_map(profile_name)

    # callbacks functions
    def on_create_session_done(self, *args, **kwargs):
        print("on_create_session_done")
        self.c.query_profile()

    def on_query_profile_done(self, *args, **kwargs):
        print("on_query_profile_done")
        self.profile_lists = kwargs.get("data")
        if self.profile_name in self.profile_lists:
            # the profile is existed
            self.c.get_current_profile()
        else:
            # create profile
            self.c.setup_profile(self.profile_name, "create")

    def on_load_unload_profile_done(self, *args, **kwargs):
        is_loaded = kwargs.get("isLoaded")

        if is_loaded:
            # subscribe sys stream to receive Training Event
            # self.subscribe_data(["sys"])
            self.subscribe_data(self.streams)
        else:
            print("The profile " + self.profile_name + " is unloaded")
            self.profile_name = ""
            # close socket
            self.c.close()

    def on_save_profile_done(self, *args, **kwargs):
        print("Save profile " + self.profile_name + " successfully.")
        # You can test some advanced bci such as active actions, brain map, and training threshold. before unload profile
        self.unload_profile(self.profile_name)

    def on_new_data_labels(self, *args, **kwargs):
        data = kwargs.get("data")
        print("on_new_data_labels")
        # print(data)
        # Records data
        stream_name = data["streamName"]
        stream_labels = data["labels"]
        if stream_name in ["eeg", "com", "fac", "mot", "met", "pow", "sys", "dev"]:
            stream_labels += ["timestamp"]
        print("**New Dataset**")
        if stream_name != "eeg":
            self.data[stream_name] = pd.DataFrame(columns=stream_labels)
        else:
            # eeg data grows too fast to be stored in a dataframe
            self.data[stream_name] = []
            self.data["eeg_columns"] = pd.DataFrame(columns=stream_labels)
        if self.verbose:
            print("{} labels are : {}".format(stream_name, stream_labels))

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get("error_data")
        error_code = error_data["code"]
        error_message = error_data["message"]

        print(error_data)

        if error_code == cortex.ERR_PROFILE_ACCESS_DENIED:
            # disconnect headset for next use
            print(
                "Get error "
                + error_message
                + ". Disconnect headset to fix this issue for next use."
            )
            self.c.disconnect_headset()

    def unsub(self, streams):
        """
        To unsubscribe to one or more data streams
        'eeg': EEG
        'mot' : Motion
        'dev' : Device information
        'met' : Performance metric
        'pow' : Band power

        Parameters
        ----------
        streams : list, required
            list of streams. For example, ['eeg', 'mot']

        Returns
        -------
        None
        """
        self.c.unsub_request(streams)

    # On new data functions from export_sub.py

    def on_new_eeg_data(self, *args, **kwargs):
        """
        To handle eeg data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array eeg match the labels in the array labels return at on_new_data_labels
        For example:
           {'eeg': [99, 0, 4291.795, 4371.795, 4078.461, 4036.41, 4231.795, 0.0, 0], 'time': 1627457774.5166}
        """
        data = kwargs.get("data")
        timestamp = data["time"]
        eeg_data = data["eeg"] + [timestamp]
        # self.data["eeg"].loc[len(self.data["eeg"])] = eeg_data
        self.data["eeg"].append(eeg_data)
        if self.verbose:
            print("eeg data: {}".format(data))

    def on_new_mot_data(self, *args, **kwargs):
        """
        To handle motion data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array motion match the labels in the array labels return at on_new_data_labels
        For example: {'mot': [33, 0, 0.493859, 0.40625, 0.46875, -0.609375, 0.968765, 0.187503, -0.250004, -76.563667, -19.584995, 38.281834], 'time': 1627457508.2588}
        """
        data = kwargs.get("data")
        timestamp = data["time"]
        mot_data = data["mot"] + [timestamp]
        self.data["mot"].loc[len(self.data["mot"])] = mot_data
        if self.verbose:
            print("motion data: {}".format(data))

    def on_new_dev_data(self, *args, **kwargs):
        """
        To handle dev data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array dev match the labels in the array labels return at on_new_data_labels
        For example:  {'signal': 1.0, 'dev': [4, 4, 4, 4, 4, 100], 'batteryPercent': 80, 'time': 1627459265.4463}
        """
        data = kwargs.get("data")
        timestamp = data["time"]
        dev_data = data["dev"] + [timestamp]
        self.data["dev"].loc[len(self.data["dev"])] = dev_data
        if self.verbose:
            print("dev data: {}".format(data))

    def on_new_met_data(self, *args, **kwargs):
        """
        To handle performance metrics data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array met match the labels in the array labels return at on_new_data_labels
        For example: {'met': [True, 0.5, True, 0.5, 0.0, True, 0.5, True, 0.5, True, 0.5, True, 0.5], 'time': 1627459390.4229}
        """
        data = kwargs.get("data")
        timestamp = data["time"]
        met_data = data["met"] + [timestamp]
        self.data["met"].loc[len(self.data["met"])] = met_data
        if self.verbose:
            print("pm data: {}".format(data))

    def on_new_pow_data(self, *args, **kwargs):
        """
        To handle band power data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array pow match the labels in the array labels return at on_new_data_labels
        For example: {'pow': [5.251, 4.691, 3.195, 1.193, 0.282, 0.636, 0.929, 0.833, 0.347, 0.337, 7.863, 3.122, 2.243, 0.787, 0.496, 5.723, 2.87, 3.099, 0.91, 0.516, 5.783, 4.818, 2.393, 1.278, 0.213], 'time': 1627459390.1729}
        """
        data = kwargs.get("data")
        timestamp = data["time"]
        pow_data = data["pow"] + [timestamp]
        self.data["pow"].loc[len(self.data["pow"])] = pow_data
        self.queue.put(data)
        print("pow data put in queue: {}".format(pow_data))
        if self.verbose:
            print("pow data: {}".format(data))

    def on_new_com_data(self, *args, **kwargs):
        data = kwargs.get("data")
        com_data = list(data.values())
        self.data["com"].loc[len(self.data["com"])] = com_data
        if self.verbose:
            print("com data: {}".format(data))

    def on_new_fe_data(self, *args, **kwargs):
        data = kwargs.get("data")
        fe_data = list(data.values())
        self.data["fac"].loc[len(self.data["fac"])] = fe_data
        if self.verbose:
            print("fe data: {}".format(data))


# -----------------------------------------------------------
#
# GETTING STARTED
#   - Please reference to https://emotiv.gitbook.io/cortex-api/ first.
#   - Connect your headset with dongle or bluetooth. You can see the headset via Emotiv Launcher
#   - Please make sure the your_app_client_id and your_app_client_secret are set before starting running.
#   - The function on_create_session_done,  on_query_profile_done, on_load_unload_profile_done will help
#          handle create and load an profile automatically . So you should not modify them
#   - The functions on_new_data_labels(), on_new_sys_data() will help to control  action by action training.
#          You can modify these functions to control the training such as: reject an training, use advanced bci api.
# RESULT
#   - train mental command action
#
# -----------------------------------------------------------
