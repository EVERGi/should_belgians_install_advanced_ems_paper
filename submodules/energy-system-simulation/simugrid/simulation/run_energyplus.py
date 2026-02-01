"""
Code taken from the following repository:
https://github.com/mechyai/RL-EmsPy/tree/main
"""

import datetime
import numpy as np
import sys
import csv

from simugrid.simulation.environment import BaseListValue, Environment


def J_to_kWh(joules: float) -> float:
    """Converts joules to kilowatt hours."""
    return joules / 3_600_000


def parse_epw_weather(epw_file: str, keep_cols: list[str]) -> dict:

    with open(epw_file, "r") as csvdata:
        csvreader = csv.reader(csvdata)

        colnames = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "data_source_unct",
            "temp_air",
            "temp_dew",
            "relative_humidity",
            "atmospheric_pressure",
            "etr",
            "etrn",
            "ghi_infrared",
            "glob_hor_rad",
            "dir_norm_rad",
            "dif_hor_rad",
            "global_hor_illum",
            "direct_normal_illum",
            "diffuse_horizontal_illum",
            "zenith_luminance",
            "wind_direction",
            "wind_speed",
            "total_sky_cover",
            "opaque_sky_cover",
            "visibility",
            "ceiling_height",
            "present_weather_observation",
            "present_weather_codes",
            "precipitable_water",
            "aerosol_optical_depth",
            "snow_depth",
            "days_since_last_snowfall",
            "albedo",
            "liquid_precipitation_depth",
            "liquid_precipitation_quantity",
        ]
        epw_data = {colname: [] for colname in keep_cols}
        epw_data["datetime"] = []
        for row in csvreader:
            if len(row) != len(colnames):
                continue
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            hour = int(row[3]) - 1
            minute = int(row[4])

            timestep_dt = datetime.datetime(year, month, day, hour, minute)
            epw_data["datetime"].append(timestep_dt)
            for idx, colname in enumerate(colnames[5:]):
                if colname in keep_cols:
                    epw_data[colname].append(float(row[idx + 5]))

    return epw_data


def interpolate_epw(
    epw_data: dict, time_step: datetime.timedelta, interpollation="linear_post"
) -> dict:
    epw_data_interpolated = {key: [] for key in epw_data.keys()}

    original_timestep = epw_data["datetime"][1] - epw_data["datetime"][0]
    timestep_ratio = int(original_timestep / time_step)

    if timestep_ratio == 1:
        return epw_data

    for i, dt in enumerate(epw_data["datetime"]):
        for j in range(timestep_ratio):
            for col_name in epw_data.keys():

                if col_name == "datetime":
                    cur_dt = dt + j * time_step
                    new_value = cur_dt

                else:
                    cur_point = epw_data[col_name][i]
                    if i == len(epw_data[col_name]) - 1:
                        next_point = epw_data[col_name][i]
                    else:
                        next_point = epw_data[col_name][i + 1]
                    if interpollation == "linear_post":
                        new_value = cur_point + (next_point - cur_point) * (j + 1) / (
                            timestep_ratio + 1
                        )
                    # elif interpollation == "linear_pre":
                    # new_value = cur_point + (next_point - cur_point) * j / (
                    #    timestep_ratio
                    # )
                    elif interpollation == "constant":
                        new_value = cur_point

                epw_data_interpolated[col_name].append(new_value)

    for col_name in epw_data_interpolated.keys():
        if col_name == "datetime":
            continue
        if len(epw_data_interpolated[col_name]) != len(
            epw_data_interpolated["datetime"]
        ):
            raise ValueError(f"Length of {col_name} is not the same as datetime")

    return epw_data_interpolated


def add_epw_dataframe_to_environment(epw_data: dict, environment: Environment):
    microgrid = environment.microgrid
    datetime_column = epw_data["datetime"]
    for colname in epw_data.keys():
        if colname == "datetime":
            continue
        data_column = epw_data[colname]
        dt_and_values = []
        for i, value in enumerate(data_column):
            dt_and_values.append([datetime_column[i], value])

        csv_value = BaseListValue(microgrid, dt_and_values)
        value_name = f"epw_{colname}"
        environment.env_values[value_name] = csv_value


def add_epw_file_to_environment(
    epw_file, environment, keep_cols=["temp_air", "dir_norm_rad"]
):

    epw_data = parse_epw_weather(epw_file, keep_cols)
    epw_data_interpolated = interpolate_epw(epw_data, environment.microgrid.time_step)
    add_epw_dataframe_to_environment(epw_data_interpolated, environment)


class EmsPy:
    """A meta-class wrapper to the EnergyPlus Python API to simplify/constrain usage for RL-algorithm purposes."""

    available_weather_metrics = [
        "sun_is_up",
        "is_raining",
        "is_snowing",
        "albedo",
        "beam_solar",
        "diffuse_solar",
        "horizontal_ir",
        "liquid_precipitation",
        "outdoor_barometric_pressure",
        "outdoor_dew_point",
        "outdoor_dry_bulb",
        "outdoor_relative_humidity",
        "sky_temperature",
        "wind_direction",
        "wind_speed",
    ]

    available_timing_metrics = [
        "t_actual_date_times",
        "t_actual_times",
        "t_current_times",
        "t_years",
        "t_months",
        "t_days",
        "t_hours",
        "t_minutes",
        "t_datetimes",
        "timesteps_zone",
        "timesteps_zone_num",
        "callbacks",
    ]

    available_calling_points = [
        # Sim Star
        "callback_after_component_get_input",  # 0,   1.a)   @ env period, once, only autosize control
        "callback_end_zone_sizing",  # 1,   1.b)   @ env period, once
        "callback_end_system_sizing",  # 2,   1.c)   @ env period, once
        "callback_unitary_system_sizing",  # 3,   1.c.1) @ env period,
        "callback_begin_new_environment",  # 4,   2)     @ start env, 1
        "callback_after_new_environment_warmup_complete",  # 5,   3.a)   @ evn post warmup, 1
        # Beginning of Zone Timestep
        "callback_begin_zone_timestep_before_set_current_weather",  # 6,   3.b)   @ zone ts start
        "callback_begin_zone_timestep_before_init_heat_balance",  # 7,   3.c)   @ zone ts
        "callback_begin_zone_timestep_after_init_heat_balance",  # 8,   3.d)   @ zone ts
        # Zone Load "Predictor" / "Corrector
        "callback_begin_system_timestep_before_predictor",
        # 9*,  ?.?)   # TODO zone or sys ts, discrepancy in API and EMS documentation (TEST)
        "callback_after_predictor_before_hvac_managers",  # 10, 4.a)  @ zone ts, predictor EMS overwrite by managers
        "callback_after_predictor_after_hvac_managers",  # 11, 4.b)  @ zone ts, EMS overwrite managers
        # HVAC Iteration Loop (System Timestep)
        "callback_inside_system_iteration_loop",  # 12,  5.a) @ sys start, HVAC loop
        "callback_end_system_timestep_before_hvac_reporting",  # 13,  5.b) @ sys ts, HVAC loop
        "callback_end_system_timestep_after_hvac_reporting",  # 14,  5.c) @ sys ts, HVAC loop
        # End of Zone Timestep
        "callback_end_zone_timestep_before_zone_reporting",  # 15,  6.a) @ zone ts final
        "callback_end_zone_timestep_after_zone_reporting",  # 16,  6.b) @ zone ts end
        # Misc.
        "_callback_register_external_hvac_manager",  # 17,  (placeholder, not well supported as of 9.5)
        "callback_message",  # 18,  @ stdout, misc.
        "callback_progress",  # 19,  @ end of day, misc.
    ]

    def __init__(
        self,
        ep_path: str,
        ep_idf_to_run: str,
        timesteps: int,
        tc_var: dict,
        tc_intvar: dict,
        tc_meter: dict,
        tc_actuator: dict,
        tc_weather: dict,
    ):
        """
        Establish connection to EnergyPlusAPI and initializes desired EMS sensors, actuators, and weather data.

        This instantiation will implement the meta-class functionality - various handle and data list attributes will
        be created based on the user's input of Table of Contents (ToCs, or 'tc') of desired EMS sensors, actuators,
        and weather data. Understanding what sensors and actuators are available, and how they are labeled, requires a
        reasonably good understanding of your EnergyPlus model and its .idf file, as well as the .edd and .rdd output
        files. Other functionality of the EnergyPlusAPI, not provided by this simplified class, may also be accessed
        directly through the .pyapi pointer.

        :param ep_path: absolute path to EnergyPlus download directory in user's file system
        :param ep_idf_to_run: absolute/relative path to EnergyPlus building energy model to be simulated, .idf file
        :param timesteps: number of timesteps per hour set in EnergyPlus model .idf file
        :param tc_var: ToC dict of desired output Variables, with each EMS object provided as
        'user_var_name': ['variable_name', 'variable_key'] within the dict
        :param tc_intvar: ToC dict of desired Internal Variables (static), with each object provided as
        'user_var_name': ['variable_type', 'variable_key'] within the dict
        :param tc_meter: ToC dict of desired Meters, with each object provided as
        'user_var_name': 'meter_name' within the dict
        :param tc_actuator: ToC dict of desired EMS Actuators, with each object provided as
        'user_var_name': ['component_type', 'control_type', 'actuator_key'] within the dict
        :param tc_weather: ToC dict of desired weather types, with each object provided as
        'user_var_name': 'weather_metric' within the dict - see list of available weather metrics. Any such weather
         metric can also be called directly for Today / Tomorrow for a given hour and timestep, if desired.
        """

        self.ep_path = ep_path
        sys.path.insert(0, ep_path)  # set path to E+
        import pyenergyplus.api  # type: ignore
        from pyenergyplus.api import EnergyPlusAPI  # type: ignore

        self.pyapi = pyenergyplus.api
        self.api = EnergyPlusAPI()  # instantiation of Python EMS API

        # instance important
        self.state = self._new_state()
        self.idf_file = ep_idf_to_run  # E+ idf file to simulation

        # Table of Contents for EMS sensors, actuators. & weather
        self.tc_var = tc_var
        self.tc_intvar = tc_intvar
        self.tc_meter = tc_meter
        self.tc_actuator = tc_actuator
        self.tc_weather = tc_weather

        # dataframe elements
        self.df_count = 0
        self.df_custom_dict = (
            {}
        )  # key = custom_dict_name, val = ([ems_data_list], 'calling_point', update freq)
        self.df_var = None
        self.df_intvar = None
        self.df_meter = None
        self.df_actuator = None
        self.df_weather = None
        self.custom_dataframes_initialized = False

        self.default_dfs_reset = False  # trigger to reinitialize dfs, #TODO how to track over n consecutive simulations

        # summary dicts and lists
        self.ems_names_master_list = self.available_timing_metrics[
            :
        ]  # keeps track of all user & default EMS var names
        self.ems_type_dict = (
            {}
        )  # keep track of EMS metric names and associated EMS type, quick lookup
        self.ems_num_dict = (
            {}
        )  # keep track of EMS categories and num of vars for each tracked
        self.ems_current_data_dict = (
            {}
        )  # collection of all ems metrics (keys) and their current values (val)
        self.calling_point_callback_dict = (
            {}
        )  # links cp to callback fxn & its needed args

        # create attributes of sensor and actuator .idf handles and data arrays
        self._init_ems_handles_and_data()  # creates ems_handle = int & ems_data = [] attributes, and variable counts
        self.got_ems_handles = False
        self.static_vars_obtained = False  # static (internal) variables, gather once
        # create attributes for weather
        self._init_weather_data()  # creates weather_data = [] attribute, useful for present/prior weather data tracking

        # timing data
        self.t_actual_date_times = []
        self.t_actual_times = []
        self.t_current_times = []
        self.t_years = []
        self.t_months = []
        self.t_days = []
        self.t_hours = []
        self.t_minutes = []
        self.t_datetimes = []
        self.t_holiday_index = []
        # timestep
        self.timestep_input = timesteps
        self.timesteps_zone_num = []
        self.timestep_zone_num_current = 0  # fluctuate from 1 to # of timesteps/hour
        self.timestep_total_count = 0  # cnt for entire simulation
        self.timestep_per_hour = None  # sim timesteps per hour, initialized later
        self.timestep_period = (
            None  # minute duration of each timestep of simulation, initialized later
        )
        self.timestep_params_initialized = False

        # callback data
        self.callback_calling_points = []
        self.callbacks_count = []
        self.callback_current_count = 0

        # reward data  # TODO does this have to be handled by this class?
        self.rewards_created = False
        self.rewards_multi = False
        self.rewards = []
        self.reward_current = None
        self.rewards_cnt = None

        # simulation data
        self._actuators_used_set = (
            set()
        )  # keep track of what EMS actuators are actually actuated
        self.simulation_success = 1  # 1 fail, 0 success

        # print("\n*NOTE: Simulation emspy class and instance created!")

    def _init_ems_handles_and_data(self):
        """
        Creates and initializes the necessary instance attributes for all EMS sensors/actuators given by the user.

        This will initialize data list and EMS handle attributes to the proper Null value for each EMS variable,
        internal variable, meter, and actuator as outlined by the user in their respective EMS Table of Content(s).
        All of these attributes need to be initialized for later use, using the 'variable name' of the object in the
        first element of each ToC element. 'handle_' and 'data_' will be prefixed to the given name to further specify
        the created attribute. 'setpoint_' will prefix actuators to track their user input setpoints.

        This will also update the EMS dictionary which tracks which EMS variable types are in use and how many for each
        category. This dictionary attribute is used elsewhere for quick data fetching.
        """
        # set attribute handle names and data arrays given by user to None
        ems_types = ["var", "intvar", "meter", "actuator"]
        for ems_type in ems_types:
            ems_tc = getattr(self, "tc_" + ems_type)
            if ems_tc is not None and ems_tc:  # catch 'None' and '{}' input for TC:
                for ems_name in ems_tc:  # iterate through EMS key names
                    if ems_name in self.ems_names_master_list:  # check duplicate input
                        raise ValueError(
                            f"ERROR: EMS metric user-defined names must be unique, "
                            f"{ems_name}({self.ems_type_dict[ems_name]}) != {ems_name}({ems_type})"
                        )
                    setattr(
                        self, "handle_" + ems_type + "_" + ems_name, None
                    )  # real handle found at runtime
                    setattr(
                        self, "data_" + ems_type + "_" + ems_name, []
                    )  # init as empty list
                    if ems_type == "actuator":  # handle associated actuator setpoints
                        setpoint_name = "setpoint_" + ems_name
                        setattr(self, "data_" + setpoint_name, [])
                        self.ems_type_dict[setpoint_name] = "setpoint"
                        self.ems_names_master_list.append(setpoint_name)
                    self.ems_type_dict[ems_name] = ems_type
                    self.ems_names_master_list.append(
                        ems_name
                    )  # all ems metrics collected
                self.ems_num_dict[ems_type] = len(
                    ems_tc
                )  # num of metrics per ems category
                self.df_count += 1  # 1 default df per ems_type
        # automatically handle all available timing data dict type
        for t in self.available_timing_metrics:
            self.ems_type_dict[t] = "time"

    def _init_weather_data(self):
        """Creates and initializes the necessary instance attributes given by the user for present weather metrics."""

        if (
            self.tc_weather is not None and self.tc_weather
        ):  # catch 'None' and '{}' input for Weather TC
            # verify provided weather ToC is accurate/acceptable
            for weather_name, weather_metric in self.tc_weather.items():
                if weather_metric not in EmsPy.available_weather_metrics:
                    raise Exception(
                        f"ERROR: [{weather_metric}] weather metric is misspelled or not provided by"
                        f" EnergyPlusAPI."
                    )
                if weather_name in self.ems_names_master_list:
                    raise ValueError(
                        f"ERROR: EMS metric user-defined names must be unique, "
                        f"{weather_name}({self.ems_type_dict[weather_name]}) != {weather_name}(weather)"
                    )
                setattr(self, "data_weather_" + weather_name, [])
                self.ems_names_master_list.append(weather_name)
                self.ems_type_dict[weather_name] = "weather"
            self.ems_num_dict["weather"] = len(self.tc_weather)
            self.df_count += 1

    def _init_timestep(self) -> int:
        """This function is used to fetch the timestep input from the IDF model & verify with user input."""

        # returns fractional hour, convert to timestep/hr
        try:
            # TODO determine robustness of the api.exchange function - IT IS NOT (20 defaults to 60 etc)
            timestep = int(1 // self.api.exchange.zone_time_step(self.state))
            available_timesteps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
            if timestep in available_timesteps:
                if timestep != self.timestep_input:
                    # TODO add traceback
                    raise SystemExit(
                        f"User input timestep [{self.timestep_input}] must equal the model timestep "
                        f"of [{timestep}]. Please check your IDF.\nAvailable timesteps are "
                        f"{available_timesteps}"
                    )
                self.timestep_period = 60 // timestep
                self.timestep_per_hour = timestep
                # print(
                #    f"\n*NOTE: Your simulation timestep period is {self.timestep_period} minutes @ {timestep}"
                #    f" timestep(s) an hour.\n"
                # )
                self.timestep_params_initialized = True
                return timestep
        except ZeroDivisionError:
            pass

    def _init_reward(self, reward):
        """This sets the reward attributes to the needs set by user."""

        # attribute creation
        if not self.rewards_created:  # first iteration, do once
            try:  # multi obj rewards
                self.rewards_cnt = len(reward)
                self.rewards_multi = True
                self.rewards = [[] for _ in range(self.rewards_cnt)]
            except TypeError:  # catch single reward, no length
                self.rewards_cnt = 1
                self.rewards = []
            self.rewards_created = True
            self.reward_current = [0] * self.rewards_cnt

    def _set_ems_handles(self):
        """Gets and reassigns the gathered sensor/actuators handles to their according _handle instance attribute."""

        ems_types = ["var", "intvar", "meter", "actuator"]
        for ems_type in ems_types:
            ems_tc = getattr(self, "tc_" + ems_type)
            if ems_tc is not None:
                for name, handle_inputs in ems_tc.items():
                    setattr(
                        self,
                        "handle_" + ems_type + "_" + name,
                        self._get_handle(ems_type, handle_inputs),
                    )
        # print("\n*NOTE: Got all EMS handles.\n")

    def _get_handle(self, ems_type: str, ems_obj_details):
        """
        Returns the EMS object handle to be used as its ID for calling functions on it in the running simulation.

        :param ems_type: The EMS object type (variable, internal variable, meter, actuator)
        :param ems_obj_details: The specific object details provided by the user to attain the handle
        """
        state = self.state
        datax = self.api.exchange
        try:
            handle = ""
            if ems_type == "var":
                handle = datax.get_variable_handle(
                    state, ems_obj_details[0], ems_obj_details[1]  # var name
                )  # var key
            elif ems_type == "intvar":
                handle = datax.get_internal_variable_handle(
                    state, ems_obj_details[0], ems_obj_details[1]  # int var name
                )  # int var key
            elif ems_type == "meter":
                handle = datax.get_meter_handle(state, ems_obj_details)  # meter name
            elif ems_type == "actuator":
                handle = datax.get_actuator_handle(
                    state,
                    ems_obj_details[0],  # component type
                    ems_obj_details[1],  # control type
                    ems_obj_details[2],
                )  # actuator key
            # catch error handling by EMS E+
            if handle == -1:
                raise Exception(
                    f"ERROR: [{str(ems_obj_details)}]: The EMS sensor/actuator handle could not be "
                    "found. Please consult the .idf and/or your ToC for accuracy"
                )
            else:
                return handle
        except IndexError:
            raise IndexError(
                f"ERROR: [{str(ems_obj_details)}]: This [{ems_type}] object does not have all the "
                f"required fields to get the EMS handle. Check the API documentation."
            )

    def _update_time(self):
        """Updates all time-keeping and simulation timestep attributes of running simulation."""

        # simplify repetition
        state = self.state
        datax = self.api.exchange

        # gather data
        year = datax.year(state)
        month = datax.month(state)
        day = datax.day_of_month(state)
        hour = datax.hour(state)
        minute = datax.minutes(state)
        timestep_zone_num = datax.zone_time_step_number(state)

        # set, append
        self.t_actual_date_times.append(datax.actual_date_time(state))
        self.t_actual_times.append(datax.actual_time(state))
        self.t_current_times.append(datax.current_time(state))
        self.t_years.append(year)
        self.t_months.append(month)
        self.t_days.append(day)
        self.t_hours.append(hour)
        self.t_minutes.append(minute)
        # holiday
        self.t_holiday_index.append(datax.holiday_index(state))  # 1 holiday, 0 no
        # timesteps
        self.timesteps_zone_num.append(timestep_zone_num)
        self.timestep_zone_num_current = timestep_zone_num

        # manage datetime tracking
        timedelta = datetime.timedelta()
        if hour == 24:
            hour = 23.0
            timedelta += datetime.timedelta(hours=1)
        if minute == 60:
            minute = 59
            timedelta += datetime.timedelta(minutes=1)
        # time keeping dataframe management
        # if minute == 1440:
        #    minute = 0

        dt = datetime.datetime(
            year=year, month=month, day=day, hour=hour, minute=minute
        )
        dt += timedelta
        self.t_datetimes.append(dt)
        self.ems_current_data_dict["Datetime"] = dt  # TODO not used

        # timesteps total
        if (
            len(self.t_datetimes) < 2
            or self.t_datetimes[-1] != self.t_datetimes[-2]
            or self.timesteps_zone_num[-1] != self.timesteps_zone_num[-2]
        ):
            # verify new timestep if current & previous timestep num and datetime are different
            self.timestep_total_count += 1

    def _update_ems_data_attributes(
        self, ems_type: str, ems_name: str, data_val: float
    ):
        """Helper function to update EMS attributes with current values."""

        getattr(self, "data_" + ems_type + "_" + ems_name).append(data_val)
        self.ems_current_data_dict[ems_name] = data_val

    def _update_ems_and_weather_vals(self, ems_metrics_list: list):
        """Fetches and updates given sensor/actuator/weather values to data lists/dicts from running simulation."""

        # TODO how to handle user-specified TIMING updates separate from state, right now they are joint
        # specific data exchange API function calls
        datax = self.api.exchange
        ems_datax_func = {
            "var": datax.get_variable_value,
            "intvar": datax.get_internal_variable_value,
            "meter": datax.get_meter_value,
            "actuator": datax.get_actuator_value,
        }

        for ems_name in ems_metrics_list:
            ems_type = self.ems_type_dict[ems_name]
            # SKIP time and setpoint updates, each have their OWN updates
            if ems_type == "time" or ems_type == "setpoint":
                continue
            if ems_type == "weather":
                data_i = self._get_weather(
                    [ems_name],
                    "today",
                    self.t_hours[-1],
                    self.timestep_zone_num_current,
                )
            elif ems_type == "intvar":  # internal(static) vars updated ONCE, separately
                if not self.static_vars_obtained:
                    data_i = ems_datax_func[ems_type](
                        self.state, getattr(self, "handle_" + ems_type + "_" + ems_name)
                    )
                    self.static_vars_obtained = True
            else:  # rest: var, meter, actuator
                # get data from E+ sim
                data_i = ems_datax_func[ems_type](
                    self.state, getattr(self, "handle_" + ems_type + "_" + ems_name)
                )

            # store data in obj attributes
            self._update_ems_data_attributes(ems_type, ems_name, data_i)

    def _update_reward(self, reward):
        """Updates attributes related to the reward. Works for single-obj(scalar) and multi-obj(vector) reward fxns."""

        # handle single vs multi reward together
        if self.rewards_multi:
            rewards = []  # init for multi
        else:
            reward = [reward]  # need to make single val iterable

        for reward_i in reward:
            if not np.isscalar(reward_i):
                raise TypeError(
                    f"ERROR: Reward returned from the observation function, [{reward_i}] must be of"
                    f" type float or int."
                )
            else:
                if self.rewards_multi:
                    rewards.append(reward_i)  # [[r11, r12, r13], [r21, r22, r23], ...]
                else:
                    rewards = reward_i  # single reward
        # reward data update
        self.rewards.append(rewards)
        self.reward_current = rewards

    def _get_weather(
        self, weather_metrics: list, when: str, hour: int, zone_ts: int
    ) -> list:
        """
        Gets desired weather metric data for a given hour and zone timestep, either for today or tomorrow in simulation.

        :param weather_metrics: the weather metrics to call from E+ API, only specific fields from ToC are granted
        :param when: the day in question, 'today' or 'tomorrow', relative to current simulation time
        :param hour: the hour of the day to call the weather value
        :param zone_ts: the zone timestep of the given hour to call the weather value
        :return: list of updated weather data in order of weather_metrics input list
        """
        # input error handling
        if not (when == "today" or when == "tomorrow"):
            raise Exception(
                "ERROR: Weather data must either be called from sometime today or tomorrow relative to"
                " current simulation timestep."
            )
        if hour > 23 or hour < 0:
            raise Exception(
                "ERROR: The hour of the day cannot exceed 23 or be less than 0"
            )
        try:
            if zone_ts > self.timestep_per_hour:
                raise Exception(
                    f"ERROR: The desired weather forecast timestep, [{zone_ts}] cannot exceed the subhourly"
                    f" simulation timestep frequency set for the model, [{self.timestep_per_hour}]."
                )
        except TypeError:
            # catch error ONLY during design days when timestep has not been initialized yet
            # TODO is this needed?
            pass

        # fetch weather
        weather_data = []
        for weather_name in weather_metrics:
            # input error handling
            if weather_name not in self.tc_weather:
                raise Exception(
                    f"ERROR: Invalid weather metric [{weather_name}] given. Please see your weather ToC for"
                    " available weather metrics."
                )

            weather_metric = self.tc_weather[weather_name]
            # sun weather type is unique to rest, doesn't follow consistent naming system
            if weather_metric != "sun_is_up":
                weather_data.append(
                    getattr(
                        self.api.exchange,
                        when + "_weather_" + weather_metric + "_at_time",
                    )(self.state, hour, zone_ts)
                )
            elif weather_metric == "sun_is_up":
                weather_data.append(self.api.exchange.sun_is_up(self.state))

        if len(weather_metrics) == 1:  # single metric
            return weather_data[0]
        else:
            return weather_data

    def _actuate(self, actuator_handle: str, actuator_val):
        """Sets value of a specific actuator in running simulation, or relinquishes control back to EnergyPlus."""

        # use None to relinquish control
        # TODO should I handle out-of-range actuator values??? (can this be managed with auto internal var lookup?)
        if actuator_val is None:
            self.api.exchange.reset_actuator(
                self.state, actuator_handle
            )  # return actuator control to EnergyPlus
        else:
            self.api.exchange.set_actuator_value(
                self.state, actuator_handle, actuator_val
            )

    def _actuate_from_list(self, calling_point: str, actuator_setpoint_dict: dict):
        """
        This iterates through list of actuator name and value setpoint pairs to be set in simulation.

        CAUTION: Actuation functions written by user must return an actuator_name(key)-value pair dictionary

        :param calling_point: only used for error output message to user
        :param actuator_setpoint_dict: dict of actuator name keys (str) & associated setpoint val. A setpoint of None
        returns control back to EnergyPlus from EMS
        """
        if (
            actuator_setpoint_dict is not None
        ):  # in case some 'actuation functions' does not actually act
            for actuator_name, actuator_setpoint in actuator_setpoint_dict.items():
                if (
                    actuator_name not in self.tc_actuator
                ):  # TODO only do this once, @ the proper place and point
                    raise Exception(
                        f"ERROR: Either this actuator [{actuator_name}] is not tracked, or misspelled."
                        f" Check your Actuator ToC."
                    )
                # actuate and update data tracking
                actuator_handle = getattr(self, "handle_actuator_" + actuator_name)
                self._actuate(actuator_handle, actuator_setpoint)
                self._actuators_used_set.add(
                    actuator_name
                )  # to keep track of what actuators from TC are actually used
                # update SETPOINT value of actuators
                getattr(self, "data_setpoint_" + actuator_name).append(
                    actuator_setpoint
                )
        else:
            pass
            # print(
            #    f"\n*NOTE: No actuators/values defined for actuation function at calling point [{calling_point}],"
            #    f" timestep [{self.timestep_zone_num_current}]\n"
            # )

    def _enclosing_callback(
        self,
        calling_point: str,
        observation_fxn,
        actuation_fxn,
        update_state: bool = False,
        update_observation_frequency: int = 1,
        update_actuation_frequency: int = 1,
        observation_function_kwargs: dict = None,
        actuation_function_kwargs: dict = None,
    ):
        """
        Decorates the main callback function to set the user-defined calling point and set timing and data params.

        :param calling_point: the calling point at which the callback function will be called during simulation runtime
        :param observation_fxn: the user defined observation function to be called at runtime calling point and desired
        timestep frequency
        :param actuation_fxn: the user defined actuation function to be called at runtime calling point and desired
        timestep frequency
        :param update_state: whether EMS and time/timestep data should be updated from simulation for this calling point
        :param update_observation_frequency: the number of zone timesteps per running the observation function
        :param update_actuation_frequency: the number of zone timesteps per updating the actuators from the actuation
        function
        :param observation_function_kwargs: a dictionary to be fixed input to your observation function as **kwargs.
        **kwargs must be in your passed observation function to use, otherwise leave as None
        :param actuation_function_kwargs: a dictionary to be fixed input to your actuation function as **kwargs.
        **kwargs must be in your passed actuation function to use, otherwise leave as None
        """

        def _callback_function(state_arg):
            """
            The callback function passed to the running EnergyPlus simulation, this commands the runtime interaction.

            :param state_arg: NOT USED by this class - passed to and used internally by EnergyPlus simulation
            """

            # CALLBACK INIT
            # get EMS handles ONCE
            if not self.got_ems_handles:
                # verify ems objects are ready for access, skip until
                if not self.api.exchange.api_data_fully_ready(state_arg):
                    return
                self._set_ems_handles()
                self.got_ems_handles = True
            # skip callback IF simulation in WARMUP
            if self.api.exchange.warmup_flag(state_arg):
                return
            # init Timestep params ONCE, after warmup and EMS handles
            if not self.timestep_params_initialized:
                self._init_timestep()

            # HANDLE SYSTEM TIMESTEP ITERATIONS
            # get current timestep via API for update frequency
            self.timestep_zone_num_current = self.api.exchange.zone_time_step_number(
                state_arg
            )
            current_timestep = (
                self.timestep_zone_num_current
            )  # preserve for callback # TODO implement

            # TODO verify this is proper way to prevent sub-timestep callbacks, make separate function
            # FAIL with multiple CPs since they share timesteps
            # catch and skip sub-timestep callbacks, when the timestep num is the same as before
            try:
                if self.timesteps_zone_num[-1] == self.timestep_zone_num_current:
                    # verify with (timestep/hr) * (24 hrs) * (# of days of sim) == data/df length
                    # print('-- Sub-Timestep Callback --')
                    # return  # skip callback
                    pass
            except IndexError:
                pass  # catch first iter when no data available

            # -- STATE UPDATE & OBSERVATION --
            if update_state:
                # update & append simulation data
                self._update_time()  # note timing update is first
                self._update_ems_and_weather_vals(
                    self.ems_names_master_list
                )  # update sensor/actuator/weather/ vals
                self.callback_calling_points.append(calling_point)
                # run user-defined agent state update function
                if (
                    observation_fxn is not None
                    and self.timestep_zone_num_current % update_observation_frequency
                    == 0
                ):
                    # execute user's state/reward observation
                    if observation_function_kwargs:
                        # with kwargs
                        reward = observation_fxn(**observation_function_kwargs)
                    else:
                        # w/out kwargs
                        reward = observation_fxn()

                    if reward is not None:  # reward returned
                        if not self.rewards_created:
                            self._init_reward(reward)
                        self._update_reward(reward)

            # -- ACTION UPDATE --
            if (
                actuation_fxn is not None
                and self.timestep_zone_num_current % update_actuation_frequency == 0
            ):
                # execute user's actuation function
                if actuation_function_kwargs:
                    # with kwargs
                    self._actuate_from_list(
                        calling_point, actuation_fxn(**actuation_function_kwargs)
                    )
                else:
                    # w/out kwargs
                    self._actuate_from_list(calling_point, actuation_fxn())

            # -- INIT/UPDATE CUSTOM DFS --
            # Init
            if not self.custom_dataframes_initialized:
                # TODO what if simulation loops without new instance, data overload (auto-clear in run fxn?)
                self._init_custom_dataframe_dict()
                self.custom_dataframes_initialized = True
            # Update
            self._update_custom_dataframe_dicts(calling_point)

            # -- UPDATE DATA --
            # callback count
            self.callback_current_count += 1
            self.callbacks_count.append(self.callback_current_count)

        return _callback_function

    def _init_calling_points_and_callback_functions(self):
        """This iterates through the Calling Point Dict{} to set runtime calling points with actuation functions."""

        update_state_callback = False
        update_cp_list = []
        for calling_key in self.calling_point_callback_dict:
            # check if user-specified calling point is correct and available
            if calling_key not in self.available_calling_points:
                raise Exception(
                    f"ERROR: The calling point [{calling_key}] is not a valid calling point. Please see"
                    f" the Python API 0.2 documentation and available calling point list: "
                    f"emspy.available_calling_points class attribute."
                )
            else:
                # unpack observation & actuation fxns and callback fxn arguments
                (
                    observation_fxn,
                    actuation_fxn,
                    update_state,
                    update_observation_freq,
                    update_act_freq,
                    observation_fxn_kwargs,
                    actuation_fxn_kwargs,
                ) = self.calling_point_callback_dict[calling_key]

                # verify only one EMS update per timestep is advised
                if update_state:
                    update_cp_list.append(calling_key)
                    if update_state_callback:
                        print(
                            f"*WARNING: You are updating your entire EMS state multiple times a timestep, "
                            f"at calling points [{update_cp_list}]. Only once is advised."
                        )
                    update_state_callback = True

                # via API, establish calling points at runtime and create/pass its custom callback function
                getattr(self.api.runtime, calling_key)(
                    self.state,
                    self._enclosing_callback(
                        calling_key,
                        observation_fxn,
                        actuation_fxn,
                        update_state,
                        update_observation_freq,
                        update_act_freq,
                        observation_fxn_kwargs,
                        actuation_fxn_kwargs,
                    ),
                )

                # report message summary to user
                actuation_msg = "Yes" if actuation_fxn is not None else "No"
                observation_msg = "Yes" if observation_fxn is not None else "No"
                # print(
                #    f"\n*NOTE: Callback Function Summary:"
                #    f"\n\t\t\t- Calling Point [{calling_key}]"
                #    f"\n\t\t\t- Observation: [{observation_msg}]"
                #    f"\n\t\t\t- Actuation: [{actuation_msg}]"
                #    f"\n\t\t\t- State Update: [{update_state}]"
                #    f"\n\t\t\t- State Update Freq: [{update_observation_freq}]"
                #    f"\n\t\t\t- Action Update Freq: [{update_act_freq}]"
                # )

    def _init_custom_dataframe_dict(self):
        """Initializes custom EMS metric dataframes attributes at specific calling points & frequencies."""

        # add to dataframe  to fetch & track data during sim
        for df_name in self.df_custom_dict:
            ems_metrics, calling_point, update_freq = self.df_custom_dict[df_name]
            self.df_count += 1
            if calling_point not in self.calling_point_callback_dict:
                raise Exception(
                    f"ERROR: Invalid Calling Point name [{calling_point}].\nSee your declared available"
                    f" calling points {self.calling_point_callback_dict.keys()}."
                )
            # metric names must align with the EMS metric names assigned in var, intvar, meters, actuators, weather ToC
            ems_custom_dict = {"Datetime": [], "Timestep": []}
            # handle reward tracking
            if self.rewards:
                is_reward = "rewards"
            else:
                is_reward = ""
            for metric in ems_metrics:
                # verify proper input
                if metric not in self.ems_names_master_list + [is_reward]:
                    raise Exception(
                        f"ERROR: Incorrect EMS metric name, [{metric}], was entered for custom "
                        f"dataframes."
                    )
                # unused actuators
                if (
                    metric in self.tc_actuator
                    and metric not in self._actuators_used_set
                ):
                    raise Exception(
                        "ERROR: The EMS actuator [{metric}] was not by user and has no data to track."
                    )
                # create dict to collect data for pandas dataframe
                if metric == "rewards" and self.rewards_multi:  # multiple reward
                    for i in range(self.rewards_cnt):
                        metric = "reward" + str(i + 1)  # reward#, 1-n
                        ems_custom_dict[metric] = []
                else:
                    ems_custom_dict[metric] = []  # single reward, all else EMS
            # update custom df tracking list
            self.df_custom_dict[df_name][0] = ems_custom_dict

    def _update_custom_dataframe_dicts(self, calling_point):
        """Updates dataframe data based on desired calling point, timestep frequency, and specific ems vars."""

        # TODO handle redundant data collection when cp & freq are identical to default (may not always be applicable)
        if not self.df_custom_dict:
            return  # no custom dicts created
        # iterate through and update all default and user-defined dataframes
        for df_name in self.df_custom_dict:
            ems_dict, cp, update_freq = self.df_custom_dict[df_name]  # unpack value
            if (
                cp is calling_point
                and self.timestep_zone_num_current % update_freq == 0
            ):
                reward_index = 0  # TODO make independent of perfect order of reward, make robust to reward name int
                for ems_name in ems_dict:
                    # get most recent data point
                    if ems_name == "Datetime":
                        data_i = self.t_datetimes[-1]
                    elif ems_name == "Timestep":
                        data_i = self.timesteps_zone_num[-1]
                    elif "reward" in ems_name:
                        if self.rewards_multi:
                            data_i = self.rewards[-1][
                                reward_index
                            ]  # extra ith reward of most recent reward
                            reward_index += 1
                        else:
                            data_i = self.rewards[-1]
                    else:
                        # normal ems types
                        ems_type = self._get_ems_type(ems_name)
                        if ems_type == "setpoint":  # actuator setpoints
                            data_list_name = (
                                "data_" + ems_name
                            )  # setpoint is redundant, user must input themselves
                        else:  # all other
                            data_list_name = "data_" + ems_type + "_" + ems_name
                        data_i = getattr(self, data_list_name)[-1]

                    # append to dict list
                    self.df_custom_dict[df_name][0][ems_name].append(data_i)

    def _get_ems_type(self, ems_metric: str):
        """Returns EMS (var, intvar, meter, actuator, weather) or time type string for a given ems metric variable."""

        return self.ems_type_dict[
            ems_metric
        ]  # used to create attribute var names 'data_' + type

    def _post_process_data(self):
        """Handles various cleanup of data after the simulation has ran, necessary for certain features.

        1): All unused actuators & their meta-data is removed from the class, and reported to the user
        2): ...
        """

        # (1) remove data of unused actuators, if applicable
        if self.tc_actuator:
            unused_actuators = []
            for actuator_name in self.tc_actuator:
                if actuator_name not in self._actuators_used_set:
                    # print(
                    #    f"*NOTE: The actuator [{actuator_name}] was not used by EMS to actuator. Their EMS tracked "
                    #    f"null data attributes will be removed."
                    # )
                    # remove their data attributes
                    delattr(self, "data_actuator_" + actuator_name)
                    unused_actuators.append(actuator_name)
            # update EMS actuator number dictionary - relates to default DF creation,
            original_num = self.ems_num_dict["actuator"]
            updated_num = original_num - len(unused_actuators)
            # report to user
            if updated_num == 0:  # last actuator left
                self.ems_num_dict.pop("actuator")
                # print(
                #    f"*NOTE: No EMS actuators of [{original_num}] were used, all have been removed from your "
                #    f"simulation object."
                # )
            else:
                self.ems_num_dict["actuator"] = updated_num
                # print(
                #    f"*NOTE: [{updated_num}] of [{original_num}] actuators were used in this simulation."
                # )

    def _user_input_check(self):
        """Iterates through a pre-check to make sure all user input is valid and that this class was used properly.
        Warning:
            1) Check if calling point dictionary was instantiated by user
        Error:
            1)...
        """

        # TODO create function that checks if all user-input attributes has been specified and add help directions
        # warnings
        if not self.calling_point_callback_dict:
            print(
                "\n*WARNING: No calling points or callback actuation/observation functions were initialized."
                "Will just run simulation!\n"
            )
        # TODO detrimental errors

    def _new_state(self):
        """Creates & returns a new state instance that's required to pass into EnergyPlus Runtime API function calls."""

        return self.api.state_manager.new_state()

    def reset_state(self):
        """Resets the state instance of a simulation per EnergyPlus State API documentation."""

        self.api.state_manager.reset_state(self.state)

    def delete_state(self):
        """Deletes the existing state instance."""

        self.api.state_manager.delete_state(self.state)

    def run_simulation(self, weather_file: str):
        """This runs the EnergyPlus simulation and RL experiment."""

        # check valid input by user
        self._user_input_check()

        # create callback function(s) and link with calling point(s), if applicable
        if self.calling_point_callback_dict:
            self._init_calling_points_and_callback_functions()

        # RUN SIMULATION
        # print("\n* * * Running E+ Simulation * * *\n")
        self.api.runtime.set_console_output_status(self.state, print_output=False)
        self.simulation_success = self.api.runtime.run_energyplus(
            self.state, ["-w", weather_file, "-d", "out", self.idf_file]
        )  # cmd line args
        if self.simulation_success != 0:
            print("\n* * * Simulation FAILED * * *\n")
        # simulation successful
        else:
            # print("\n* * * Simulation Done * * *")
            self._post_process_data()

        # Free cPython memory
        self.api.runtime.clear_callbacks()
        self.delete_state()


class BcaEnv(EmsPy):
    """
    Building Control Agent & ENVironemnt (BcaEnv)

    This class represents the interface to the Building Control Agent (bca) and Environment for RL. It exposes a higher
    level of abstraction to its parent class, emspy, encapsulating more complex features.
    Users should be able to perform all necessary interactions for control algorithm experimentation mostly or
    completely through this class.
    """

    # Building Control Agent (bca) & Environment
    def __init__(
        self,
        ep_path: str,
        ep_idf_to_run: str,
        timesteps: int,
        tc_vars: dict,
        tc_intvars: dict,
        tc_meters: dict,
        tc_actuator: dict,
        tc_weather: dict,
    ):
        """See emspy.__init__() documentation."""

        # follow same init procedure as parent class emspy
        super().__init__(
            ep_path,
            ep_idf_to_run,
            timesteps,
            tc_vars,
            tc_intvars,
            tc_meters,
            tc_actuator,
            tc_weather,
        )
        self.ems_list_update_checked = (
            False  # TODO get rid off, doesnt work with multiple method instances
        )

    def set_calling_point_and_callback_function(
        self,
        calling_point: str,
        observation_function,
        actuation_function,
        update_state: bool,
        update_observation_frequency: int = 1,
        update_actuation_frequency: int = 1,
        observation_function_kwargs: dict = None,
        actuation_function_kwargs: dict = None,
    ):
        """
        Sets connection for runtime calling points and custom callback function specification with defined arguments.

        This will be used to created user-defined callback functions, including an (optional) observation function,
        (optional) actuation function, state update condition, and state update and action update frequencies.
        This allows the user to create the conventional RL agent interaction -> get state -> take action, each with
        desired timestep frequencies of implementation.

        :param calling_point: the calling point at which the callback function will be called during simulation runtime
        :param observation_function: the user defined observation function to be called at runtime calling point and desired
        timestep frequency, to be used to gather state data for agent before taking actions.
        :param actuation_function: the user defined actuation function to be called at runtime calling point and desired
        timestep frequency, function must return dict of actuator names (key) and associated setpoint (value)
        :param update_state: whether EMS and time/timestep data should be updated from simulation for this calling point.
        :param update_observation_frequency: the number of zone timesteps per running the observation function
        :param update_actuation_frequency: the number of zone timesteps per updating the actuators from the actuation
        function
        :param observation_function_kwargs: a dictionary to be fixed input to your observation function as **kwargs.
        **kwargs must be in your passed observation function to use, otherwise leave as None
        :param actuation_function_kwargs: a dictionary to be fixed input to your actuation function as **kwargs.
        **kwargs must be in your passed actuation function to use, otherwise leave as None
        """

        if update_actuation_frequency > update_observation_frequency:
            print(
                f"\n*WARNING: It is unusual to have your action update more frequent than your state update\n"
            )
        if calling_point in self.calling_point_callback_dict:  # overwrite error
            raise Exception(
                f"ERROR: You have overwritten the calling point '{calling_point}'. Keep calling points unique."
            )
        else:
            self.calling_point_callback_dict[calling_point] = [
                observation_function,
                actuation_function,
                update_state,
                update_observation_frequency,
                update_actuation_frequency,
                observation_function_kwargs,
                actuation_function_kwargs,
            ]

    def _check_ems_metric_input(self, ems_metric):
        """Verifies user-input of EMS metric/type list is valid."""

        # catch EMS type names (var, intvar, meter, actuator, etc.)
        if ems_metric in self.ems_num_dict:
            raise Exception(
                f"ERROR: EMS categories can only be called by themselves, please only call one at a "
                f"time."
            )
        # catch invalid EMS metric names
        elif (
            ems_metric not in self.ems_names_master_list
            and ems_metric not in self.available_timing_metrics
        ):
            raise Exception(
                f"ERROR: The EMS/timing metric [{ems_metric}] is not valid. Please see your EMS ToCs"
                f" or emspy.ems_master_list & emspy.times_master_list for available EMS & "
                f"timing metrics"
            )

    def get_ems_data(
        self, ems_metric_list: list, time_reverse_index=0, return_dict: bool = False
    ):
        """
        This takes desired EMS metric(s) (or type) & returns the entire current data set(s) OR at specific time indices.

        This function should be used to collect user-defined state space OR time information at each timestep. 1 to all
        EMS metrics and timing can be called, or just a single EMS Category (either 'var', 'intvar', 'meter', 'actuator'
        or, 'weather'), and 1 data point to the entire current data set can be returned.

        It is likely that the user will want the most recent data point only, this is a default argument.

        If calling any default timing data, see emspy.available_timing_metrics for available timing data.

        :param ems_metric_list: list of strings (or single element) of any available EMS/timing metric(s) to be called,
        or ONLY ONE entire EMS category ('var', 'intvar', 'meter', 'actuator', 'weather', 'time')
        :param time_reverse_index: list (or single value) of timestep indexes, applied to all EMS/timing metrics starting
        from index 0 as most recent available data point. Passing an empty list [] will return the entire current data
        list for each specified EMS metric.
        :param return_dict: True if you want to return data in dictionary format, with EMS name as the key, otherwise
        raw list will be returned in order of EMS items in the input list or EMS ToC
        :return return_data_list: nested list of data for each EMS metric at each time index specified, or entire list
        """

        # Handle single val inputs -> convert to list for rest of function
        single_val = False
        single_metric = False
        # metrics
        if type(ems_metric_list) is not list:  # assuming single metric
            ems_metric_list = [ems_metric_list]
        if len(ems_metric_list) == 1:
            single_metric = True
        # time indexes
        if (
            type(time_reverse_index) is not list
            and type(time_reverse_index) is not range
        ):  # assuming single time
            time_reverse_index = [time_reverse_index]  # make single time iterable
        if len(time_reverse_index) == 1:
            single_val = True

        # Check if only an EMS category called
        ems_type = ems_metric_list[0]
        if ems_type in self.ems_num_dict:
            if single_metric:
                # reassign entire metrics list to a single EMS type
                if ems_type == "time":
                    ems_metric_list = self.available_timing_metrics
                else:
                    ems_metric_list = list(
                        getattr(self, "tc_" + ems_type).keys()
                    )  # get all EMS metrics of that type
                if len(ems_metric_list) > 1:
                    single_metric = False
            else:
                raise ValueError(
                    f"If returning data for an entire EMS Category, you can only do one at a time from the"
                    f'list "var", "intvar", "meter", "weather", or "actuator. Your input was '
                    f"{ems_metric_list}"
                )

        if return_dict:
            return_data = {}  # organized by key EMS name
        else:
            return_data = []  # organized by order, only raw values

        for ems_metric in ems_metric_list:

            # TODO do once, again at each timestep is redundant?
            # self._check_ems_metric_input(ems_metric)  # verify valid input
            ems_type = self._get_ems_type(ems_metric)  # for attribute variable name

            if not time_reverse_index:
                # no time index specified, return ALL current data available,  #TODO this doens't compensate for time data
                if return_dict:
                    return_data[ems_metric] = getattr(
                        self, "data_" + ems_type + "_" + ems_metric
                    )
                else:
                    return_data.append(
                        getattr(self, "data_" + ems_type + "_" + ems_metric)
                    )
            else:
                # iterate through previous time indexes
                return_data_indexed = []
                for time in time_reverse_index:
                    if ems_type != "time":
                        ems_name = "data_" + ems_type + "_" + ems_metric
                    else:
                        ems_name = ems_metric  # TODO update timing metrics
                    try:
                        data_indexed = getattr(self, ems_name)[-1 - time]
                        if single_val:
                            return_data_indexed = data_indexed  # so that nested list of single-element is Not returned
                        else:
                            return_data_indexed.append(data_indexed)
                    except IndexError:
                        # print(
                        #    "\n*NOTE: Not enough simulation time elapsed to collect data at specified index.\n"
                        # )
                        pass
                        # TODO add feature that will add what data is available, IFF helpful

                # No unnecessarily nested lists
                if single_metric:
                    # handle dictionary form factor if needed
                    if return_dict:
                        return {ems_metric: return_data_indexed}
                    else:
                        return return_data_indexed
                else:
                    if return_dict:
                        return_data[ems_metric] = return_data_indexed
                    else:
                        return_data.append(return_data_indexed)

        return return_data

    def get_weather_forecast(
        self, weather_metrics: list, when: str, hour: int, zone_ts: int
    ):
        """
        Fetches given weather metric from today/tomorrow for a given hour of the day and timestep within that hour.

        :param weather_metrics: list of desired weather metric(s) (1 to all) from weather ToC dict
        :param when: 'today' or 'tomorrow' relative to current timestep
        :param hour: hour of day
        :param zone_ts: timestep of hour
        """

        return self._get_weather(weather_metrics, when, hour, zone_ts)

    def update_ems_data(self, ems_metric_list: list, return_data: bool):
        """
        This takes desired EMS metric(s) (or type) to update from the sim (and opt return val) at calling point.

        This OPTIONAL function can be used to update/collect specific EMS at each timestep for a given calling point.
        One to all EMS metrics can be called, or just EMS category (var, intvar, meter, actuator, weather) and have the
        data point updated & returned.

        *This is ONLY NEEDED if the user wants to update specific EMS metrics at a unique calling point, separate from
        ALL EMS metrics if using default state update.

        This also works for default timing data.  # TODO does not work currently with _update_ems_and_weather_val

        :param ems_metric_list: list of any available EMS metric(s) to be called, or ONLY ONE entire EMS category
        (var, intvar, meter, actuator, weather) in a list
        :param return_data: Whether or not to return an order list of the data points fetched
        :return return_data_list: list of fetched data for each EMS metric at each time index specified or None
        """

        if ems_metric_list[0] in self.ems_num_dict and len(ems_metric_list) == 1:
            # if only single EMS category called
            ems_metric_list = list(getattr(self, "tc_" + ems_metric_list[0]).keys())
        else:
            # TODO get rid of, doesnt work with multiple method instances - OR make work with multiple
            if not self.ems_list_update_checked:
                for ems_metric in ems_metric_list:
                    self._check_ems_metric_input(ems_metric)
                    self.ems_list_update_checked = True
                    print("Check once 1")

        self._update_ems_and_weather_vals(ems_metric_list)
        if return_data:
            return self.get_ems_data(ems_metric_list)  # return most recent update
        else:
            return None

    def init_custom_dataframe_dict(
        self, df_name: str, calling_point: str, update_freq: int, ems_metrics: list
    ):
        """
        Initialize custom EMS metric dataframes attributes at specific calling points & frequencies to be tracked.

        Desired setpoint data from actuation actions can be acquired and compared to updated system setpoints - Use
        'setpoint' + your_actuator_name as the EMS metric name. Rewards can also be fetched. These are NOT collected
        by default dataframes.

        :param df_name: user-defined df variable name
        :param calling_point: the calling point at which the df should be updated
        :param update_freq: how often data will be posted, it will be posted every X timesteps
        :param ems_metrics: list of EMS metric names, 'setpoint+...', or 'rewards', to store their data points in df
        """
        self.df_count += 1
        self.df_custom_dict[df_name] = [ems_metrics, calling_point, update_freq]

    def run_env(self, weather_file_path: str):
        """
        Runs E+ simulation for given .IDF building model and EPW Weather File
        """

        self.run_simulation(weather_file_path)


class Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.

    NOTE: The observation and actuation functions are callback functions that, depending on your configuration (calling
    points,

    Depending on your implementation, an observation function may not be needed. The observation function just
    allows for extra flexibility in gathering state data from the simulation, as it happens before the actuation
    function is calle, and it can be used to return and store "reward" throughout the simulation, if desired.

    The actuation function is used to take actions. And must return an "actuation dictionary" that has the specific
    EMS actuator variable with its corresponding control value.
    """

    def __init__(self, bca: BcaEnv):
        self.bca = bca

        self.input_queue = None
        self.results_queue = None

        self.first_timestep = True

    def observation_function(self):
        time = self.bca.get_ems_data(["t_datetimes"])

        var_names = list(self.bca.tc_var.keys())

        var_data = self.bca.get_ems_data(var_names)

        var_dict = {var_names[i]: v for i, v in enumerate(var_data)}

        indoor_temp = var_dict["zn0_temp"]

        electricity_ts = [v for k, v in var_dict.items() if "electricity" in k]

        consumption = J_to_kWh(sum(electricity_ts))

        observation = {
            "datetime": time,
            "indoor_temp_C": indoor_temp,
            "consumption_kWh": consumption,
        }

        if "heating_coil_energy" in var_dict:
            main_heating_energy = J_to_kWh(var_dict["heating_coil_energy"])
            observation["heat_energy_main_kWh"] = main_heating_energy
            main_heating_electricity = J_to_kWh(
                var_dict["heating_coil_electricity_energy"]
            )
            observation["heat_electricity_main_kWh"] = main_heating_electricity

        if "heating_backup_coil_energy" in var_dict:
            heating_backup_energy = J_to_kWh(var_dict["heating_backup_coil_energy"])
            observation["heat_energy_backup_kWh"] = heating_backup_energy
            heating_backup_electricity = J_to_kWh(
                var_dict["heating_backup_coil_electricity_energy"]
            )
            observation["heat_electricity_backup_kWh"] = heating_backup_electricity

        if "cooling_coil_total_energy" in var_dict:
            cooling_backup_energy = J_to_kWh(var_dict["cooling_coil_total_energy"])
            observation["cool_energy_kWh"] = cooling_backup_energy
            cooling_backup_electricity = J_to_kWh(var_dict["cooling_coil_electricity"])
            observation["cool_electricity_kWh"] = cooling_backup_electricity

        self.results_queue.put(observation)

    def actuation_function(self):
        time = self.bca.get_ems_data(["t_datetimes"])

        setpoints = self.input_queue.get()

        if setpoints == "stop":
            self.bca.api.runtime.stop_simulation(self.bca.state)
            return

        return setpoints


if __name__ == "__main__":
    keep_cols = ["temp_air", "dir_norm_rad"]
    time_step = datetime.timedelta(minutes=15)
    epw_file = "data/common/energyplus_models/brussels_2022.epw"

    epw_data = parse_epw_weather(epw_file, keep_cols)
    epw_data_interpolated = interpolate_epw(epw_data, time_step)
    print("Interpolation done")
