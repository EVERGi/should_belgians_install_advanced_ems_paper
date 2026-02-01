from simugrid.assets.asset import Asset
from simugrid.simulation.power import Power
from simugrid.simulation.run_energyplus import (
    BcaEnv,
    EmsPy,
    Agent,
    add_epw_file_to_environment,
)

from threading import Thread
import queue
import datetime
import pytz
from copy import deepcopy
import warnings


def temp_c_to_f(temp_c: float, arbitrary_arg=None):
    """Convert temp from C to F. Test function with arbitrary argument, for example."""
    x = arbitrary_arg
    return 1.8 * temp_c + 32


class EnergyplusThread(Thread):
    def __init__(self, sim, epw_weather, agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.sim = sim
        self.epw_weather = epw_weather
        self.agent = agent

        self.agent.input_queue = self.input_queue
        self.agent.results_queue = self.results_queue

        self.daemon = True

    def run(self):

        self.sim.run_env(self.epw_weather)


class EnergyPlus(Asset):
    def __init__(self, node, name):
        super().__init__(node, name)
        self.size = 1

        self.idf_model = None
        self.epw_weather = None
        self.eplus_dir = None
        self.sim = None
        self.occupancy = {
            "weekday": {
                "00:00-07:00": "sleeping",
                "07:00-08:00": "at home",
                "08:00-18:00": "absent",
                "18:00-23:15": "at home",
                "23:15-24:00": "sleeping",
            },
            "saturday": {
                "00:00-07:30": "sleeping",
                "07:30-22:30": "at home",
                "22:30-24:00": "sleeping",
            },
            "sunday": {
                "00:00-07:30": "sleeping",
                "07:30-22:30": "at home",
                "22:30-24:00": "sleeping",
            },
        }
        self.timezone = "Europe/Brussels"

        self.zn0 = "conditioned space"

        self.zn0_temp = 21  # deg C
        self.cur_t_low = 5  # deg C
        self.cur_t_up = 40  # deg C

        self._saved_readings = None
        self._saved_t_days = dict()
        self._thread_started = False
        self._weather_in_env = False
        self._cached_comfort_range = dict()

        self.tc_intvar = {}
        self.tc_var = {
            "zn0_temp": [
                "Zone Mean Air Temperature",
                "conditioned space",
            ],  # deg C
            "fan_electricity": [
                "Fan Electricity Energy",
                "air source heat pump supply fan",
            ],  # J for each timestep
            # "heating_backup_coil_energy": [
            #    ("Heating Coil Heating Energy", "air source heat pump backup htg coil")
            # ],  # J for each timestep: Usefull but not used
            "heating_backup_coil_electricity_energy": [
                "Heating Coil Electricity Energy",
                "air source heat pump backup htg coil",
            ],  # J for each timestep
            # "heating_coil_energy": [
            #    ("Heating Coil Heating Energy", "air source heat pump htg coil")
            # ],  # J for each timestep: Usefull but not used
            "heating_coil_electricity_energy": [
                "Heating Coil Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            # Heating Coil Crankcase Heater Electricity Energy
            "heating_coil_crankcase_heater_electricity": [
                "Heating Coil Crankcase Heater Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            # Heating Coil Defrost Electricity Energy
            "heating_coil_defrost_electricity": [
                "Heating Coil Defrost Electricity Energy",
                "AIR SOURCE HEAT PUMP HTG COIL",
            ],  # J for each timestep
            "cooling_coil_electricity": [
                "Cooling Coil Electricity Energy",
                "AIR SOURCE HEAT PUMP CLG COIL",
            ],  # J for each timestep
            # "cooling_coil_total_energy": [
            #    "Cooling Coil Total Cooling Energy",
            #    "air source heat pump clg coil",
            # ],  # J for each timestep: Usefull but not used
        }
        self.tc_meter = {
            # "electricity_facility": [f"Electricity:Zone:{self.zn0}"],  # J
        }

        self.tc_weather = {
            # "oa_rh": [("outdoor_relative_humidity")],  # %RH
            # "oa_db": [("outdoor_dry_bulb"), temp_c_to_f],  # deg C
            # "oa_pa": [("outdoor_barometric_pressure")],  # Pa
            # "sun_up": [("sun_is_up")],  # T/F
            # "rain": [("is_raining")],  # T/F
            # "snow": [("is_snowing")],  # T/F
            # "wind_dir": [("wind_direction")],  # deg
            # "wind_speed": [("wind_speed")],  # m/s
        }
        self.tc_actuator = {
            # HVAC Control Setpoints
            "zn0_cooling_sp": [
                "Zone Temperature Control",
                "Cooling Setpoint",
                self.zn0,
            ],  # deg C
            "zn0_heating_sp": [
                "Zone Temperature Control",
                "Heating Setpoint",
                self.zn0,
            ],  # deg C
            # "zn0_water_heater": None,
        }

        time_step = self.parent_node.microgrid.time_step
        t_s_minutes = time_step.total_seconds() / 60

        self.sim_timesteps = 60 // t_s_minutes  # every 60 / sim_timestep minutes

        self.max_consumption_power = 10_000  # kW very high to avoid limiting power

        self.pickle_deepcopy = False

    @property
    def environment_keys(self):
        return dict()

    def set_power_limits(self, environment):
        self.power_limit_low = Power(electrical=-self.max_consumption_power)
        self.power_limit_high = Power()

    def check_and_set_model(self):
        super().check_and_set_model()
        model_set = self.idf_model is not None
        weather_file_set = self.epw_weather is not None
        eplus_path_set = self.eplus_dir is not None
        all_models_set = model_set and weather_file_set and eplus_path_set

        add_weather_to_env = weather_file_set and not self._weather_in_env

        if add_weather_to_env and self.parent_node.environment is not None:
            environment = self.parent_node.environment
            add_epw_file_to_environment(self.epw_weather, environment)
            self._weather_in_env = True

        if all_models_set and not self._thread_started:
            self.start_energyplus_thread()

    def power_consequences(self):
        comfort_range = self.get_comfort_range()
        self.cur_t_low = comfort_range[0]
        self.cur_t_up = comfort_range[1]

    def start_energyplus_thread(self):
        ep_path = f"{self.eplus_dir}"
        self.sim = BcaEnv(
            ep_path=ep_path,
            ep_idf_to_run=self.idf_model,
            timesteps=self.sim_timesteps,
            tc_vars=self.tc_var,
            tc_intvars=self.tc_intvar,
            tc_meters=self.tc_meter,
            tc_actuator=self.tc_actuator,
            tc_weather=self.tc_weather,
        )
        self.agent = Agent(self.sim)

        calling_point_for_callback_fxns = EmsPy.available_calling_points[
            6
        ]  # 5-15 valid for timestep loop during simulation

        self.sim.set_calling_point_and_callback_function(
            calling_point=calling_point_for_callback_fxns,
            observation_function=None,
            actuation_function=self.agent.actuation_function,
            update_state=False,
            update_actuation_frequency=1,
        )
        calling_point_for_callback_fxns = EmsPy.available_calling_points[
            15
        ]  # 5-15 valid for timestep loop during simulation
        self.sim.set_calling_point_and_callback_function(
            calling_point=calling_point_for_callback_fxns,
            observation_function=self.agent.observation_function,
            actuation_function=None,
            update_state=True,
            update_observation_frequency=1,
        )

        # Execute self.sim.run_env() in a different thread
        self.thread = EnergyplusThread(
            self.sim, self.epw_weather, self.agent, name="EnergyplusThread"
        )

        self.thread.start()
        self._thread_started = True

    def stop_energyplus_thread(self):
        if self.thread is not None:
            self.thread.input_queue.put("stop")
            self.thread.join()
            self._thread_started = False

    def get_readings(self):

        return self._saved_readings

    def set_default_comfort_range(self):
        comfort_range = self.get_comfort_range()
        heating_setpoint = comfort_range[0]
        cooling_setpoint = comfort_range[1]

        setpoints = {
            "zn0_heating_sp": heating_setpoint,
            "zn0_cooling_sp": cooling_setpoint,
        }
        self.set_setpoints(setpoints)

    def set_setpoints(self, setpoints):
        self.thread.input_queue.put(setpoints)
        self.fetch_readings()
        self.set_reading_limits()

    def fetch_readings(self):
        try:
            readings = self.thread.results_queue.get(timeout=5)
        except queue.Empty:
            self.stop_energyplus_thread()
            # Raise an error 
            raise RuntimeError("EnergyPlus simulation crashed or timed out.")

        self._saved_readings = readings

        self.zn0_temp = readings["indoor_temp_C"]

    def set_reading_limits(self):
        ts_hours = self.microgrid.time_step.total_seconds() / 3600
        eletric_power = self._saved_readings["consumption_kWh"] / ts_hours

        self.power_limit_low = Power(electrical=-eletric_power)
        self.power_limit_high = Power(electrical=-eletric_power)

    def get_comfort_range(self, mode="approximated"):
        dt = self.parent_node.microgrid.utc_datetime
        comfort_range = self.get_comfort_dt(dt, mode)

        return comfort_range

    def get_comfort_dt(self, dt, mode="approximated", cached=True):

        if cached:
            if mode not in self._cached_comfort_range.keys():
                self._cached_comfort_range[mode] = dict()
            if dt in self._cached_comfort_range[mode].keys():
                return self._cached_comfort_range[mode][dt]

        local_dt = dt.astimezone(pytz.timezone(self.timezone))
        t_ref = self.get_t_ref(local_dt, mode)

        status = self.calc_occupancy_status(self.occupancy, local_dt)
        comfort_range = self.calc_comfort_range(t_ref, status)

        if cached:
            self._cached_comfort_range[mode][dt] = comfort_range

        return comfort_range

    def get_comfort_sequence(self, start_dt, end_dt, mode="approximated"):
        time_step = self.parent_node.microgrid.time_step
        cur_dt = start_dt
        comfort_sequence = {"t_low": [], "t_up": []}
        while cur_dt < end_dt:
            comfort_range = self.get_comfort_dt(cur_dt, mode)
            comfort_sequence["t_low"].append(comfort_range[0])
            comfort_sequence["t_up"].append(comfort_range[1])
            cur_dt += time_step

        return comfort_sequence

    def get_t_ref(self, dt, mode="approximated"):
        if mode not in self._saved_t_days:
            self._saved_t_days[mode] = dict()

        cur_date = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        used_dates = [cur_date - datetime.timedelta(hours=i * 24) for i in range(4)]
        if mode == "approximated":
            # Use the previous day twice to approximate the current day t_ref
            used_dates[0] = cur_date - datetime.timedelta(hours=24)

        t_days = []
        for date in used_dates:
            if date not in self._saved_t_days[mode]:
                self._saved_t_days[mode][date] = self.fetch_t_day(date)

            t_days.append(self._saved_t_days[mode][date])

        t_ref = self.calc_t_ref(t_days)
        return t_ref

    def fetch_t_day(self, cur_date):
        local_tz = pytz.timezone(self.timezone)

        start_dt = local_tz.localize(cur_date).astimezone(pytz.utc)
        end_dt = cur_date + datetime.timedelta(hours=24)
        end_dt = local_tz.localize(end_dt).astimezone(pytz.utc)

        env_value = self.parent_node.environment.env_values["epw_temp_air"]
        day_temps = env_value.get_forecast(start_dt, end_dt)["values"]
        t_day = (min(day_temps) + max(day_temps)) / 2
        return t_day

    @staticmethod
    def calc_comfort_range(t_ref, status):
        # W and Alpha values for 10% predicted percentage of dissatisfied
        # Source: https://doi.org/10.1016/j.apenergy.2008.07.011
        W = 5
        ALPHA = 0.7

        if status == "at home":
            if t_ref < 12.5:
                t_n = 20.4 + 0.06 * t_ref
            else:
                t_n = 16.63 + 0.36 * t_ref

            t_up = t_n + ALPHA * W
            t_low = max(t_n - (1 - ALPHA) * W, 18)
        elif status == "sleeping":
            if t_ref < 0:
                t_n = 16
            elif 0 <= t_ref < 12.6:
                t_n = 16 + 0.23 * t_ref
            elif 12.6 <= t_ref < 21.8:
                t_n = 9.18 + 0.77 * t_ref
            else:
                t_n = 26

            t_up = min(t_n + ALPHA * W, 26)
            t_low = max(t_n - (1 - ALPHA) * W, 16)
        elif status == "absent":
            t_low = 5
            t_up = 40

        return [t_low, t_up]

    @staticmethod
    def calc_t_ref(t_days):
        t_ref = (t_days[0] + 0.8 * t_days[1] + 0.4 * t_days[2] + 0.2 * t_days[3]) / 2.4
        return t_ref

    @staticmethod
    def calc_occupancy_status(occupancy_profile, dt):
        if dt.weekday() < 5:
            hour_profile = occupancy_profile["weekday"]
        elif dt.weekday() == 5:
            hour_profile = occupancy_profile["saturday"]
        elif dt.weekday() == 6:
            hour_profile = occupancy_profile["sunday"]
        str_time = dt.strftime("%H:%M")

        for time_range, status in hour_profile.items():
            start, end = time_range.split("-")

            if start <= str_time < end:
                return status

    def get_specs(self):
        specs_to_find = {
            "heating_power_w": "Gross Rated Heating Capacity {W}",
            "cooling_power_w": "Gross Rated Total Cooling Capacity {W}",
            "backup_power_w": "Nominal Capacity {W}",
        }

        specs = dict()
        with open(self.idf_model, "r") as f:
            idf_model = f.readlines()
            for line in idf_model:
                for spec_name, spec in specs_to_find.items():
                    if spec in line:
                        if spec_name in specs.keys():
                            print(
                                f"Warning: Spec {spec_name} already found with search term {spec}"
                            )
                        specs[spec_name] = float(line.split(",")[0].replace(" ", ""))
        return specs

    def __deepcopy__(self, memo):
        if self.microgrid.start_time != self.microgrid.utc_datetime:
            warnings.warn(
                "Deepcopy does not work correctly for Energyplus objects when simulation already started."
            )

        # Stop simulation
        # if self.thread is not None:
        #    self.thread.input_queue.put("stop")

        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["thread", "sim", "agent"]:
                v = None
            setattr(obj, k, deepcopy(v, memo))

        if not self.pickle_deepcopy:

            obj.start_energyplus_thread()

        return obj
