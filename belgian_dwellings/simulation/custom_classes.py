from simugrid.management.rational import RationalManager
from simugrid.rewards.reward import Reward
from simugrid.assets.battery import Battery
from simugrid.assets.charger import Charger
from simugrid.assets.energyplus import EnergyPlus
from simugrid.assets.public_grid import PublicGrid
from simugrid.assets.water_heater import WaterHeater

from treec.utils import denormalise_input
from copy import deepcopy
import numpy as np

import pytz

import datetime


class DayAheadEngie(Reward):
    def __init__(self):
        list_KPI = [
            "day_ahead",
            "trans_dis",
            "capacity",
            "year_fixed",
            "opex",
            "discomfort",
            "forced_charged_energy",
            "soc_quantile",
        ]
        Reward.__init__(self, list_KPI)

        self.prev_cap_tariff = 0
        self.cur_peak_power = 2.5

        self.prev_temp = None

        self.soc_diff_list = list()
        self.energy_diff_list = list()

    def calculate_kpi(self):
        microgrid = self.microgrid

        time_step_h = microgrid.time_step.total_seconds() / 3600

        for asset in microgrid.assets:
            if isinstance(asset, PublicGrid):
                env = asset.parent_node.environment
                kwh_offtake_cost = env.env_values["kwh_offtake_cost"].value
                capacity_tariff = env.env_values["capacity_tariff"].value  # €/kW/month
                year_cost = env.env_values["year_cost"].value
                injection_extra = env.env_values["injection_extra"].value
                offtake_extra = env.env_values["offtake_extra"].value

                grid_energy = asset.power_output.electrical * time_step_h

                day_ahead_price = env.env_values["day_ahead_price"].value

                if grid_energy > 0:
                    day_ahead_price = day_ahead_price + offtake_extra
                    if day_ahead_price > 0:
                        day_ahead_cost = grid_energy * day_ahead_price * 1.06
                    else:
                        day_ahead_cost = grid_energy * day_ahead_price
                    self.KPIs["day_ahead"] -= day_ahead_cost
                    self.KPIs["trans_dis"] -= grid_energy * kwh_offtake_cost
                else:
                    day_ahead_price = day_ahead_price + injection_extra
                    day_ahead_cost = grid_energy * day_ahead_price
                    self.KPIs["day_ahead"] -= day_ahead_cost
                self.calc_capacity_tarriff(
                    asset.power_output.electrical, capacity_tariff
                )
                time_step_prop_year = time_step_h / (365 * 24)
                self.KPIs["year_fixed"] -= time_step_prop_year * year_cost

                self.KPIs["opex"] = (
                    self.KPIs["day_ahead"]
                    + self.KPIs["trans_dis"]
                    + self.KPIs["capacity"]
                    + self.KPIs["year_fixed"]
                )
            if isinstance(asset, EnergyPlus):
                utc_datetime = self.microgrid.utc_datetime
                time_step = self.microgrid.time_step
                prev_step = utc_datetime - time_step
                cur_temp = asset._saved_readings["indoor_temp_C"]
                if self.prev_temp is not None:
                    last_comfort_range = asset.get_comfort_dt(prev_step)
                    discomfort = discomfort_calc(
                        last_comfort_range, self.prev_temp, cur_temp, time_step
                    )
                    self.KPIs["discomfort"] += discomfort

                self.prev_temp = cur_temp
            elif isinstance(asset, Charger):
                management_system = self.microgrid.management_system
                forced_charged_power = management_system.forced_charged_power
                forced_charged_energy = forced_charged_power * time_step_h
                self.KPIs["forced_charged_energy"] += forced_charged_energy

                if asset.det == 1:
                    soc_diff = asset.soc_f - asset.soc
                    self.soc_diff_list.append(soc_diff)
                    energy_diff = soc_diff * asset.size / asset.eff
                    self.energy_diff_list.append(energy_diff)

                    self.KPIs["soc_quantile"] = np.quantile(
                        self.soc_diff_list, 0.95, method="hazen"
                    )

    def calc_capacity_tarriff(self, grid_power, capacity_tariff):
        utc_datetime = self.microgrid.utc_datetime
        time_step = self.microgrid.time_step

        start_month = datetime.datetime(utc_datetime.year, utc_datetime.month, 1, 0, 0)
        start_month = start_month.replace(tzinfo=pytz.utc)
        # Get the first day of next month
        start_next_month = (start_month + datetime.timedelta(days=32)).replace(day=1)

        is_new_month = utc_datetime == start_month
        if is_new_month:
            self.prev_cap_tariff = self.KPIs["capacity"]
            self.cur_peak_power = 2.5

        if grid_power > self.cur_peak_power:
            self.cur_peak_power = grid_power
        # Number of seconds in current month
        tot_sec_month = (start_next_month - start_month).total_seconds()
        # Get the time since beginning of the month or simulation
        if start_month >= self.microgrid.start_time:
            start_prop = start_month
        else:
            start_prop = self.microgrid.start_time

        sec_since_startprop = ((utc_datetime + time_step) - start_prop).total_seconds()

        use_prop_month = False
        if use_prop_month:
            prop_month = sec_since_startprop / tot_sec_month
        else:
            prop_month = 1
        self.KPIs["capacity"] = (
            self.prev_cap_tariff - prop_month * self.cur_peak_power * capacity_tariff
        )


def discomfort_calc(prev_comfort_range, prev_temp, cur_temp, time_step):
    time_step_h = time_step.total_seconds() / 3600

    to_low = prev_temp < prev_comfort_range[0] or cur_temp < prev_comfort_range[0]
    to_high = prev_temp > prev_comfort_range[1] or cur_temp > prev_comfort_range[1]

    if to_low:
        if prev_temp < prev_comfort_range[0] and cur_temp < prev_comfort_range[0]:
            time_discomfort = time_step_h
            discomfort = (
                abs(prev_temp - cur_temp) * time_discomfort / 2
                + (prev_comfort_range[0] - max(prev_temp, cur_temp)) * time_discomfort
            )
        elif prev_temp < prev_comfort_range[0]:
            time_discomfort = (
                time_step_h
                * (prev_comfort_range[0] - prev_temp)
                / (cur_temp - prev_temp)
            )
            discomfort = (prev_comfort_range[0] - prev_temp) * time_discomfort / 2
        else:
            time_discomfort = (
                time_step_h
                * (prev_comfort_range[0] - cur_temp)
                / (prev_temp - cur_temp)
            )
            discomfort = (prev_comfort_range[0] - cur_temp) * time_discomfort / 2
    elif to_high:
        if prev_temp > prev_comfort_range[1] and cur_temp > prev_comfort_range[1]:
            time_discomfort = time_step_h
            discomfort = (
                abs(prev_temp - cur_temp) * time_discomfort / 2
                + (min(prev_temp, cur_temp) - prev_comfort_range[1]) * time_discomfort
            )
        elif prev_temp > prev_comfort_range[1]:
            time_discomfort = (
                time_step_h
                * (prev_temp - prev_comfort_range[1])
                / (prev_temp - cur_temp)
            )
            discomfort = (prev_temp - prev_comfort_range[1]) * time_discomfort / 2
        else:
            time_discomfort = (
                time_step_h
                * (cur_temp - prev_comfort_range[1])
                / (cur_temp - prev_temp)
            )
            discomfort = (cur_temp - prev_comfort_range[1]) * time_discomfort / 2
    else:
        discomfort = 0

    return discomfort


def bound(low, high, value):
    return max(low, min(high, value))


class HouseManager(RationalManager):
    def __init__(self, microgrid):
        super().__init__(microgrid)
        self.grid = self.public_grid[0]
        self.control_points = dict()

        self.num_setpoints = 0

        self.t_low = 5
        self.t_high = 40

        self.cooling = True

        for energyplus in self.energyplus:
            self.control_points[energyplus] = {
                "zn0_heating_sp": self.t_low,
                "zn0_cooling_sp": self.t_high,
            }
            self.num_setpoints += 2

        for asset in self.chargers:
            self.control_points[asset] = {"power_sp": -asset.max_consumption_power}
            self.num_setpoints += 1
        for asset in self.water_heaters + self.batteries:
            self.control_points[asset] = {"power_sp": 0}
            self.num_setpoints += 1

        self.batteries = []
        self.chargers = []
        self.energyplus = []
        self.water_heaters = []

        self.forced_charged_power = 0

        self.forced_charging = False
        self.do_forced_charging = True

    def simulate_step(self):
        self.update_control_points()
        for asset in self.control_points:
            asset_class = type(asset)
            if issubclass(asset_class, Battery):
                battery = asset
                power_sp = self.control_points[battery]["power_sp"]
                bounded_power = bound(
                    battery.power_limit_low.electrical,
                    battery.power_limit_high.electrical,
                    power_sp,
                )
                self.exec_power_trans(battery, self.grid, power_send=bounded_power)
            elif issubclass(asset_class, Charger):
                charger = asset
                power_sp = self.control_points[charger]["power_sp"]
                if self.do_forced_charging:
                    charger_low, charger_high = charger.get_powers_to_reach_soc_final()
                    EPSILON = 0.0000001
                    forced_charge = -charger_high
                    if forced_charge > EPSILON:
                        self.forced_charging = True
                        # print("EMS: forced charging detected")
                    if forced_charge < EPSILON:
                        self.forced_charging = False
                    # if forced_diff > 0:
                    #    print("EMS: Charger forced to charge")
                    self.forced_charged_power = forced_charge
                else:

                    charger_low = charger.power_limit_low.electrical
                    charger_high = charger.power_limit_high.electrical
                bounded_charger = bound(charger_low, charger_high, power_sp)

                # Log forced power to save in the reward

                if bounded_charger != 0:
                    pass
                self.exec_power_trans(
                    self.grid, charger, power_send=abs(bounded_charger)
                )

            elif issubclass(asset_class, EnergyPlus):
                setpoints = deepcopy(self.control_points[asset])
                comfort_range = asset.get_comfort_range()
                if "zn0_heating_sp" in setpoints:
                    setpoints["zn0_heating_sp"] = min(
                        max(setpoints["zn0_heating_sp"], comfort_range[0]),
                        comfort_range[1],
                    )
                else:
                    setpoints["zn0_heating_sp"] = comfort_range[0]

                if "zn0_cooling_sp" in setpoints:
                    setpoints["zn0_cooling_sp"] = max(
                        min(setpoints["zn0_cooling_sp"], comfort_range[1]),
                        comfort_range[0],
                    )
                else:
                    if self.cooling:
                        setpoints["zn0_cooling_sp"] = comfort_range[1]
                    else:
                        setpoints["zn0_cooling_sp"] = 40
                asset.set_setpoints(setpoints)
                self.exec_power_trans(self.grid, asset)

            elif issubclass(asset_class, WaterHeater):
                water_heater = asset
                if water_heater.t_tank >= water_heater.high_setpoint:
                    power_sp = 0
                elif water_heater.t_tank < water_heater.low_setpoint:
                    power_sp = -water_heater.max_consumption_power
                else:
                    power_sp = self.control_points[water_heater]["power_sp"]
                self.exec_power_trans(self.grid, water_heater, power_send=abs(power_sp))

        super().simulate_step()

    def update_control_points(self):
        pass


class TreeManager(HouseManager):
    def __init__(self, microgrid, trees, input_func):
        super().__init__(microgrid)
        self.trees = trees
        self.input_func = input_func
        self.all_nodes_visited = []

    def update_control_points(self):
        action_nodes = list()

        leaf_indexes = list()

        input_dict = {"microgrid": self.microgrid}

        features, _ = self.input_func(input_dict)

        for i, tree in enumerate(self.trees):
            node = tree.get_action(features[i])

            leaf_indexes.append(tree.node_stack.index(node))

            action_nodes.append(node)

        action_count = 0
        for asset, setpoints in self.control_points.items():

            for setpoint_name in setpoints.keys():
                action_value = action_nodes[action_count].value
                action_name = action_nodes[action_count].feature

                if setpoint_name == "power_sp":
                    new_setpoint = action_value

                elif setpoint_name.startswith("zn0"):
                    new_setpoint = self.special_setpoint_heating(
                        action_name, action_value, asset
                    )

                else:
                    print("Unknown setpoint")

                self.control_points[asset][setpoint_name] = new_setpoint
                action_count += 1
            if (
                "zn0_heating_sp" in setpoints.keys()
                and "zn0_cooling_sp" in setpoints.keys()
            ):
                eplus_setpoints = self.control_points[asset]
                if (
                    eplus_setpoints["zn0_heating_sp"]
                    > eplus_setpoints["zn0_cooling_sp"]
                ):
                    eplus_setpoints["zn0_cooling_sp"] = setpoints["zn0_heating_sp"]

        self.all_nodes_visited.append(leaf_indexes)

    def special_setpoint_heating(self, action_name, action_value, asset):
        if action_name.endswith(" setpoint (°C): "):
            setpoint = action_value
        elif action_name.endswith(" current (°C): "):
            cur_temp = asset.zn0_temp
            setpoint_shift = action_value
            if action_name.startswith("Temperature above"):
                setpoint = cur_temp + setpoint_shift
            elif action_name.startswith("Temperature below"):
                setpoint = cur_temp - setpoint_shift
        elif action_name.endswith("temperature for next (h): "):
            delta_t_comfort = datetime.timedelta(hours=action_value)

            start_dt = self.microgrid.utc_datetime
            end_dt = start_dt + delta_t_comfort + self.microgrid.time_step
            comfort_sequence = asset.get_comfort_sequence(start_dt, end_dt)

            if action_name.startswith("Match highest low"):
                setpoint = max(comfort_sequence["t_low"])
            elif action_name.startswith("Match lowest high"):
                setpoint = min(comfort_sequence["t_up"])
        else:
            print("Unknown setpoint")
            print(action_name)
        return setpoint


if __name__ == "__main__":
    for i in [0.40, 0.41]:
        output = denormalise_input(i, 5, 40)
        print(i)
        print(output)
        print()
