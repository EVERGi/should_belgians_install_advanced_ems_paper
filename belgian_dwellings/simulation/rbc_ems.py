from belgian_dwellings.simulation.custom_classes import HouseManager, DayAheadEngie

from simugrid.assets.energyplus import EnergyPlus
from simugrid.assets.charger import Charger
from simugrid.simulation.config_parser import parse_config_file

import datetime
import matplotlib.pyplot as plt


class RuleBasedManager(HouseManager):
    def __init__(self, microgrid, delta_t_comfort=datetime.timedelta(minutes=0)):
        super().__init__(microgrid)

        self.delta_t_comfort = delta_t_comfort

    def update_control_points(self):
        for asset in self.control_points:
            asset_class = type(asset)
            if issubclass(asset_class, EnergyPlus):
                start_dt = self.microgrid.utc_datetime
                end_dt = start_dt + self.delta_t_comfort + self.microgrid.time_step
                comfort_sequence = asset.get_comfort_sequence(start_dt, end_dt)
                comfort_t_low = max(comfort_sequence["t_low"])
                comfort_t_up = min(comfort_sequence["t_up"])
                self.control_points[asset]["zn0_heating_sp"] = comfort_t_low
                self.control_points[asset]["zn0_cooling_sp"] = comfort_t_up


def execute_rule_base(config_file, delta_t_comfort=None):

    if delta_t_comfort is None:
        delta_t_comfort = datetime.timedelta(minutes=3 * 60)
    microgrid = parse_config_file(config_file)

    reward = DayAheadEngie()
    microgrid.set_reward(reward)
    RuleBasedManager(microgrid, delta_t_comfort)

    end_time = microgrid.end_time
    # end_time = microgrid.start_time + datetime.timedelta(days=30)
    microgrid.attribute_to_log("EnergyPlus_0", "zn0_temp")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_low")
    microgrid.attribute_to_log("EnergyPlus_0", "cur_t_up")
    microgrid.attribute_to_log("Charger_0", "soc")
    microgrid.attribute_to_log("WaterHeater_0", "t_tank")

    while microgrid.utc_datetime < end_time:
        # Print every new week
        if (
            microgrid.utc_datetime.weekday() == 0
            and microgrid.utc_datetime.hour == 0
            and microgrid.utc_datetime.minute == 0
        ):
            pass
            # print(microgrid.utc_datetime)
        microgrid.management_system.simulate_step()

    # print(f"Final rule base opex: {microgrid.tot_reward.KPIs['opex']}")
    # print(f"Final rule base discomfort: {microgrid.tot_reward.KPIs['discomfort']}")

    for asset in microgrid.assets:
        if isinstance(asset, EnergyPlus):
            asset.thread.join()
    return microgrid


def generate_pareto_rbc(config_file):
    delta_t_comf_minutes = [15, 30, 60, 2 * 60, 4 * 60, 12 * 60]
    discomfort = []
    opex = []
    for delta_t_comf_min in delta_t_comf_minutes:
        delta_t_comf = datetime.timedelta(minutes=delta_t_comf_min)
        microgrid = execute_rule_base(config_file, delta_t_comf)
        discomfort.append(microgrid.tot_reward.KPIs["discomfort"])
        opex.append(microgrid.tot_reward.KPIs["opex"])

        for asset in microgrid.assets:
            if isinstance(asset, EnergyPlus):
                asset.thread.join()

    fig, ax = plt.subplots()
    ax.plot(discomfort, opex, marker="o")
    ax.set_xlabel("Discomfort")
    ax.set_ylabel("Opex")
    ax.set_title("Pareto front for rule-based control")
    plt.show()


if __name__ == "__main__":
    config_file = "data/houses_belgium_100/house_5.json"
    # import tracemalloc

    # tracemalloc.start()
    execute_rule_base(config_file)

    # generate_pareto_rbc(config_file)
