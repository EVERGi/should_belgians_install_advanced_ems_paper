from simugrid.assets.heat_grid import HeatGrid
from simugrid.assets.solar_thermal import SolarThermal
from simugrid.simulation import Node
from simugrid.simulation import Microgrid
from simugrid.simulation.config_parser import parse_config_file
from simugrid.rewards.reward import DefaultReward
from simugrid.management.rational import RationalManager

json_config = {
    "Microgrid": {
        "number_of_nodes": "1",
        "start_time": "01/01/2000 00:00:00",
        "end_time": "07/01/2000 23:00:00",
        "timezone": "UTC",
        "time_step": "01:00:00",
    },
    "Assets": {
        "Asset_0": {
            "node_number": "0",
            "name": "SolarThermal_0",
        },
        "Asset_1": {
            "node_number": "0",
            "name": "HeatGrid_0",
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
            "buy_from_grid": "0.5",
            "sell_to_grid": "0.5",
            "temperature": "20",
            "irradiation": "1",
            "sell_to_heatgrid": "0.5",
            "buy_from_heatgrid": "0.5",
        }
    },
}


class TestSimugrid:
    def setup_method(self):
        self.m = parse_config_file(json_config)
        reward = DefaultReward()
        self.m.set_reward(reward)
        RationalManager(self.m)

    def test_simugrid(self):
        assert isinstance(self.m, Microgrid)
        assert isinstance(self.m.nodes[0], Node)
        assert isinstance(self.m.nodes[0].assets[1], HeatGrid)
        assert isinstance(self.m.nodes[0].assets[0], SolarThermal)

    def run_one_step(self):
        self.m.management_system.simulate_step()


if __name__ == "__main__":
    a = TestSimugrid()
    a.setup_method()
    a.test_simugrid()
    a.run_one_step()
