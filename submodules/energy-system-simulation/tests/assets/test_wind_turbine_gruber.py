from typing import Any
from simugrid.assets.public_grid import PublicGrid
from simugrid.assets.wind_turbine_gruber import WindTurbineGruber
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
            "name": "WindTurbineGruber",
            "size": "500",
        },
        "Asset_1": {
            "node_number": "0",
            "name": "PublicGrid_0",
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
            "wind_speed": "5",
            "buy_from_grid": "0.5",
            "sell_to_grid": "0.5",
            "wind_speed_10": "5",
            "wind_speed_100": "10",
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
        assert isinstance(self.m.nodes[0].assets[1], PublicGrid)
        assert isinstance(self.m.nodes[0].assets[0], WindTurbineGruber)


if __name__ == "__main__":
    a = TestSimugrid()
    a.setup_method()
    a.test_simugrid()
