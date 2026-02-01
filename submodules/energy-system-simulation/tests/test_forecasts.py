import datetime
import simugrid.assets as assets
from simugrid.simulation import Node, Microgrid
from simugrid.simulation.config_parser import parse_config_file
from simugrid.rewards.reward import DefaultReward
from simugrid.management.rational import RationalManager
from simugrid.utils import ROOT_DIR


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
            "name": "SolarPv_0",
        },
        "Asset_1": {
            "node_number": "0",
            "name": "PublicGrid_0",
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
            "buy_from_grid": "0.5",
            "sell_to_grid": "0.5",
            "irradiation": "./example/mordor/environment/irradiation_24h.csv",
        }
    },
}


class TestEnvironment:
    def setup_method(self):
        self.m = parse_config_file(json_config)
        reward = DefaultReward()
        self.m.set_reward(reward)
        RationalManager(self.m)

    def test_get_forecast_env_perfect_quality(self):
        start_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2000, 1, 2, 0, 0, 0)
        # UTC timezone
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        value_object = self.m.nodes[0].environment.env_values.get("irradiation")
        forecast = value_object.get_forecast(start_time, end_time, quality="perfect")
        tenth_value = forecast["values"][9]

        assert tenth_value == 146.55

    def test_get_forecast_env_naive_quality(self):
        start_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2000, 1, 2, 0, 0, 0)
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        value_object = self.m.nodes[0].environment.env_values.get("irradiation")
        forecast = value_object.get_forecast(start_time, end_time, quality="naive")
        tenth_value = forecast["values"][9]
        assert tenth_value == 16.7

    def test_get_forecast_asset_perfect_quality(self):
        start_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2000, 1, 2, 0, 0, 0)
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        value_object = (
            self.m.nodes[0]
            .assets[0]
            .get_forecast(start_time, end_time, quality="perfect")
        )
        tenth_value = value_object["electric_power"]["values"][9]
        assert tenth_value == 0.11724000000000001

    def test_get_forecast_asset_naive_quality(self):
        start_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2000, 1, 2, 0, 0, 0)
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        value_object = (
            self.m.nodes[0]
            .assets[0]
            .get_forecast(start_time, end_time, quality="naive")
        )

        tenth_value = value_object["electric_power"]["values"][9]

        assert tenth_value == 0.01336


if __name__ == "__main__":
    a = TestEnvironment()
    a.setup_method()
    a.test_get_forecast_env_perfect_quality()
    a.test_get_forecast_env_naive_quality()
    a.test_get_forecast_asset_perfect_quality()
    a.test_get_forecast_asset_naive_quality()
