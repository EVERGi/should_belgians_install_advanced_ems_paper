from simugrid.assets.charger import Charger
from simugrid.assets.public_grid import PublicGrid
from simugrid.simulation import Node
from simugrid.simulation import Microgrid
from simugrid.simulation.config_parser import parse_config_file
from simugrid.rewards.reward import DefaultReward
from simugrid.management.rational import RationalManager

import numpy as np

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
            "name": "Charger_0",
            "efficiency": "1",
            "max_charge_cp": "11",
            "max_discharge_cp": "0",
            "kind": "Normal",
            "ID": "0",
        },
        "Asset_1": {
            "node_number": "0",
            "name": "PublicGrid_0",
        },
    },
    "Environments": {
        "Environment_0": {
            "nodes_number": "0",
            "p_max_0": "11",
            "det_0": "5",
            "soc_i_0": "0.5",
            "soc_f_0": "0.9",
            "capa_0": "40",
            "buy_from_grid": 0.2,
        }
    },
}


class TestCharger:
    def setup_method(self):
        """
        Setup microgrid, environement, etc to make test possible
        """
        self.m = parse_config_file(json_config)
        reward = DefaultReward()
        self.m.set_reward(reward)
        RationalManager(self.m)

        # Get charger object
        self.charger = self.m.nodes[0].assets[0]

    def test_simugrid(self):
        assert isinstance(self.m, Microgrid)
        assert isinstance(self.m.nodes[0], Node)
        assert isinstance(self.m.nodes[0].assets[1], PublicGrid)
        assert isinstance(self.m.nodes[0].assets[0], Charger)

    def test_ev_model(self):
        """
        Test indivdual EV model
        """
        # Use case
        power = 11
        soc = 0.6
        state = {
            "det": 15,
            "size": 40,
            "soc": soc,
            "soc_f": 1,
            "soc_min": soc,
            "max_charge": power,
            "max_discharge": 0,
        }

        # Get next SOC
        soc_f, p = self.charger.ev_model(state, power, soc)

        assert isinstance(soc_f, float)
        assert np.round(soc_f, decimals=4) == 0.8612

    def test_battery_simulator(self):
        """
        Test battery simulator output
        """
        power = 11
        soc = 0.6
        state = {
            "det": 15,
            "size": 40,
            "soc": soc,
            "soc_f": 1,
            "soc_min": soc,
            "max_charge": power,
            "max_discharge": 0,
        }

        # Get SOC boundaries
        bounds = self.charger.battery_simulator(state)

        assert isinstance(bounds["Lower SOC"], list)
        assert isinstance(bounds["Upper SOC"], list)

        assert len(bounds["Lower SOC"]) == len(bounds["Upper SOC"]) == 16

    def test_get_power_limits(self):
        """
        Test get power limits function
        """
        charge, discharge = self.charger.get_power_limits()

        assert isinstance(charge, float)
        assert isinstance(discharge, float)

        assert np.round(charge, decimals=4) == -11
        assert discharge == 0

    def run_one_step(self):
        self.m.management_system.simulate_step()
