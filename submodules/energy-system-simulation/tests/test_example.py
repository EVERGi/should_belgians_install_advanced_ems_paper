from simugrid.misc.log_plot_micro import plot_simulation
from simugrid.simulation.config_parser import parse_config_file
from simugrid.rewards.reward import DefaultReward
from simugrid.management.rational import RationalManager
from simugrid.utils import ROOT_DIR
from example.green_energy_park.custom_reward import SizingReward


class TestMordor:
    def setup_method(self):
        # Import Mordor CSV
        config_file = ROOT_DIR + "/example/mordor/microgrid_config.json"
        self.m = parse_config_file(config_file)

        # Create KPI object and add it to microgrid
        reward = DefaultReward()
        self.m.set_reward(reward)

        # Add EMS to microgrid
        RationalManager(self.m)

    def test_simulate(self):
        # Number of simulation timesteps
        tot_steps = 24 * 1

        # Initiate simulation
        for i in range(tot_steps):
            self.m.management_system.simulate_step()


class TestGEP:
    def setup_method(self):
        config_file = ROOT_DIR + "/example/green_energy_park/microgrid_config.json"
        self.m = parse_config_file(config_file)

        # Create KPI object and add it to microgrid
        reward = SizingReward()
        self.m.set_reward(reward)

        # Add EMS to microgrid
        RationalManager(self.m)

    def test_simulate(self):
        tot_steps = 24 * 4 * 7

        # Initiate simulation
        for i in range(tot_steps):
            self.m.management_system.simulate_step()


if __name__ == "__main__":
    test_case = "green_energy_park"
    if test_case == "mordor":
        a = TestMordor()
    elif test_case == "green_energy_park":
        a = TestGEP()

    a.setup_method()
    a.test_simulate()

    plot_simulation(a.m)
