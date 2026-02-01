from simugrid.assets.asset import Asset
import numpy as np
from simugrid.simulation.power import Power

import datetime

W_DENS = 1000  # Water density (kg/m³)
W_SPEC = 4186  # Water specific heat (J/kg/K)


class WaterHeater(Asset):
    def __init__(self, node, name):
        super().__init__(node, name)
        self.t_in = 15  # Incoming water temperature (°C)
        self.r_value = 10  # Tank insulation resistance (found values ranging from 2.2 to 24) (Km²/W)
        # Radius and height for a 150L tank
        self.radius = 0.257  # Tank inside radius (m)
        self.height = 0.72  # Tank inside height (m)
        self.volume = np.pi * self.radius**2 * self.height  # Tank water volume  (m³)
        self._surface = (
            2 * np.pi * self.radius * self.height + 2 * np.pi * self.radius**2
        )  # Tank inside surface (m²)

        self.t_tank = 60  # Tank temperature (°C)

        self.max_consumption_power = 2  # Maximum power (W)
        self.efficiency = 1  # Output heat power/ input electrical power (W/W)

        self.low_setpoint = 60  # Lower setpoint temperature (°C)
        self.high_setpoint = None  # Higher setpoint temperature (°C)

    @property
    def environment_keys(self):
        return {
            "hot_water_flow": [f"{self.name}_flow"],
        }

    def set_attributes(self, var_dict):
        super().set_attributes(var_dict)

        calc_volume = np.pi * self.radius**2 * self.height

        height_radius_given = (
            "radius" in var_dict.keys() and "height" in var_dict.keys()
        )
        volume_given = "volume" in var_dict.keys()
        if self.volume != calc_volume:
            if height_radius_given and not volume_given:
                self.volume = calc_volume

            else:
                # Comes from two equations, new_radius/new_height = self.radius/self.height
                # and np.pi * new_radius**2 * new_height = self.volume
                # <=> new_height = self.volume / (np.pi * new_radius**2)
                # so new_radius = self.radius / self.height * new_height
                # <=> new_radius = self.radius / self.height * self.volume / (np.pi * new_radius**2)
                # <=> new_radius**3 = (self.radius / self.height * self.volume / np.pi)

                new_radius = ((self.radius / self.height) * self.volume / np.pi) ** (
                    1 / 3
                )
                new_height = self.height * new_radius / self.radius

                self.radius = new_radius
                self.height = new_height
        self._surface = (
            2 * np.pi * self.radius * self.height + 2 * np.pi * self.radius**2
        )

    def calc_end_temp(self, init_t_tank, flow, power_w, cur_datetime, time_step):
        """
        power: Power in W

        Calculation source:
        https://doi.org/10.1109/PES.2007.386024
        Or earlier:
        https://doi.org/10.1016/0378-7796(95)01011-4
        """

        if self.t_in == "Belgium":
            t_in = calculate_mains_temperature_belgium(cur_datetime)
        else:
            t_in = self.t_in

        T_AIR = 15  # Ambient air temperature (°C)
        W_DENS = 1000  # Water density (kg/m³)
        W_SPEC = 4186  # Water specific heat (J/kg/K)

        G = self._surface / self.r_value  # Conductance tank (W/K)

        B = flow * W_DENS * W_SPEC  # Conductance water (W/K)
        C = self.volume * W_DENS * W_SPEC  # Heat capacity (J/K)
        r_prime = 1 / (G + B)  # Equivalent resistance ? (K/W)
        time_step_s = time_step.total_seconds()

        D = np.exp(-1 / r_prime / C * time_step_s)
        eff = self.efficiency

        new_t_tank = init_t_tank * D + (
            G * r_prime * T_AIR + B * r_prime * t_in + eff * power_w * r_prime
        ) * (1 - D)

        return new_t_tank

    def set_power_limits(self, environment):

        self.power_limit_low = Power(electrical=-self.max_consumption_power)
        self.power_limit_high = Power(electrical=0)

    def power_consequences(self):
        environment = self.parent_node.environment
        flow = environment.env_values[f"{self.name}_flow"].value
        power = -self.power_output.electrical
        cur_datetime = self.parent_node.microgrid.utc_datetime
        time_step = self.parent_node.microgrid.time_step
        init_t_tank = self.t_tank

        power_w = power * 1e3  # Convert to W
        self.t_tank = self.calc_end_temp(
            init_t_tank, flow, power_w, cur_datetime, time_step
        )


def calculate_mains_temperature_belgium(dt):
    """
    Source temperature:
    https://www.engie.be/fr/blog/conseils-energie/impact-temperature-eau-sur-consommation-machine-laver/
    Source sinusoid shaper of mains:
    https://doi.org/10.1016/j.rser.2017.05.229

    "L'hiver la température moyenne de l'eau de distribution est de 11°C, l'été 16°C."
    The sinusoid is calibrated based on this information.
    Calibrated with lowest value on the 15th of January.
    """

    day_year = dt.timetuple().tm_yday
    mains_temp = 13.5 + 3 * np.sin(2 * np.pi / 365 * (day_year - 105))
    return mains_temp


if __name__ == "__main__":
    water_heater = WaterHeater()

    water_heater.t_tank = 60
    water_heater.t_in = "Belgium"
    water_heater.r_value = 24

    new_temp = water_heater.calc_end_temp(
        water_heater.t_tank,
        6.309e-5,
        0,
        datetime.datetime(2000, 1, 1),
        datetime.timedelta(hours=1),
    )

    print(new_temp)
    print(water_heater.volume)
