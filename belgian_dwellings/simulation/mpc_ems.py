import os


from belgian_dwellings.simulation.custom_classes import HouseManager
from belgian_dwellings.simulation.forecaster import PerfectForecaster, EasyForcaster
import gurobipy as gp

from gurobipy import GRB

from simugrid.assets.charger import first_order_fit, Charger
from simugrid.assets.energyplus import EnergyPlus
from simugrid.assets.battery import Battery
from simugrid.assets.water_heater import WaterHeater

import datetime
import pytz


class ModelAttributes(object):
    pass


class MPCManager(HouseManager):
    def __init__(self, microgrid, energyplus_calibration):
        super().__init__(microgrid)

        self.model = gp.Model("mpc")  # , env=env)
        self.model.Params.LogToConsole = 0
        self.model.Params.SoftMemLimit = 2
        self.model.Params.Threads = 4
        self.model.Params.TimeLimit = 10
        # self.model.Params.DualReductions = 0
        # self.model.Params.LogFile = "mpc.log"
        # self.model.Params.OutputFlag = 1
        # self.model.Params.InfUnbdInfo = 1
        self.model_att = ModelAttributes()

        self.model_initialized = False

        self.horizon = None

        self.mpc_batteries = []
        self.mpc_chargers = []
        self.mpc_energyplus = []
        self.mpc_waterheaters = []

        self.store_assets()

        self.forecaster = None

        self.energyplus_calibration = energyplus_calibration

        self.monthly_peak = 2.5

        self.fixed_horizon = False

    def store_assets(self):
        for asset in self.microgrid.assets:
            if isinstance(asset, Battery):
                self.mpc_batteries.append(asset)
            elif isinstance(asset, Charger):
                self.mpc_chargers.append(asset)
            elif isinstance(asset, EnergyPlus):
                self.mpc_energyplus.append(asset)
            elif isinstance(asset, WaterHeater):
                self.mpc_waterheaters.append(asset)

    def build_sets(self):
        model_att = self.model_att
        model_att.time = range(self.horizon)
        model_att.batt_id = range(len(self.mpc_batteries))
        model_att.charger_id = range(len(self.mpc_chargers))
        model_att.energyplus_id = range(len(self.mpc_energyplus))
        model_att.waterheater_id = range(len(self.mpc_waterheaters))

    def build_vars(self):
        model = self.model
        model_att = self.model_att

        ub_batt_pos = [batt.max_production_power for batt in self.mpc_batteries]
        model_att.batt_pos = model.addVars(
            model_att.batt_id,
            model_att.time,
            lb=0,
            ub=ub_batt_pos,
            vtype=GRB.CONTINUOUS,
            name="batt_pos",
        )

        lb_batt_neg = [-batt.max_consumption_power for batt in self.mpc_batteries]
        model_att.batt_neg = model.addVars(
            model_att.batt_id,
            model_att.time,
            lb=lb_batt_neg,
            ub=0,
            vtype=GRB.CONTINUOUS,
            name="batt_neg",
        )

        model_att.batt_soc = model.addVars(
            model_att.batt_id,
            model_att.time,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="batt_soc",
        )

        ub_charger_power = [
            [charger.max_consumption_power for _ in model_att.time]
            for charger in self.mpc_chargers
        ]
        model_att.charger_power = model.addVars(
            model_att.charger_id,
            model_att.time,
            lb=0,
            ub=ub_charger_power,
            vtype=GRB.CONTINUOUS,
            name="charger_power",
        )
        model_att.ev_soc = model.addVars(
            model_att.charger_id,
            model_att.time,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="ev_soc",
        )

        eplus_specs = [eplus.get_specs() for eplus in self.mpc_energyplus]

        cool_eff = [eplus_cal["cool_eff"] for eplus_cal in self.energyplus_calibration]
        ub_eplus_cooling = [
            [eplus_spec["cooling_power_w"] / 1e3 / cool_eff[i] for _ in model_att.time]
            for i, eplus_spec in enumerate(eplus_specs)
        ]
        # lb_eplus_cooling = [0 for _ in ub_eplus_cooling]

        model_att.energyplus_cooling = model.addVars(
            model_att.energyplus_id,
            model_att.time,
            lb=0,
            ub=ub_eplus_cooling,
            vtype=GRB.CONTINUOUS,
            name="energyplus_cooling",
        )

        heat_eff = [eplus_cal["main_eff"] for eplus_cal in self.energyplus_calibration]
        ub_eplus_heating = [
            [eplus_spec["heating_power_w"] / 1e3 / heat_eff[i] for _ in model_att.time]
            for i, eplus_spec in enumerate(eplus_specs)
        ]
        model_att.energyplus_heating = model.addVars(
            model_att.energyplus_id,
            model_att.time,
            lb=0,
            ub=ub_eplus_heating,
            vtype=GRB.CONTINUOUS,
            name="energyplus_heating",
        )

        backup_eff = [
            eplus_cal["backup_eff"] for eplus_cal in self.energyplus_calibration
        ]
        ub_eplus_backup_heating = [
            [eplus_spec["backup_power_w"] / 1e3 / backup_eff[i] for _ in model_att.time]
            for i, eplus_spec in enumerate(eplus_specs)
        ]
        model_att.energyplus_backup_heating = model.addVars(
            model_att.energyplus_id,
            model_att.time,
            lb=0,
            ub=ub_eplus_backup_heating,
            vtype=GRB.CONTINUOUS,
            name="energyplus_backup_heating",
        )

        model_att.indoor_temp = model.addVars(
            model_att.energyplus_id,
            model_att.time,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="indoor_temp",
        )

        ub_waterheater_power = [
            [waterheater.max_consumption_power for _ in model_att.time]
            for waterheater in self.mpc_waterheaters
        ]
        model_att.waterheater_power = model.addVars(
            model_att.waterheater_id,
            model_att.time,
            lb=0,
            ub=ub_waterheater_power,
            vtype=GRB.CONTINUOUS,
            name="waterheater_power",
        )

        ub_tank_temp = [
            [water_heater.high_setpoint for _ in model_att.time]
            for water_heater in self.mpc_waterheaters
        ]

        model_att.tank_temp = model.addVars(
            model_att.waterheater_id,
            model_att.time,
            lb=0,
            ub=ub_tank_temp,
            vtype=GRB.CONTINUOUS,
            name="tank_temp",
        )

        model_att.heater_condition = model.addVars(
            model_att.waterheater_id,
            model_att.time,
            vtype=GRB.BINARY,
            name="heater_condition",
        )

        model_att.grid_power = model.addVars(
            model_att.time,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="grid_power",
        )

        # monthly_peak = self.monthly_peak
        model_att.grid_peak = model.addVar(
            lb=2.5, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="grid_peak"
        )
        # Cost variables
        model_att.extra_vat_offtake = model.addVars(
            model_att.time,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="extra_vat_offtake",
        )
        model_att.extra_cost_offtake = model.addVars(
            model_att.time,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="extra_cost_offtake",
        )
        model_att.extra_cost_injection = model.addVars(
            model_att.time,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="extra_cost_injection",
        )
        model_att.trans_dis_cost = model.addVars(
            model_att.time,
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="trans_dis_cost",
        )

        model_att.obj_cost = model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="obj_cost"
        )

    def build_constr(self, retry=0):
        forced_power = [
            -charger.get_powers_to_reach_soc_final()[1] for charger in self.mpc_chargers
        ]
        forced_power = [0]

        forecast = self.forecaster.get_forecast(self.horizon)

        pv_and_load = forecast["pv_and_load"]
        self.set_grid_power_constr(pv_and_load)

        self.set_batt_soc_constr()

        soc_imposed_i = forecast["soc_i"]
        soc_imposed_f = forecast["soc_f"]
        det = forecast["det"]
        cap = forecast["cap"]
        p_max = forecast["p_max"]

        self.set_charger_soc_constr(
            soc_imposed_i, soc_imposed_f, det, cap, forced_power
        )

        self.set_charger_power_curve_constr(p_max, det, forced_power)
        self.set_cost_constr()
        self.set_grid_peak_constr()

        outdoor_temp = forecast["outdoor_temperature"]
        solar_irradiation = forecast["solar_radiation"]
        upper_bound = forecast["upper_bound"]
        lower_bound = forecast["lower_bound"]
        self.set_indoor_temp_constr(
            outdoor_temp, solar_irradiation, retry, upper_bound, lower_bound
        )
        self.set_comfort_bounds_constr(lower_bound, upper_bound)

        flow = forecast["flow"]
        self.set_waterheater_constr(flow)

    def retry_temp_adjust(self, upper_bound, lower_bound, retry, indoor_temp):
        low_index = len(lower_bound)
        up_index = len(upper_bound)
        for i, low_lim in enumerate(lower_bound):
            if low_lim > indoor_temp:
                low_index = i
                break
        for i, up_lim in enumerate(upper_bound):
            if up_lim < indoor_temp:
                up_index = i
                break
        if low_index < up_index:
            new_indoor_temp = indoor_temp + 0.1 * retry
        else:
            new_indoor_temp = indoor_temp - 0.1 * retry

        return new_indoor_temp

    def set_grid_power_constr(self, pv_and_load):
        model = self.model
        model_att = self.model_att

        time = model_att.time
        batt_id = model_att.batt_id
        energyplus_id = model_att.energyplus_id
        waterheater_id = model_att.waterheater_id

        for t in time:
            bat_power = sum(
                model_att.batt_pos[b, t] + model_att.batt_neg[b, t] for b in batt_id
            )
            charger_power = sum(
                model_att.charger_power[c, t] for c in model_att.charger_id
            )
            energyplus_power = sum(
                model_att.energyplus_cooling[e, t]
                + model_att.energyplus_heating[e, t]
                + model_att.energyplus_backup_heating[e, t]
                for e in energyplus_id
            )
            water_heater_power = sum(
                model_att.waterheater_power[w, t] for w in waterheater_id
            )
            pv_and_load_power = pv_and_load[t]

            model.addConstr(
                model_att.grid_power[t]
                == bat_power
                - charger_power
                - energyplus_power
                - water_heater_power
                + pv_and_load_power,
                name=f"set_grid_power{t}",
            )

    def set_grid_peak_constr(self):
        model = self.model
        model_att = self.model_att

        model.addConstr(
            model_att.grid_peak >= self.monthly_peak * 0.9, name="calc_grid_peak"
        )

        for t in model_att.time:
            model.addConstr(
                model_att.grid_peak >= -model_att.grid_power[t], name="calc_grid_peak"
            )

    def set_batt_soc_constr(self):
        model_att = self.model_att
        model = self.model

        batt_capacity = [batt.size for batt in self.mpc_batteries]
        batt_charge_eff = [batt.charge_eff for batt in self.mpc_batteries]
        batt_discharge_eff = [batt.discharge_eff for batt in self.mpc_batteries]
        soc_init = [batt.soc for batt in self.mpc_batteries]
        time_step = self.microgrid.time_step
        dt_h = time_step.total_seconds() / 3600

        for b in model_att.batt_id:
            for t in model_att.time:
                if t == 0:
                    soc_prev = soc_init[b]
                else:
                    soc_prev = model_att.batt_soc[b, t - 1]
                model.addConstr(
                    model_att.batt_soc[b, t]
                    == soc_prev
                    + (
                        model_att.batt_pos[b, t] * batt_discharge_eff[b]
                        + model_att.batt_neg[b, t] / batt_charge_eff[b]
                    )
                    * dt_h
                    / batt_capacity[b]
                )

    def set_charger_soc_constr(
        self,
        soc_imposed_i,
        soc_imposed_f,
        det,
        cap,
        forced_power,
    ):
        model_att = self.model_att
        model = self.model
        dt_h = self.microgrid.time_step.total_seconds() / 3600

        eff = [charger.eff for charger in self.mpc_chargers]

        for c in model_att.charger_id:

            for t in model_att.time:

                if soc_imposed_f[c][t] > 0.01:
                    model.addConstr(model_att.ev_soc[c, t] == soc_imposed_f[c][t])
                elif det[c][t] == 0:
                    model.addConstr(model_att.ev_soc[c, t] == 0)

                if t == 0:
                    model.addConstr(model_att.charger_power[c, t] >= forced_power[c])

                set_soc_i = False
                if soc_imposed_i[c][t] > 0.001:
                    soc_i = soc_imposed_i[c][t]
                    set_soc_i = True
                elif det[c][t] != 0:
                    soc_i = model_att.ev_soc[c, t - 1]
                    set_soc_i = True

                if set_soc_i:
                    model.addConstr(
                        model_att.ev_soc[c, t]
                        == soc_i
                        + model_att.charger_power[c, t] * dt_h * eff[c] / cap[c][t]
                    )

    def set_charger_power_curve_constr(self, p_max, det, forced_power):
        model_att = self.model_att
        model = self.model

        for c in model_att.charger_id:
            if not self.mpc_chargers[c].charge_curve:
                continue
            pow = [p_max[c][t], 1.4]
            soc = [0.9, 1.0]
            a, b = first_order_fit(soc, pow)
            charge_is_enforced = forced_power[c] >= 0.01
            for t in model_att.time:
                if det[c][t] != 0 and not (t == 0 and charge_is_enforced):
                    model.addConstr(
                        model_att.charger_power[c, t] <= a * model_att.ev_soc[c, t] + b
                    )

    def set_cost_constr(self):

        time_step = self.microgrid.time_step
        dt_h = time_step.total_seconds() / 3600
        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_ind_cost[s,t]}

        model = self.model
        model_att = self.model_att
        time = model_att.time

        env_values = self.microgrid.environments[0].env_values

        offtake_extra = env_values["offtake_extra"].value
        injection_extra = env_values["injection_extra"].value
        kwh_offtake_cost = env_values["kwh_offtake_cost"].value
        capacity_tariff = env_values["capacity_tariff"].value

        start_dt = self.microgrid.utc_datetime
        end_dt = start_dt + time_step * self.horizon
        day_ahead = env_values["day_ahead_price"].get_forecast(start_dt, end_dt)[
            "values"
        ]

        for t in time:
            if day_ahead[t] + offtake_extra >= 0:
                model.addConstr(
                    model_att.extra_vat_offtake[t]
                    >= -model_att.grid_power[t]
                    * (day_ahead[t] + offtake_extra)
                    * dt_h
                    * 0.06,
                    name="vat_offtake",
                )
            else:
                model.addConstr(model_att.extra_vat_offtake[t] == 0, name="vat_offtake")

            model.addConstr(
                model_att.extra_cost_offtake[t]
                >= -model_att.grid_power[t] * dt_h * offtake_extra,
                name="offtake_extra",
            )

            model.addConstr(
                model_att.extra_cost_injection[t]
                >= model_att.grid_power[t] * dt_h * injection_extra,
                name="injection_extra",
            )

            model.addConstr(
                model_att.trans_dis_cost[t]
                >= -model_att.grid_power[t] * dt_h * kwh_offtake_cost,
                name="trans_dis_calc",
            )
        day_ahead_cost = sum(
            -model_att.grid_power[t] * dt_h * day_ahead[t] for t in time
        )
        tot_extra_cost = sum(
            model_att.extra_vat_offtake[t]
            + model_att.extra_cost_offtake[t]
            + model_att.extra_cost_injection[t]
            for t in time
        )
        day_ahead_cost += tot_extra_cost

        trans_dis_cost = sum(model_att.trans_dis_cost[t] for t in time)

        cap_cost = model_att.grid_peak * capacity_tariff

        model.addConstr(
            model_att.obj_cost == day_ahead_cost + trans_dis_cost + cap_cost,
            name="objective",
        )

    def set_indoor_temp_constr(
        self, outdoor_temp, solar_irradiation, retry, upper_bound, lower_bound
    ):
        """
        Source:
        https://doi.org/10.1016/j.arcontrol.2020.09.001
        """
        model = self.model
        model_att = self.model_att
        time = model_att.time

        init_temp = []
        for i, eplus in enumerate(self.mpc_energyplus):
            if eplus._saved_readings is None:
                temp = 20
            else:

                temp = eplus._saved_readings["indoor_temp_C"]
                if retry > 0:
                    temp = self.retry_temp_adjust(
                        upper_bound[i],
                        lower_bound[i],
                        retry,
                        temp,
                    )

            init_temp.append(temp)

        time_step = self.microgrid.time_step
        dt_s = time_step.total_seconds()

        eplus_cal = self.energyplus_calibration
        backup_eff = [eplus_spec["backup_eff"] for eplus_spec in eplus_cal]
        heat_eff = [eplus_spec["main_eff"] for eplus_spec in eplus_cal]
        cool_eff = [eplus_spec["cool_eff"] for eplus_spec in eplus_cal]
        gA = [eplus_spec["ga"] for eplus_spec in eplus_cal]
        therm_cap = [eplus_spec["therm_cap"] for eplus_spec in eplus_cal]
        therm_res = [eplus_spec["therm_res"] for eplus_spec in eplus_cal]

        for e in model_att.energyplus_id:
            for t in time:

                thermal_power = (
                    model_att.energyplus_backup_heating[e, t] * backup_eff[e]
                    + model_att.energyplus_heating[e, t] * heat_eff[e]
                    - model_att.energyplus_cooling[e, t] * cool_eff[e]
                ) * 1000  # To bring it to watts

                if t == 0:
                    previous_temp = init_temp[e]
                else:
                    previous_temp = model_att.indoor_temp[e, t - 1]

                model.addConstr(
                    model_att.indoor_temp[e, t]
                    == self.temp_function(
                        previous_temp,
                        thermal_power,
                        solar_irradiation[t],
                        outdoor_temp[t],
                        gA[e],
                        therm_cap[e],
                        therm_res[e],
                        dt_s,
                    )
                )
                """
                    previous_temp
                    + dt_s
                    / therm_cap[e]
                    * (
                        thermal_power
                        + solar_irradiation[t] * gA[e]
                        + (outdoor_temp[t] - previous_temp) / therm_res[e]
                    )
                )
                """

    @staticmethod
    def temp_function(
        previous_temp,
        thermal_power,
        solar_irradiation,
        outdoor_temp,
        gA,
        therm_cap,
        therm_res,
        dt_s,
    ):
        return previous_temp + dt_s / therm_cap * (
            thermal_power
            + solar_irradiation * gA
            + (outdoor_temp - previous_temp) / therm_res
        )

    def set_comfort_bounds_constr(self, lower_bound, upper_bound):
        model = self.model
        model_att = self.model_att
        time = model_att.time

        for e in model_att.energyplus_id:
            for t in time:
                model.addConstr(model_att.indoor_temp[e, t] <= upper_bound[e][t])
                model.addConstr(model_att.indoor_temp[e, t] >= lower_bound[e][t])

    def set_waterheater_constr(self, flow):
        model = self.model
        model_att = self.model_att

        time_step = self.microgrid.time_step
        cur_datetime = self.microgrid.utc_datetime

        eps = 1e-10

        for w in model_att.waterheater_id:
            water_heater = self.mpc_waterheaters[w]

            BOILING_POINT = 100
            max_abs_temp = BOILING_POINT
            M = max_abs_temp + eps
            for t in model_att.time:

                power = model_att.waterheater_power[w, t]
                if t == 0:
                    # To avoid constraint errors when the water heater was heated too much
                    init_t_tank = float(
                        min(water_heater.t_tank, water_heater.high_setpoint)
                    )
                else:
                    init_t_tank = model_att.tank_temp[w, t - 1]
                power_w = power * 1e3  # Convert to W
                new_temp = water_heater.calc_end_temp(
                    init_t_tank, flow[w][t], power_w, cur_datetime, time_step
                )
                model.addConstr(
                    model_att.tank_temp[w, t] == new_temp,
                    name="update_tank_temp",
                )
                # Source:
                # https://support.gurobi.com/hc/en-us/articles/4414392016529-How-do-I-model-conditional-statements-in-Gurobi
                b = model_att.heater_condition[w, t]
                model.addConstr(
                    init_t_tank >= water_heater.low_setpoint + eps - M * (1 - b),
                    name="water_bigM_constr1",
                )
                model.addConstr(
                    init_t_tank <= water_heater.low_setpoint + M * b,
                    name="water_bigM_constr2",
                )

                # Add indicator constraints
                model.addConstr(
                    model_att.waterheater_power[w, t]
                    >= water_heater.max_consumption_power * (1 - b),
                    name="heat_water_below_setpoint",
                )

            cur_datetime += time_step

    def build_obj(self):
        model = self.model
        model_att = self.model_att
        model.setObjective(model_att.obj_cost, GRB.MINIMIZE)

    def model_remove_constr(self):
        for c in self.model.getConstrs():
            self.model.remove(c)

    def model_remove_vars(self):
        for v in self.model.getVars():
            self.model.remove(v)

    def model_remove_all(self):
        self.model_remove_constr()
        self.model_remove_vars()

    def build_whole_model(self, retry=0):

        self.build_sets()
        self.build_vars()
        self.build_constr(retry)
        self.build_obj()

    def build_model(self, retry=0):
        if not self.model_initialized:
            self.build_whole_model(retry)
            self.model_initialized = True
        else:
            if self.fixed_horizon:
                self.model_remove_constr()
                self.build_constr(retry)
            else:
                self.model_remove_all()
                self.build_whole_model(retry)

    def update_control_points(self):
        self.update_grid_peak()
        self.update_horizon()

        self.build_model()

        self.model.optimize()
        status = self.model.status
        retry = 0

        if status == GRB.TIME_LIMIT:
            print("Time limit reached. Using non optimal solution.")
        elif status == GRB.MEM_LIMIT:
            print("Memory limit reached. Using non optimal solution.")

        while status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT]:
            retry += 1

            self.build_model(retry)
            self.model.optimize()
            status = self.model.status

            # print("Model not optimal")
            if retry > 40:
                print(f"Model not optimal. Retry number: {retry}")
                # If config_file is an attribute of microgrid
                microgrid = self.microgrid
                if hasattr(microgrid, "config_file"):
                    day = microgrid.utc_datetime.day
                    month = microgrid.utc_datetime.month
                    year = microgrid.utc_datetime.year
                    file_name = f"debug_{microgrid.config_file}_{year}_{month}_{day}.lp"
                    debug_dir = "debug_mpc"
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    self.model.write(f"{debug_dir}/{file_name}")
                else:
                    self.model.write("debug_mpc.lp")
                return

        grid_power = self.model_att.grid_power[0].x
        # print(grid_power)

        for b, battery in enumerate(self.mpc_batteries):
            batt_power = (
                self.model_att.batt_pos[b, 0].x + self.model_att.batt_neg[b, 0].x
            )
            self.control_points[battery] = {"power_sp": batt_power}
        for c, charger in enumerate(self.mpc_chargers):
            charger_power = self.model_att.charger_power[c, 0].x
            self.control_points[charger] = {"power_sp": -charger_power}
        for e, energyplus in enumerate(self.mpc_energyplus):
            cooling_power = self.model_att.energyplus_cooling[e, 0].x
            heating_power = self.model_att.energyplus_heating[e, 0].x
            backup_heating_power = self.model_att.energyplus_backup_heating[e, 0].x
            energyplus_power = cooling_power + heating_power + backup_heating_power
            mpc_temp = self.model_att.indoor_temp[e, 0].x

            if energyplus._saved_readings is None:
                current_temp = 20
            else:
                current_temp = energyplus._saved_readings["indoor_temp_C"]

            """
            print(self.microgrid.utc_datetime)
            print(f"MPC temp: {mpc_temp}")
            print(f"Current temp: {current_temp}")
            print(f"Cooling power: {cooling_power}")
            print(f"Heating power: {heating_power}")
            print(f"Backup heating power: {backup_heating_power}")
            """

            if heating_power > 0.001:
                temp_setpoint = {
                    "zn0_heating_sp": mpc_temp,
                    "zn0_cooling_sp": 40,
                }
            elif cooling_power > 0.001:
                temp_setpoint = {
                    "zn0_heating_sp": 5,
                    "zn0_cooling_sp": mpc_temp,
                }
            else:
                temp_setpoint = {
                    "zn0_heating_sp": 5,
                    "zn0_cooling_sp": 40,
                }
            log_mpc = False
            if log_mpc:
                with open("energyplus_log.csv", "a+") as f:
                    f.write(
                        f"{self.microgrid.utc_datetime},{cooling_power},{heating_power},{backup_heating_power},{mpc_temp},{current_temp}\n"
                    )
            # print(temp_setpoint)
            self.control_points[energyplus] = temp_setpoint

        for w, waterheater in enumerate(self.mpc_waterheaters):
            power_sp = self.model_att.waterheater_power[w, 0].x
            self.control_points[waterheater] = {"power_sp": -power_sp}
            #  print(power_sp)
            power_setpoints = [
                self.model_att.waterheater_power[w, t].x for t in self.model_att.time
            ]
            # print(f"Setpoints: {power_setpoints}")
            t_tank = [self.model_att.tank_temp[w, t].x for t in self.model_att.time]
            # print(f"Temperature tank: {t_tank}")

    def update_grid_peak(self):
        cur_time = self.microgrid.utc_datetime
        if (
            cur_time.day == 1 and cur_time.hour == 0 and cur_time.minute == 0
        ) or cur_time == self.microgrid.start_time:
            self.monthly_peak = 2.5
        else:
            grid_power = self.microgrid.power_hist[0]["PublicGrid_0"][-1].electrical
            self.monthly_peak = max(self.monthly_peak, grid_power)

    def update_horizon(self):
        pass


class PerfectMPC(MPCManager):
    def __init__(self, microgrid, energyplus_calibration):
        super().__init__(microgrid, energyplus_calibration)

        self.forecaster = PerfectForecaster(self.microgrid)
        self.fixed_horizon = True

    def update_horizon(self):
        dt_h = self.microgrid.time_step.total_seconds() / 3600
        self.horizon = int(48 / dt_h)


class MPCRealistic(MPCManager):
    def __init__(self, microgrid, energyplus_calibration, ev_forecating_values=None):
        super().__init__(microgrid, energyplus_calibration)

        self.forecaster = EasyForcaster(self.microgrid, ev_forecating_values)

    def update_horizon(self):
        utc_datetime = self.microgrid.utc_datetime
        time_step = self.microgrid.time_step
        local_tz = pytz.timezone("Europe/Brussels")
        local_time = utc_datetime.astimezone(local_tz)

        start_day = local_time.replace(hour=0, minute=0, second=0)

        # Hour of the day when the day-ahead prices are available for the next day
        DAY_AHEAD_HOUR = 14
        if local_time.hour < DAY_AHEAD_HOUR:
            end_horizon = start_day + datetime.timedelta(days=1)
        else:
            end_horizon = start_day + datetime.timedelta(days=2)

        horizon_timedelta = end_horizon - local_time
        # Remove if no more bugs
        # debug = False
        # if debug:
        # For debugging purposes, set lower horizon
        #    self.horizon = 30
        # else:
        #    self.horizon = horizon_timedelta // time_step
        self.horizon = int(horizon_timedelta // time_step)
        self.model_att.time = range(self.horizon)


if __name__ == "__main__":
    print("Hello world")
    pass
