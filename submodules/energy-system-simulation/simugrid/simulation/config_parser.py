import csv
from simugrid.simulation import Microgrid, Node, Environment, Branch
import datetime
from simugrid.utils import ROOT_DIR
from importlib import import_module

import simugrid.assets

import os
import json


def strisfloat(string):
    if not isinstance(string, str):
        return False
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


def strisint(string):
    if not isinstance(string, str):
        return False
    try:
        int(string)
        return True
    except (ValueError, TypeError):
        return False


def get_all_asset_classes(assets_dict, custom_class):
    native_asset_modules = find_all_native_assets()

    used_classes = dict()
    for _, asset_info in assets_dict.items():
        asset_name = asset_info["name"].split("_")[0]
        if asset_name in custom_class.keys():
            used_classes[asset_name] = custom_class[asset_name]
        elif asset_name in native_asset_modules.keys():
            asset_module = native_asset_modules[asset_name]
            # from import_path import asset_name
            module = import_module(asset_module)
            used_classes[asset_name] = getattr(module, asset_name)

    return used_classes


def find_all_native_assets():
    native_asset_modules = dict()
    assets_dir_path = os.path.dirname(simugrid.assets.__file__)
    for file in os.listdir(assets_dir_path):
        if file.endswith(".py"):
            module_path = "simugrid.assets." + file.replace(".py", "")

            with open(os.path.join(assets_dir_path, file), "r") as f:
                file_lines = f.readlines()
                for line in file_lines:
                    if line.startswith("class "):
                        class_name = line.split("(")[0].split(" ")[1]
                        native_asset_modules[class_name] = module_path

    return native_asset_modules


def parse_microgrid(microgrid_dict, microgrid, custom_class):

    # Keys names change for json config files
    # To delete once csv are not used anymore
    start_time = "start_time"
    end_time = "end_time"
    time_step = "time_step"
    number_of_nodes = "number_of_nodes"

    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    for form in formats:
        try:
            start_t = datetime.datetime.strptime(microgrid_dict[start_time], form)
            end_t = datetime.datetime.strptime(microgrid_dict[end_time], form)
            break
        except ValueError as e:
            if form == formats[-1]:
                raise e

    datetime_step = datetime.datetime.strptime(microgrid_dict[time_step], "%H:%M:%S")
    time_step = datetime.timedelta(
        hours=datetime_step.hour,
        minutes=datetime_step.minute,
        seconds=datetime_step.second,
    )

    if "Microgrid" not in custom_class.keys():
        microgrid = Microgrid(
            start_t, end_t, time_step, timezone=microgrid_dict["timezone"]
        )
    else:
        microgrid = custom_class["Microgrid"](
            start_t, end_t, time_step, timezone=microgrid_dict["timezone"]
        )

    for _ in range(int(microgrid_dict[number_of_nodes])):
        if "Node" not in custom_class.keys():
            Node(microgrid)
        else:
            custom_class["Node"](microgrid)

    return microgrid


def parse_branch(dict_args, microgrid, custom_class, config_file):
    if "nodes_index" not in dict_args.keys():
        return microgrid

    nodes_index = [int(i) for i in dict_args["nodes_index"].split("to")]

    if "Branch" not in custom_class.keys():
        branch = Branch(nodes_index)
    else:
        branch = custom_class["Branch"](nodes_index)
    attributes = attribute_list_to_dict(dict_args, config_file)
    branch.set_attributes(attributes)
    microgrid.branches.append(branch)
    return microgrid


def parse_branches(branch_dict, microgrid, custom_class, config_file):

    for _, branch_args in branch_dict.items():
        parse_branch(branch_args, microgrid, custom_class, config_file)

    return microgrid


def attribute_list_to_dict(dict_args, config_file, node_number="node_number"):
    attributes = {
        key: None if value == "" else value
        for key, value in dict_args.items()
        if key not in ["name", node_number, "nodes_index"]
    }

    attributes = {
        key: int(value) if strisint(value) else value
        for key, value in attributes.items()
    }
    attributes = {
        key: float(value) if strisfloat(value) and isinstance(value, str) else value
        for key, value in attributes.items()
    }
    attributes = {
        key: str2bool(value) if isinstance(value, str) and isBool(value) else value
        for key, value in attributes.items()
    }

    # Transform string to absolute path if file exists
    attributes = {
        key: (path_to_abs_path(value, config_file) if isinstance(value, str) else value)
        for key, value in attributes.items()
    }
    attributes.pop("", None)

    return attributes


def str2bool(v):
    return v.lower() == "true"


def isBool(v):
    return v.lower() in ["true", "false"]


def path_to_abs_path(path, config_file):
    if path[0] == "/" or path[1:3] == ":/":
        abs_path = path
    elif path[0] == ".":
        if type(config_file) in [list, dict]:
            point_dir = os.getcwd()
        else:
            abs_path = os.path.abspath(config_file)
            abs_path = abs_path.replace("\\", "/")
            point_dir = "/".join(abs_path.split("/")[:-1])

        abs_path = point_dir + path[1:]
    else:
        abs_path = ROOT_DIR + path

    # Check if path exists
    if os.path.exists(abs_path):
        return abs_path
    else:
        return path


def parse_asset(dict_args, microgrid, asset_classes, config_file):
    if "name" not in dict_args.keys():
        return

    # Keys names change for json config files
    # To delete once csv are not used anymore
    node_number = "node_number"
    if node_number not in dict_args:
        node_number = "node_number"

    class_name = dict_args["name"].split("_")[0]

    asset_class = asset_classes[class_name]

    node_num = int(dict_args[node_number])
    asset = asset_class(microgrid.nodes[node_num], dict_args["name"])

    attributes = attribute_list_to_dict(dict_args, config_file, node_number)
    asset.set_attributes(attributes)


def parse_assets(assets_dict, microgrid, custom_class, config_file):

    asset_classes = get_all_asset_classes(assets_dict, custom_class)
    for _, asset_args in assets_dict.items():
        parse_asset(asset_args, microgrid, asset_classes, config_file)

    return microgrid


def parse_environment(env_args, microgrid, config_file):

    new_env = Environment(microgrid)
    env_nodes = []

    for env_key, env_value in env_args.items():
        if env_key == "nodes_number":
            if isinstance(env_value, str):
                node_nums = env_value.split(",")
            elif isinstance(env_value, list):
                node_nums = env_value
            elif isinstance(env_value, int):
                node_nums = [env_value]
            env_nodes = [int(node) for node in node_nums]

        elif env_key[-3:] == ".py":
            abs_path = path_to_abs_path(env_key, config_file)
            new_env.add_function_values(abs_path)

        elif env_key[-4:] == ".csv":
            abs_path = path_to_abs_path(env_key, config_file)
            new_env.add_multicolumn_csv_values(abs_path)
        elif strisfloat(env_value):
            new_env.add_value(env_key, float(env_value))
        elif not isinstance(env_value, str):
            new_env.add_value(env_key, env_value)
        elif env_value[-4:] == ".csv":
            abs_path = path_to_abs_path(env_value, config_file)
            new_env.add_value(env_key, abs_path)

    for node_ind in env_nodes:
        microgrid.nodes[node_ind].set_environment(new_env)


def parse_environments(environments_dict, microgrid, config_file):

    for _, environment_args in environments_dict.items():
        parse_environment(environment_args, microgrid, config_file)

    return microgrid


def set_model_all_assets(microgrid):
    for node in microgrid.nodes:
        for asset in node.assets:
            asset.check_and_set_model()


def parse_config_file(config_file, custom_class=dict()):
    """
    :ivar config_file: the path to config_file
    :type config_file: string
    :ivar custom_class: the custom classes for microgrid, node, assets
    :type custom_class: dict

    :return: the parsed microgrid
    :rtype: Microgrid
    """

    microgrid = None

    if type(config_file) == list:
        config_dict = config_list_to_config_dict(config_file)

    elif type(config_file) == dict:
        config_dict = config_file
    elif config_file.endswith(".csv"):
        with open(config_file) as csvfile:
            csvreader = csv.reader(csvfile)
            config_dict = config_list_to_config_dict(list(csvreader))
    elif config_file.endswith(".json"):
        with open(config_file) as jsonfile:
            config_dict = json.load(jsonfile)

    else:
        raise Exception("Config file is not a string path neither a list.")

    microgrid = parse_microgrid(config_dict["Microgrid"], microgrid, custom_class)

    if "Branches" in config_dict:
        microgrid = parse_branches(
            config_dict["Branches"], microgrid, custom_class, config_file
        )
    if "Assets" in config_dict:
        microgrid = parse_assets(
            config_dict["Assets"], microgrid, custom_class, config_file
        )

    if "Environments" in config_dict:
        microgrid = parse_environments(
            config_dict["Environments"], microgrid, config_file
        )

    set_model_all_assets(microgrid)

    microgrid.env_simulate()

    return microgrid


def config_list_to_config_dict(config_list):
    sections = ["Microgrid", "Branches", "Assets", "Environments"]
    subsection = ["Branch", "Asset", "Environment"]

    config_dict = {sec: {} for sec in sections}
    current_section = ""
    for row in config_list:

        # Define current_sections
        if len(row) == 0 or row[0] == "END":
            continue
        elif row[0] in sections:
            current_section = row[0]
            cur_dict = config_dict[current_section]
            continue
        elif row[0].split("_")[0] in subsection:
            config_dict[current_section][row[0]] = {}
            cur_dict = config_dict[current_section][row[0]]
            continue
        elif len(row) == 2:
            cur_dict[row[0]] = row[1]

        elif len(row) == 1 and row[0] != "END":
            cur_dict[row[0]] = None
        else:
            print(row)
            raise Exception("Row in config file not understood")

    return config_dict


if __name__ == "__main__":
    simple_config = {
        "Microgrid": {
            "number_of_nodes": "2",
            "start_time": "01/01/2000 00:00:00",
            "end_time": "07/01/2000 23:00:00",
            "timezone": "UTC",
            "time_step": "01:00:00",
        },
        "Branches": {
            "Branch_0": {"nodes_index": "0to1", "max_power_electrical": 10000}
        },
        "Assets": {
            "Asset_0": {
                "node_number": "0",
                "name": "WindTurbine_0",
                "v_cin": "4",
                "v_cout": "25",
                "v_rated": "10",
                "size": "500",
            },
            "Asset_1": {
                "node_number": "1",
                "name": "PublicGrid_0",
            },
        },
        "Environments": {
            "Environment_0": {
                "nodes_number": "0,1",
                "wind_speed": 5,
                "sell_to_grid": 0.4,
                "buy_from_grid": 0.2,
            }
        },
    }

    m = parse_config_file(simple_config)
