import os
import json


def tmp_2023_config(config_path):
    config_dir = os.path.dirname(config_path)
    tmp_dir = config_dir + "_tmp_2023"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    config_content = json.load(open(config_path))
    config_content["Microgrid"]["start_time"] = config_content["Microgrid"][
        "start_time"
    ].replace("2022", "2023")
    config_content["Microgrid"]["end_time"] = config_content["Microgrid"][
        "end_time"
    ].replace("2023", "2024")
    for asset in config_content["Assets"].values():
        if asset["name"].startswith("EnergyPlus_"):
            asset["epw_weather"] = asset["epw_weather"].replace("2022", "2023")

    # Save the modified config file in the tmp directory
    config_file = os.path.basename(config_path)
    tmp_config_path = os.path.join(tmp_dir, config_file)
    with open(tmp_config_path, "w") as f:
        json.dump(config_content, f, indent=4)

    return tmp_config_path
