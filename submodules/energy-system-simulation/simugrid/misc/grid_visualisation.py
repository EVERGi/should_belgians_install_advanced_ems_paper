from pyvis.network import Network
import os

from simugrid.simulation.config_parser import parse_config_file
from simugrid.utils import DATA_DIR

non_default_images = {
    "BatteryModel": "Battery.png",
    "ConsumerFlex": "Consumer.png",
    "SolarPvLib": "SolarPv.png",
    "WindTurbineGruber": "WindTurbine.png",
}

image_folder = DATA_DIR + "/visualisation/images/"

image_files = os.listdir(image_folder)

GOOD_BIG_SIZE = 40
GOOD_SMALL_SIZE = 10


def attribute_to_size(asset):
    size = asset.size

    if asset.name.startswith("Charger"):
        size = asset.max_charge_cp

    return size


def get_asset_hover_text(asset, asset_size):
    hover_text = ""
    hover_text += f"name: {asset.name}\n"
    hover_text += f"size: {round(asset_size,2)}\n"
    for description, value in asset.init_var.items():
        if description == "size":
            continue
        if value is not None:
            value = round(value, 2)

        hover_text += f"{description}: {value}\n"
    return hover_text


def get_branch_hover_text(branch):
    hover_text = ""
    hover_text += f"Branch {branch.name}\n"
    for description, value in branch.init_var.items():
        if value is not None:
            value = round(value, 2)

        hover_text += f"{description}: {value}\n"
    return hover_text


def node_size_scale(asset_sizes):
    max_size = max(asset_sizes)

    size_scale = GOOD_BIG_SIZE / max_size

    return size_scale


def microgrid_to_visual_format(microgrid):
    visu_info = dict()
    visu_info["Nodes"] = list()
    for i, node in enumerate(microgrid.nodes):
        node_dict = dict()
        visu_info["Nodes"].append(node_dict)

        for asset in node.assets:
            asset_object_str = asset.name.split("_")[0]
            if asset_object_str in non_default_images.keys():
                image_file = non_default_images[asset_object_str]
            else:
                image_file = asset_object_str + ".png"
            asset_size = attribute_to_size(asset)
            asset_hover_text = get_asset_hover_text(asset, asset_size)
            asset_info = {
                "size": asset_size,
                "image_file": image_file,
                "hover_text": asset_hover_text,
            }
            node_dict[asset.name] = asset_info

    visu_info["Branches"] = list()
    for i, branch in enumerate(microgrid.branches):
        branch_info = dict()
        visu_info["Branches"].append(branch_info)
        branch_info["nodes_index"] = branch.nodes_index
        branch_info["hover_text"] = get_branch_hover_text(branch)

    return visu_info


def group_assets(nodes):
    for node_info in nodes:
        encountered_assets = []
        for asset_name, asset_info in sorted(node_info.items()):
            asset_type = asset_name.split("_")[0]
            if asset_type not in encountered_assets:
                encountered_assets.append(asset_type)
                node_info[asset_type] = node_info.pop(asset_name)
            else:
                node_info.pop(asset_name)
                node_info[asset_type]["size"] += asset_info["size"]


def define_sizes(nodes):
    asset_sizes = []
    for node_info in nodes:
        for _, asset_info in sorted(node_info.items()):
            asset_sizes += [asset_info["size"]]
    size_scale = node_size_scale(asset_sizes)
    for node_info in nodes:
        for asset_name, asset_info in sorted(node_info.items()):
            if asset_name.startswith("PublicGrid"):
                asset_info["node_size"] = GOOD_BIG_SIZE
            else:
                asset_info["node_size"] = (
                    asset_info["size"] * size_scale + GOOD_SMALL_SIZE
                )


def visualise_model(visu_info, show_indiv_assets=True):
    g = Network()
    # g.show_buttons(filter_=["configure", "layout", "interaction", "physics", "edges"])

    if not show_indiv_assets:
        group_assets(visu_info["Nodes"])
    define_sizes(visu_info["Nodes"])

    for node_id, node_info in enumerate(visu_info["Nodes"]):
        g.add_node(node_id, size=15, color="#1f77b4", title=f"Node {node_id}")
        add_assets(g, node_id, node_info, show_indiv_assets)

    for branch_info in visu_info["Branches"]:
        g.add_edge(
            branch_info["nodes_index"][0],
            branch_info["nodes_index"][1],
            value=10,
            title=branch_info["hover_text"],
        )

    # Display the network
    g.show("grid.html", local=False)


def add_assets(g, node_id, node_info, show_indiv_assets):
    prev_asset = ""
    in_group = False
    common_node = ""
    for asset_name, asset_info in sorted(node_info.items()):
        asset_type = asset_name.split("_")[0]
        prev_asset_type = prev_asset.split("_")[0]
        if show_indiv_assets:
            if asset_type == prev_asset_type and not in_group:
                in_group = True
                common_node = f"{asset_type}_node_{node_id}"
                g.add_node(common_node, label=" ", size=0)
                add_asset_node(g, asset_name, asset_info, show_indiv_assets)

                g.add_edge(common_node, asset_name)
                g.add_edge(common_node, prev_asset)
                g.add_edge(node_id, common_node)
            elif asset_type == prev_asset_type and in_group:
                add_asset_node(g, asset_name, asset_info, show_indiv_assets)

                g.add_edge(common_node, asset_name)
            elif asset_type != prev_asset_type and in_group:
                in_group = False
                add_asset_node(g, asset_name, asset_info, show_indiv_assets)
            else:
                if prev_asset != "":
                    g.add_edge(node_id, prev_asset)
                add_asset_node(g, asset_name, asset_info, show_indiv_assets)

            prev_asset = asset_name
        else:
            add_asset_node(g, asset_name, asset_info, show_indiv_assets)

            g.add_edge(node_id, asset_name)
    if show_indiv_assets:
        if not in_group:
            g.add_edge(node_id, prev_asset)


def add_asset_node(g, asset_name, asset_info, show_indiv_assets):
    image_path = image_folder + asset_info["image_file"]

    if asset_info["image_file"] in image_files:
        node_size = asset_info["node_size"]
        if show_indiv_assets:
            title = asset_info["hover_text"]
        else:
            asset_size = round(asset_info["size"], 2)
            title = f"name: {asset_name}\nsize: {asset_size}\n"
        g.add_node(
            asset_name,
            label=" ",
            shape="image",
            image=image_path,
            size=node_size,
            title=title,
        )
    else:
        g.add_node(asset_name, label=asset_name)
    return asset_name


if __name__ == "__main__":
    microgrid_config = f"/home/django/Documents/Thesis_MOBI/bootcamp_2023/energy-systems-simulation/example/mordor/microgrid_config.json"
    # microgrid_config = "/home/django/Documents/Thesis_MOBI/InterConnect/smart-houses-ems/data/houses_env/pv_1_cons_7.csv"
    microgrid = parse_config_file(microgrid_config)
    consumption = microgrid.assets[6].size = 500
    pv = microgrid.assets[3].size = 250
    battery = microgrid.assets[0].size = 300
    gas_turbine = microgrid.assets[2]
    microgrid.assets.remove(gas_turbine)
    microgrid.nodes[0].assets.remove(gas_turbine)
    visu_info = microgrid_to_visual_format(microgrid)
    visualise_model(visu_info, show_indiv_assets=False)
