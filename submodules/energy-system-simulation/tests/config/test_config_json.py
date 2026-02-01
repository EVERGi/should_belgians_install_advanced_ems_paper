from simugrid.simulation import Microgrid
from simugrid.simulation.config_parser import parse_config_file

simple_config = {
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
            "name": "WindTurbine_0",
            "v_cin": "4",
            "v_cout": "25",
            "v_rated": "10",
            "size": "500",
        },
        "Asset_1": {
            "node_number": "0",
            "name": "PublicGrid_0",
        },
    },
    "Environments": {"Environment_0": {"nodes_number": "0", "wind_speed": "5"}},
}


def test_parse_config_file_json():
    """
    Test if the config file is actually a json file
    """
    simple_microgrid = parse_config_file(simple_config)

    assert isinstance(simple_microgrid, Microgrid)


if __name__ == "__main__":
    test_parse_config_file_json()
