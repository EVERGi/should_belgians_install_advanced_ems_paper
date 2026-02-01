import random
import csv
from copy import deepcopy
import pandas as pd


def get_dwelling_stats():
    dwelling_stats = dict()

    """
    Source:
    https://statbel.fgov.be/fr/themes/census/logement/caracteristiques
    """
    dwelling_stats["region"] = {
        "Brussels": 525_706,
        "Flanders": 2_839_780,
        "Wallonia": 1_580_892,
    }

    dwelling_stats["heating ?"] = {
        "Brussels": {"central heating": 444_794, "no central heating": 20_780},
        "Flanders": {"central heating": 2_430_610, "no central heating": 260_141},
        "Wallonia": {"central heating": 1_232_982, "no central heating": 206_332},
    }

    dwelling_stats["rooms"] = {
        "Brussels": {
            "1 room": 14_540,
            "2 room": 38_246,
            "3 room": 103_457,
            "4 room": 154_051,
            "5 room": 86_205,
            "6 room": 46_246,
            "7 room": 26_502,
            "8 room": 18_504,
            "9 room": 31_391,
        },
        "Flanders": {
            "1 room": 25_793,
            "2 room": 77_791,
            "3 room": 224_807,
            "4 room": 490_955,
            "5 room": 689_152,
            "6 room": 618_500,
            "7 room": 357_395,
            "8 room": 179_064,
            "9 room": 157_812,
        },
        "Wallonia": {
            "1 room": 18_753,
            "2 room": 51_429,
            "3 room": 122_240,
            "4 room": 251_355,
            "5 room": 362_073,
            "6 room": 345_979,
            "7 room": 202_510,
            "8 room": 114_841,
            "9 room": 100_663,
        },
    }

    """
    Source:
    https://doi.org/10.1016/j.rser.2021.112035
    This paper used smart meter data from 10300 apartments in Belgium.
    It used as a metric bedrooms and it had 38.6% with 1 (or 0) bedroom, 40.4% with 2 bedrooms, 15.2% with 3 bedrooms, 4.2% with 4 bedrooms, 1.1% with 5 or more bedrooms and 0.3% with 6 bedrooms.
    We decided on percentage of having a certain number of bedrooms depending on the number of rooms.
    In total this would give for Belgium households where 28.3% have 1 bedroom, 40.3% have 2 bedrooms, 22.4% have 3 bedrooms, 7.5% have 4 bedrooms, 1.1% have 5 or more bedrooms and 0.3% have 6 bedrooms.
    This reduces the number of 1 bedroom houses to the favour of 3 and 4 bedroom houses which would be more common in houses than in apartments.
    """
    dwelling_stats["bedrooms"] = {
        "1 room": {"0 bedroom": 36, "1 bedroom": 74},
        "2 room": {"0 bedroom": 110, "1 bedroom": 203},
        "3 room": {"0 bedroom": 227, "1 bedroom": 518, "2 bedroom": 102},
        "4 room": {
            "0 bedroom": 227,
            "1 bedroom": 519,
            "2 bedroom": 756,
            "3 bedroom": 183,
        },
        "5 room": {
            "0 bedroom": 187,
            "1 bedroom": 425,
            "2 bedroom": 1133,
            "3 bedroom": 363,
            "4 bedroom": 30,
        },
        "6 room": {
            "0 bedroom": 25,
            "1 bedroom": 111,
            "2 bedroom": 945,
            "3 bedroom": 601,
            "4 bedroom": 207,
            "5 bedroom": 7,
        },
        "7 room": {
            "2 bedroom": 473,
            "3 bedroom": 464,
            "4 bedroom": 130,
            "5 bedroom": 27,
            "6 bedroom": 2,
        },
        "8 room": {
            "2 bedroom": 189,
            "3 bedroom": 200,
            "4 bedroom": 122,
            "5 bedroom": 55,
            "6 bedroom": 14,
        },
        "9 room": {
            "2 bedroom": 181,
            "3 bedroom": 192,
            "4 bedroom": 90,
            "5 bedroom": 47,
            "6 bedroom": 31,
        },
    }

    """
    Source:
    https://statbel.fgov.be/sites/default/files/Over_Statbel_FR/Enquete%20SocEco%202001%20-%20Monographie%202%20Le%20logement%20en%20Belgique.pdf
    Tableau IV.24 shows the number of houses per region per surface area of the dwelling.
    Tableau IV.28 shows the number of houses per region per number of rooms.
    Based on these two tables we estimated the relation between the number of rooms and the surface area of the dwelling.
    Some common sense constraintswere used such as smaller area is correlated with fewer rooms.
    The estimation was not perfect but close.
    """
    dwelling_stats["surface area"] = {
        "1 room": {"<35m²": 0.9, "35-54m²": 0.1},
        "2 room": {"<35m²": 0.5, "35-54m²": 0.3, "55-84m²": 0.2},
        "3 room": {"<35m²": 0.1, "35-54m²": 0.3, "55-84m²": 0.4, "85-104m²": 0.2},
        "4 room": {"35-54m²": 0.3, "55-84m²": 0.4, "85-104m²": 0.2, "105-124m²": 0.1},
        "5 room": {
            "35-54m²": 0.2,
            "55-84m²": 0.3,
            "85-104m²": 0.3,
            "105-124m²": 0.1,
            ">124m²": 0.1,
        },
        "6 room": {
            "35-54m²": 0.15,
            "55-84m²": 0.25,
            "85-104m²": 0.3,
            "105-124m²": 0.2,
            ">124m²": 0.1,
        },
        "7 room": {
            "55-84m²": 0.1,
            "85-104m²": 0.3,
            "105-124m²": 0.3,
            ">124m²": 0.3,
        },
        "8 room": {
            "85-104m²": 0.1,
            "105-124m²": 0.3,
            ">124m²": 0.8,
        },
        "9 room": {
            "85-104m²": 0.1,
            "105-124m²": 0.3,
            ">124m²": 0.8,
        },
    }

    """
    Source:

    https://statbel.fgov.be/fr/themes/census/logement/type-de-logement#panel-12
    """
    dwelling_stats["dwelling type"] = {
        "Brussels": {"house": 15 / 100, "apartment": 85 / 100},
        "Flanders": {"house": 71.2 / 100, "apartment": 28.8 / 100},
        "Wallonia": {"house": 74 / 100, "apartment": 26 / 100},
    }

    """ 
    Source garage_prop:
    La mobilité quotidienne des Belges
    J-P Hubert et Ph. Toint, 2004
    page 68-69
    """

    # Proportion of dwellings with a garage
    dwelling_stats["garage ?"] = {
        "Brussels": {"garage": (40 / 100), "no garage": (60 / 100)},
        "Flanders": {"garage": (83 / 100), "no garage": (17 / 100)},
        "Wallonia": {"garage": (78 / 100), "no garage": (22 / 100)},
    }

    # Number of dwellings with a garage
    # garage = {
    #    key: {sub_key: val[sub_key] * num_dwellings[key] for sub_key in [True, False]}
    #    for key, val in garage_prop.items()
    # }

    prop_garage_with_cars = (45 + 21 + 3) / 77
    # Number of dwellings with a garage and a car
    dwelling_stats["home charger ?"] = {
        key: {
            "home charger": val["garage"] * prop_garage_with_cars,
            "no home charger": val["no garage"]
            + (1 - prop_garage_with_cars) * val["garage"],
        }
        for key, val in dwelling_stats["garage ?"].items()
    }

    """
    Sources:
    Newly installed heat pumps in Belgium:
    https://www.eurobserv-er.org/20th-annual-overview-barometer/
    https://www.eurobserv-er.org/22nd-annual-overview-barometer/

    """
    heat_pump_installation = {
        "2019": {"air-to-air": 94380, "air-to-water": 8678, "ground-source": 2595},
        "2020": {"air-to-air": 86723, "air-to-water": 11764, "ground-source": 3193},
        "2021": {"air-to-air": 86915, "air-to-water": 13000, "ground-source": 3605},
        "2022": {"air-to-air": 87286, "air-to-water": 23754, "ground-source": 3260},
    }

    """
    Dwellings with solar pv
    Source:
    Brussels:
    Brugel:
    https://app.powerbi.com/view?r=eyJrIjoiNDkyNWIyNDgtNWNkNi00MWY2LTgxY2QtZTZlZWI2MDM1YmRhIiwidCI6ImMwYjg2YzA3LWRhZGUtNDkyMC1hYzEzLWIwZWNhZDNiMmM5NSIsImMiOjh9
    Total installé, photovoltaique, particulier 03-09-2024: 9609
    Flanders:
    https://www.vlaanderen.be/veka/energie-en-klimaatbeleid-in-cijfers/energiekaart (evolutie van de zonnepanelen in Vlaanderen xls)
    Vermogenklasse tab, <=10 kW Bijkomend aantal: 947901 (31/07/2024)
    Wallonia:
    https://www.renouvelle.be/fr/faits-chiffres/observatoire-photovoltaique/ (end 2023)
    1810 MW installé pour puissance <=10 kW, if we take an average installation of 4 kW per house, we get 452500 houses with pv or ~28.6% of houses with pv
    The same methodology for Flanders overestimates the number of houses with pv (estimates ~48% of houses have pv)
    Other source:
    https://trends.levif.be/a-la-une/developpement-durable/16-des-menages-wallons-sont-equipes-de-panneaux-solaires-mais-les-autres-technologies-vertes-sont-desesperement-a-la-traine/
    Says 16% of houses in Wallonia have solar panels and 22% for flanders (end 11-2023)
    Based on this we estimate there are ~20% of houses with solar panels in Wallonia.
    """

    dwelling_stats["pv ?"] = {
        "Brussels": {
            "pv": (9609 / dwelling_stats["region"]["Brussels"]),
            "no pv": 1 - (9609 / dwelling_stats["region"]["Brussels"]),
        },
        "Flanders": {
            "pv": (947_901 / dwelling_stats["region"]["Flanders"]),
            "no pv": 1 - (947_901 / dwelling_stats["region"]["Flanders"]),
        },
        "Wallonia": {"pv": (20 / 100), "no pv": (80 / 100)},
    }

    """
    Type of house apartment
    Source:
    https://statbel.fgov.be/sites/default/files/Over_Statbel_FR/Enquete%20SocEco%202001%20-%20Monographie%202%20Le%20logement%20en%20Belgique.pdf
    Figure IV.13: Shows the proportion of houses that are detached, semi-detached and terraced.

    terraced houses are 2 façades
    semi-detached houses are 3 façades
    detached houses are 4 façades

    """

    dwelling_stats["construction type"] = {
        "Brussels": {
            "house": {
                "terraced": 0.775510204081633,
                "semi-detached": 0.146546310832025,
                "detached": 0.0779434850863422,
            },
            "apartment": {"apartment unit": 1},
        },
        "Flanders": {
            "house": {
                "terraced": 0.287048665620094,
                "semi-detached": 0.258320251177394,
                "detached": 0.454631083202512,
            },
            "apartment": {"apartment unit": 1},
        },
        "Wallonia": {
            "house": {
                "terraced": 0.355508374616655,
                "semi-detached": 0.217740033026657,
                "detached": 0.426751592356688,
            },
            "apartment": {"apartment unit": 1},
        },
    }

    """
    Insulated house or not
    Source:
    https://www.recticelinsulation.com/sites/default/files/country_specific/be/LivreBlanc_BarometreDeLisolation_2020_Recticel.pdf
    """

    dwelling_stats["insulation"] = {
        "Brussels": {
            "none": 0.123,
            "roof": 0.528,
            "wall roof": 0,
            "ground roof": 0.01,
            "ground wall roof": 0.339,
        },
        "Flanders": {
            "none": 0.135,
            "roof": 0.287,
            "wall roof": 0.187,
            "ground roof": 0,
            "ground wall roof": 0.391,
        },
        "Wallonia": {
            "none": 0.361,
            "roof": 0.483,
            "wall roof": 0.006,
            "ground roof": 0,
            "ground wall roof": 0.15,
        },
    }

    return dwelling_stats


def sample_houses(tot_houses=10000):
    # TODO Change the multiplying factor to fit the area of dwellings source T tests better.
    # It comes from the WARD test (so not a t test).
    """
    Other useful source:
    For area of dwellings:
    https://core.ac.uk/download/pdf/34134029.pdf
    page 74
    """
    dwelling_stats = get_dwelling_stats()
    houses = []

    dwelling_stat_counter = create_dwelling_counter(dwelling_stats, tot_houses)

    for i in range(tot_houses):
        house = dict()
        for category in dwelling_stat_counter.keys():
            house[category] = choose_random_category(
                dwelling_stat_counter[category], house
            )
        houses.append(house)

    return houses


def get_stat_dependencies_and_categories(dwelling_stats):
    dependencies = {}
    categories = {}

    for key, value in dwelling_stats.items():
        dependencies[key] = []
        categories[key] = []

        populate_dependencies_and_categories(value, dependencies[key], categories[key])

        dependencies[key] = set(dependencies[key])
        categories[key] = set(categories[key])

    return dependencies, categories


def populate_dependencies_and_categories(
    dwelling_stats, dependency_list, category_list
):
    for key, value in dwelling_stats.items():
        if isinstance(value, dict):
            dependency_list.append(key)
            populate_dependencies_and_categories(value, dependency_list, category_list)
        else:
            category_list.append(key)


def create_dwelling_counter(dwelling_stats, tot_houses):
    """
    Create a dictionary with the number of dwellings per region and per category
    """
    dependencies, categories = get_stat_dependencies_and_categories(dwelling_stats)

    dwelling_stat_counter = {}
    for stat_key, dwelling_stat in dwelling_stats.items():
        dependent_stats = list()
        for dependency in dependencies[stat_key]:
            for stat, category_list in categories.items():
                if dependency in category_list:
                    dependent_stats.append(stat)
        dependent_stats = set(dependent_stats)

        for dependent_stat in dependent_stats.copy():
            other_dependent_stats = dependent_stats.copy()
            other_dependent_stats.remove(dependent_stat)
            for other_stat in other_dependent_stats:
                for category in categories[dependent_stat]:
                    if category in dependencies[other_stat]:
                        dependent_stats.remove(dependent_stat)
                        break
        usefull_count_list = [
            remove_unused_levels(dwelling_stat_counter[key], dependencies[stat_key])
            for key in dependent_stats
        ]

        dwelling_stat_counter[stat_key] = populate_dwelling_counter(
            dwelling_stat, usefull_count_list, tot_houses
        )

    return dwelling_stat_counter


def remove_unused_levels(usefull_stat_counter, dependencies):
    list_from_dict = recur_list_from_dict(usefull_stat_counter, dependencies)
    used_levels_dict = build_dict_from_list(list_from_dict)

    return used_levels_dict


def recur_list_from_dict(input_dict, dependencies):
    level_list = []

    if not isinstance(input_dict, dict):
        return [[input_dict]]

    for key in input_dict.keys():
        for lower_list in recur_list_from_dict(input_dict[key], dependencies):
            if key in dependencies:
                level_list.append([key] + lower_list)
            else:
                level_list.append(lower_list)

    return level_list


def build_dict_from_list(input_list):
    new_dict = dict()
    for element in input_list:
        cur_level = new_dict
        for value in element[:-2]:
            if value not in cur_level:
                cur_level[value] = dict()
            cur_level = cur_level[value]
        if element[-2] not in cur_level:
            cur_level[element[-2]] = element[-1]
        else:
            cur_level[element[-2]] += element[-1]

    return new_dict


def populate_dwelling_counter(dwelling_stat, usefull_count_list, tot_houses):

    first_key = list(dwelling_stat.keys())[0]
    if not isinstance(dwelling_stat[first_key], dict):
        return count_sample_proportionally(tot_houses, dwelling_stat)

    cur_dict = {}
    for stat_key, lower_stat in dwelling_stat.items():
        for i, usefull_count in enumerate(usefull_count_list):
            if stat_key not in usefull_count.keys():
                continue

            if isinstance(usefull_count[stat_key], dict):
                new_usefull_count_list = deepcopy(usefull_count_list)
                new_usefull_count_list[i] = usefull_count[stat_key]
                cur_dict[stat_key] = populate_dwelling_counter(
                    lower_stat, new_usefull_count_list, tot_houses
                )
            else:
                cur_dict[stat_key] = populate_dwelling_counter(
                    lower_stat, usefull_count_list, usefull_count[stat_key]
                )
    return cur_dict


def choose_random_category(cur_stat_counter, house):
    first_key = list(cur_stat_counter.keys())[0]

    if not isinstance(cur_stat_counter[first_key], dict):
        choices = list(cur_stat_counter.keys())
        occurences = [cur_stat_counter[key] for key in choices]
        selected_category = random.choices(choices, occurences)[0]
        cur_stat_counter[selected_category] -= 1

        return selected_category
    else:
        for _, set_category in house.items():
            if set_category in cur_stat_counter.keys():
                new_stat_counter = cur_stat_counter[set_category]
                return choose_random_category(new_stat_counter, house)


def count_sample_proportionally(tot_sample, dict_counts):

    sum_count = sum(dict_counts.values())
    dict_proba = {key: value / sum_count for key, value in dict_counts.items()}

    dict_samples = {key: 0 for key in dict_proba.keys()}

    for _ in range(tot_sample):
        max_key = max(dict_proba, key=dict_proba.get)
        dict_samples[max_key] += 1
        dict_proba[max_key] -= 1 / tot_sample

    return dict_samples


def save_csv_sample(houses):
    # Save the houses in a csv file
    with open("houses.csv", mode="w") as file:

        writer = csv.writer(file)
        header = list(houses[0].keys())
        replace_dict = {
            "house": "dwelling type",
            "heating": "central heating",
            "home_charger": "ev chargers",
            "rooms": "number of rooms",
        }
        header = [replace_dict.get(head, head) for head in header]
        # header = ["dwelling type" if head == "house" else head for head in header]
        writer.writerow(header)

        for house in houses:
            house["house"] = "house" if house["house"] else "apartment"

            writer.writerow(house.values())


def calc_prop_heatpump_installation():
    """
    Source:
    https://www.eurobserv-er.org/22nd-annual-overview-barometer/
    This is for the number of units sold, not the number of residential installations.
    Assumptions is therefor not totally correct but it's the best we have.
    "Nowadays, almost all air-to-air HPs operate in reversible mode"
    """
    heatpump_type = ["air-to-air", "air-to-water", "ground-source"]
    heat_pump_installation = {
        2019: {"air-to-air": 94380, "air-to-water": 8678, "ground-source": 2595},
        2020: {"air-to-air": 86723, "air-to-water": 11764, "ground-source": 3193},
        2021: {"air-to-air": 86915, "air-to-water": 13000, "ground-source": 3605},
        2022: {"air-to-air": 87286, "air-to-water": 23754, "ground-source": 3260},
    }

    total_heat_pump_installation = {}

    for heatpump in heatpump_type:
        total_heat_pump_installation[heatpump] = sum(
            [val[heatpump] for val in heat_pump_installation.values()]
        )

    total_heatpump = sum(total_heat_pump_installation.values())

    for heatpump in heatpump_type:
        proportion = total_heat_pump_installation[heatpump] / total_heatpump * 100
        print(f"Percetage of {heatpump} installations: {proportion}%")


if __name__ == "__main__":
    test_dict = {
        "house": 1,
        "apartment": 1,
    }
    # dependencies = ["house", "apartment"]
    # new_list = recur_list_from_dict(test_dict, dependencies)
    # new_dict = build_dict_from_list(new_list)
    # print(new_dict)

    houses = sample_houses(10)

    for house in houses:
        print(house)

    # save_csv_sample(houses)

    # calc_prop_heatpump_installation()
