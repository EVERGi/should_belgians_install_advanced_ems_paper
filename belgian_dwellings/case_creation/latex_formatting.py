from belgian_dwellings.case_creation.scenario_creation import get_dwelling_stats, create_dwelling_counter


def print_dwelling_stats_table():

    # Print each stat in a table with the values that can be copy pasted in excel

    dwelling_stats = get_dwelling_stats()

    tot_houses = 1000
    dwelling_stat_counter = create_dwelling_counter(dwelling_stats, tot_houses)

    dwelling_stat_counter = clean_construction_type(dwelling_stat_counter)

    stat_key_dict = {
        "central heating": "With central heating",
        "no central heating": "Without central heating",
        "garage": "With garage",
        "no garage": "Without garage",
        "home charger": "With garage and car",
        "no home charger": "Without garage or car",
        "pv": "With solar panels",
        "no pv": "Without solar panels",
        "house": "House",
        "apartment": "Apartment",
        "terraced house": "Terraced house",
        "semi-detached house": "Semi-detached house",
        "detached house": "Detached house",
        "apartment unit apartment": "Apartment unit",
        "ground wall roof": "Ground, wall and roof insulation",
        "ground roof": "Ground and roof insulation",
        "wall roof": "Wall and roof insulation",
        "ground": "Ground insulation",
        "wall": "Wall insulation",
        "roof": "Roof insulation",
        "none": "No insulation",
        "1 room": "1 room",
        "2 room": "2 rooms",
        "3 room": "3 rooms",
        "4 room": "4 rooms",
        "5 room": "5 rooms",
        "6 room": "6 rooms",
        "7 room": "7 rooms",
        "8 room": "8 rooms",
        "9 room": "9+ rooms",
        "0 bedroom": "0 bedroom",
        "1 bedroom": "1 bedroom",
        "2 bedroom": "2 bedrooms",
        "3 bedroom": "3 bedrooms",
        "4 bedroom": "4 bedrooms",
        "5 bedroom": "5 bedrooms",
        "6 bedroom": "6 bedrooms",
    }

    table_captions = {
        "region": "Distribution of dwellings per region \cite{statbelCensus2021Nombre2021}",
        "heating ?": "Distribution of dwellings with central heating \cite{statbelCensus2021Nombre2021}",
        "rooms": "Distribution of dwellings per number of rooms \cite{statbelCensus2021Nombre2021a}",
        "bedrooms": "Distribution of dwellings per number of bedrooms \cite{meirelesDomesticHotWater2022}",
        "surface area": "Distribution of dwellings per surface area \cite{vannesteLogementBelgique2007}",
        "dwelling type": "Distribution of dwellings per dwelling type \cite{statbelCensus2021Nombre2021b}",
        "garage ?": "Distribution of dwellings with a garage",
        "home charger ?": "Distribution of dwellings with a garage and a car \cite{hubertMobiliteQuotidienneBelges2003}",
        "pv ?": "Distribution of dwellings with solar panels \cite{brugelEnergiesRenouvelablesRegion2024,vlaamsenergie-enklimaatagentschapZonnepanelenVlaanderen2024,lallemand16MenagesWallons2023}",
        "construction type": "Distribution of dwellings per construction type \cite{vannesteLogementBelgique2007}",
        "insulation": "Distribution of dwellings by insulation type \cite{recticelinsulationBarometreLisolation20202020}",
    }

    for stat_key, stat_value in dwelling_stat_counter.items():
        if stat_key == "garage ?":
            continue
        header = [head for head in stat_value.keys()]
        caption = table_captions[stat_key]
        if not isinstance(stat_value[header[0]], dict):

            print("\\begin{table}[H]")
            print("\\centering")
            print("\\caption{" + caption + "}")
            print("\\begin{tabular}{|" + " | ".join(["c"] * len(header)) + "|}")
            print("\\hline")
            print(" & ".join(header) + " \\\\")
            print("\\hline")
            # for head in header:
            #    print(str(stat_value[head]) + " & ", end="")
            print(
                " & ".join([str(stat_value[head] / 10) + "\\%" for head in header])
                + " \\\\"
            )
            print("\\hline")
            print("\\end{tabular}")
            label_key = stat_key.replace(" ", "_").replace("_?", "")
            print("\\label{tab:stats_" + label_key + "}")
            print("\\end{table}")
            print("\n")
        else:
            print("\\begin{table}[H]")
            print("\\centering")
            print("\\caption{" + caption + "}")
            if header[0] == "1 room":
                print("\setlength{\\tabcolsep}{4pt}")
                print("\small")
            print(
                "\\begin{tabular}{|" + " | ".join(["c"] * (len(header) + 1)) + "||c|}"
            )
            print("\\hline")
            intro_header = ""
            if header[0] == "1 room":
                header = [
                    head.replace(" room", "").replace("9", "9+") for head in header
                ]
                intro_header = (
                    "\\begin{tabular}[c]{@{}c@{}}Number of\\\\rooms\\end{tabular}"
                )

            print(f"{intro_header} & " + " & ".join(header) + " & Total \\\\")
            print("\\hline")
            all_rows = []
            for sub_key, sub_value in stat_value.items():
                for sub_sub_key in sub_value.keys():
                    if sub_sub_key not in all_rows:
                        all_rows.append(sub_sub_key)
            for row_key in all_rows:
                if row_key in stat_key_dict:
                    row_index = stat_key_dict[row_key]
                else:
                    row_index = row_key
                row = [row_index]
                row_values = [
                    (
                        stat_value[sub_key][row_key] / 10
                        if row_key in stat_value[sub_key]
                        else 0
                    )
                    for sub_key in stat_value.keys()
                ]
                total = sum(row_values)
                row += [
                    f"{value:.1f}\\%" if value != 0 else "-" for value in row_values
                ] + [f"{total:.1f}\\%"]
                print(" & ".join(row) + " \\\\")
                print("\\hline")
            print("\\end{tabular}")
            label_key = stat_key.replace(" ", "_").replace("_?", "")
            print("\\label{tab:stats_" + label_key + "}")
            print("\\end{table}")
            print("\n")


def clean_construction_type(dwelling_stat_counter):
    construction_type = dwelling_stat_counter["construction type"]
    clean_construction_type = dict()
    for region, dwelling_dict in construction_type.items():
        clean_construction_type[region] = dict()
        for dwelling_type, construction_dict in dwelling_dict.items():
            for construction_type, num_houses in construction_dict.items():
                clean_key = f"{construction_type} {dwelling_type}"
                clean_construction_type[region][clean_key] = num_houses

    dwelling_stat_counter["construction type"] = clean_construction_type
    return dwelling_stat_counter


if __name__ == "__main__":

    print_dwelling_stats_table()
