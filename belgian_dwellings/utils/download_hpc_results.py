import paramiko
from scp import SCPClient
import os
import shutil
import subprocess
import json


def createSSHClient(server, user):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=user)
    return client


def download_hpc_treec_scp(log_folder):

    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    ssh = createSSHClient("login.hpc.vub.be", user="vsc10250")
    scp = SCPClient(ssh.get_transport())

    remote_path = f"/data/brussel/102/vsc10250/ems_belgium_usefull/{log_folder}/"
    local_path = f"{log_folder}/"

    scp.get(remote_path=remote_path, local_path=local_path, recursive=True)

    scp.close()


def execute_rsync(remote_path, local_path):
    rsync_command = [
        "rsync",
        "-avz",
        "--delete",
        "-e",
        "ssh",
        remote_path,
        local_path,
    ]
    subprocess.run(rsync_command, check=True)


def download_hpc_treec_training(log_folder):
    # if os.path.exists(log_folder):
    #    shutil.rmtree(log_folder)

    remote_path = (
        f"hpc_vub:/data/brussel/102/vsc10250/ems_belgium_usefull/{log_folder}/"
    )
    local_path = f"{log_folder}/"

    # os.makedirs(local_path, exist_ok=True)

    execute_rsync(remote_path, local_path)


def download_hpc_results():
    remote_path = f"hpc_vub:/data/brussel/102/vsc10250/ems_belgium_usefull/results/"
    local_path = f"results_tmp/"
    os.makedirs(local_path, exist_ok=True)
    execute_rsync(remote_path, local_path)


def download_and_add_hpc_results(tot_houses):
    download_hpc_results()
    add_hpc_results(tot_houses)


def add_hpc_results(tot_houses):
    # Add the results from the HPC to the results folder
    tmp_results_folder = "results_tmp/"
    for result_extension in ["", "_no_enforcement"]:
        result_filename = f"belgium_usefull_{tot_houses}{result_extension}.csv"
        if not os.path.exists(f"results/{result_filename}"):
            continue

        combine_tmp_result_file(tmp_results_folder, result_filename)

        if result_extension == "_no_enforcement":
            combine_soc_json(tmp_results_folder, result_filename)


def combine_tmp_result_file(tmp_results_folder, result_filename):
    tmp_result_path = f"{tmp_results_folder}/{result_filename}"
    result_path = f"results/{result_filename}"
    # Read the temporary results file
    tmp_file = open(tmp_result_path, "r")
    tmp_lines = tmp_file.readlines()
    tmp_file.close()

    result_file = open(result_path, "r")
    result_lines = result_file.readlines()
    result_file.close()

    for tmp_line in tmp_lines:
        start_tmp = tmp_line.split(",")[:2]
        start_tmp = ",".join(start_tmp)
        line_found = False
        for i, result_line in enumerate(result_lines):
            start_result = result_line.split(",")[:2]
            start_result = ",".join(start_result)
            if start_tmp == start_result:
                # If the first two columns match, replace the line
                result_lines[i] = tmp_line
                line_found = True
                break
        if not line_found:
            # If the line was not found, append it
            result_lines.append(tmp_line)
    # Write the updated results to the result file
    with open(result_path, "w") as file:
        file.writelines(result_lines)

    sort_config_file_name(result_path)

    # Add all files from the tmp_results_folder/belgium_usefull_{tot_houses} to the results folder
    profile_foldername = result_filename.replace(".csv", "_profiles/")
    tmp_profiles_folder = f"{tmp_results_folder}/{profile_foldername}"
    profile_dir = f"results/{profile_foldername}"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    for file in os.listdir(tmp_profiles_folder):
        shutil.copy(
            os.path.join(tmp_profiles_folder, file),
            os.path.join(profile_dir, file),
        )


def combine_soc_json(tmp_results_folder, result_filename):
    tmp_result_path = f"{tmp_results_folder}/{result_filename}"
    tmp_soc_file = tmp_result_path.replace(".csv", "_soc.json")
    result_soc_file = f"results/{result_filename.replace('.csv', '_soc.json')}"
    if not os.path.exists(result_soc_file):
        shutil.copy(tmp_soc_file, result_soc_file)
    else:
        with open(tmp_soc_file, "r") as f:
            tmp_soc_data = json.load(f)
        with open(result_soc_file, "r") as f:
            result_soc_data = json.load(f)

        # Combine the two JSON files
        for config_file, ems_dict in tmp_soc_data.items():
            if config_file not in result_soc_data:
                result_soc_data[config_file] = ems_dict
            else:
                for ems_name, soc_data in ems_dict.items():
                    result_soc_data[config_file][ems_name] = soc_data

        # Write the combined JSON data to the result file
        with open(result_soc_file, "w") as f:
            json.dump(result_soc_data, f, indent=4)


def sort_config_file_name(result_path):
    with open(result_path, "r") as f:
        lines = f.readlines()
    lines = [lines[0]] + sorted(
        lines[1:], key=lambda x: int(x.split(",")[0].split("_")[-1].split(".")[0])
    )
    with open(result_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    # training_folder = "treec_train_500"
    tot_houses = 500
    # download_hpc_treec_training(f"treec_train_{tot_houses}")
    download_and_add_hpc_results(tot_houses)
    # download_hpc_results()
    # add_hpc_results(tot_houses)
