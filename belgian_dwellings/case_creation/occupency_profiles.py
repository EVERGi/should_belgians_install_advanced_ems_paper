import matplotlib.pyplot as plt
import matplotlib
import datetime
import random

"""
Source occupancy:
https://publications.ibpsa.org/conference/paper/?id=bs2013_1273
Source temperature ranges:
https://www.sciencedirect.com/science/article/pii/S0306261908001785
"""

# Source for belgian profiles:
# https://publications.ibpsa.org/conference/paper/?id=bs2013_1273
occupancy_profiles = {
    1: {
        "00:00-07:00": "sleeping",
        "07:00-08:00": "at home",
        "08:00-18:00": "absent",
        "18:00-19:00": "at home",
        "19:00-20:45": "absent",
        "20:45-23:15": "at home",
        "23:15-24:00": "sleeping",
    },
    2: {
        "00:00-07:00": "sleeping",
        "07:00-08:00": "at home",
        "08:00-16:45": "absent",
        "16:45-22:30": "at home",
        "22:30-24:00": "sleeping",
    },
    3: {
        "00:00-09:00": "sleeping",
        "09:00-13:00": "at home",
        "13:00-20:30": "absent",
        "20:30-23:45": "at home",
        "23:45-24:00": "sleeping",
    },
    4: {
        "00:00-07:30": "sleeping",
        "07:30-10:00": "at home",
        "10:00-11:00": "absent",
        "11:00-15:00": "at home",
        "15:00-18:00": "absent",
        "18:00-22:30": "at home",
        "22:30-24:00": "sleeping",
    },
    5: {
        "00:00-07:00": "sleeping",
        "07:00-10:00": "at home",
        "10:00-11:30": "absent",
        "11:30-14:00": "at home",
        "14:00-16:00": "absent",
        "16:00-22:45": "at home",
        "22:45-24:00": "sleeping",
    },
    6: {
        "00:00-08:30": "sleeping",
        "08:30-10:00": "at home",
        "10:00-12:00": "absent",
        "12:00-22:15": "at home",
        "22:15-24:00": "sleeping",
    },
    7: {
        "00:00-08:00": "sleeping",
        "08:00-23:00": "at home",
        "23:00-24:00": "sleeping",
    },
}

# Occurance taken from source
# Coverage of week, saturday and sunday is 81.4%, 67.7% and 80.0% respectively
profile_occurance = {
    "weekday": {1: 0.213, 2: 0.172, 3: 0.1, 4: 0.069, 5: 0.068, 6: 0.082, 7: 0.11},
    "saturday": {
        1: 0.082,
        2: 0.052,
        3: 0.161,
        4: 0.091,
        5: 0.072,
        6: 0.123,
        7: 0.096,
    },
    "sunday": {
        1: 0.062,
        2: 0.078,
        3: 0.183,
        4: 0.116,
        5: 0.049,
        6: 0.185,
        7: 0.127,
    },
}


def count_sample_proportionally(tot_sample, dict_counts):

    sum_count = sum(dict_counts.values())
    dict_proba = {key: value / sum_count for key, value in dict_counts.items()}

    dict_samples = {key: 0 for key in dict_proba.keys()}

    for _ in range(tot_sample):
        max_key = max(dict_proba, key=dict_proba.get)
        dict_samples[max_key] += 1
        dict_proba[max_key] -= 1 / tot_sample

    return dict_samples


def list_shuffled_samples(dict_samples):
    samples = list()
    for key, count in dict_samples.items():
        samples += [key] * count
    # Shuffle
    random.shuffle(samples)
    return samples


def create_week_occupancy_profiles(num_profiles):
    day_types = ["weekday", "saturday", "sunday"]

    occupancies = dict()
    for day_type in day_types:
        occurance = profile_occurance[day_type]
        occurance_samples = count_sample_proportionally(num_profiles, occurance)
        samples = list_shuffled_samples(occurance_samples)
        occupancies[day_type] = samples

    week_profiles = []
    for i in range(num_profiles):
        profile = dict()
        for day_type in day_types:
            profile_type = occupancies[day_type][i]
            profile[day_type] = occupancy_profiles[profile_type]
        week_profiles.append(profile)
    return week_profiles


def get_temperature_range(t_ref, occupancy):
    # w and alpha values for 10% predicted percentage of dissatisfied
    W = 5
    ALPHA = 0.7

    if occupancy == "at home":
        if t_ref < 12.5:
            t_n = 20.4 + 0.06 * t_ref
        else:
            t_n = 16.63 + 0.36 * t_ref

            t_up = t_n + ALPHA * W
            t_low = max(t_n - (1 - ALPHA) * W, 18)
    elif occupancy == "sleeping":
        if t_ref < 0:
            t_n = 16
        elif 0 <= t_ref < 12.6:
            t_n = 16 + 0.23 * t_ref
        elif 12.6 <= t_ref < 21.8:
            t_n = 9.18 + 0.77 * t_ref
        else:
            t_n = 26

        t_up = min(t_n + ALPHA * W, 26)
        t_low = max(t_n - (1 - ALPHA) * W, 16)
    elif occupancy == "absent":
        t_low = 10
        t_up = 35

    return t_low, t_up


def calc_t_eref(t_days):
    t_eref = (t_days[0] + 0.8 * t_days[1] + 0.4 * t_days[2] + 0.2 * t_days[3]) / 2.4
    return t_eref


def get_occupancy(profile_num, dt):
    cur_profile = occupancy_profiles[profile_num]
    return calc_occupancy(cur_profile, dt)


def calc_occupancy(profile, dt):
    str_time = dt.strftime("%H:%M")

    for time_range, profile in profile.items():
        start, end = time_range.split("-")

        if start <= str_time < end:
            return profile


def plot_example_range():
    t_days = [14, 13, 12, 14]
    t_eref = calc_t_eref(t_days)

    day_timesteps = [
        datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(24 * 4)
    ]

    for profile_num in [1, 2, 3, 4, 5, 6, 7]:
        t_eref_prof = t_eref + profile_num
        t_up_day = []
        t_low_day = []
        for i, time in enumerate(day_timesteps):
            current_profile = get_occupancy(profile_num, time)
            t_low, t_up = get_temperature_range(t_eref_prof, current_profile)
            t_low_day.append(t_low)
            t_up_day.append(t_up)

        color = matplotlib.colormaps.get_cmap("tab10")(profile_num)
        plt.plot(day_timesteps, t_low_day, label=f"Profile {profile_num}", color=color)
        plt.plot(day_timesteps, t_up_day, color=color)
        # Only show hours and minute on the x axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    plt.legend()
    plt.show()


def plot_week_profiles(week_profiles):
    day_timesteps = [
        datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=15 * i)
        for i in range(24 * 4 * 7)
    ]
    t_eref = calc_t_eref([14, 13, 12, 14])

    for week_profile_num, week_profile in enumerate(week_profiles):
        t_up_day = []
        t_low_day = []
        for time in day_timesteps:
            if time.weekday() < 5:
                profile = week_profile["weekday"]
            elif time.weekday() == 5:
                profile = week_profile["saturday"]
            elif time.weekday() == 6:
                profile = week_profile["sunday"]
            current_occupancy = calc_occupancy(profile, time)
            t_low, t_up = get_temperature_range(t_eref, current_occupancy)
            t_low_day.append(t_low)
            t_up_day.append(t_up)

        color = matplotlib.colormaps.get_cmap("tab10")(week_profile_num)
        plt.plot(
            day_timesteps, t_low_day, label=f"Profile {week_profile_num}", color=color
        )
        plt.plot(day_timesteps, t_up_day, color=color)
        # Only show hours and minute on the x axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    week_profiles = create_week_occupancy_profiles(10)

    plot_week_profiles(week_profiles)

    # T_eref calc
    # T_eref = (T_today+0.8*T_today-1+0.4*T_today-2+0.2*T_today-3)/2.4
    # T_today is the mean of min and max temperature of the day

    ## AT HOME
    # Neutral temperature equation at home (Other rooms in source):
    # If exterior temperature (T_eref) <12.5°C, Tn = 20.4°C+0.06*T_eref
    # else, Tn = 16.63°C+0.36*T_eref

    # T_up = Tn+0.7*5°C
    # T_low = Tn-0.3*5°C

    ## SLEEPING
    # Neutral temperature equation sleeping (Bedroom in source):
    # If T_eref<0°C, Tn = 16°C
    # Else if 0°C<=T_eref<12.6°C, Tn = 16°C+0.23*T_eref
    # Else if 12.6°C<=T_eref<21.8°C, Tn = 9.18°C+0.77*T_eref
    # Else Tn = 26°C

    # T_up = min(Tn + 0.7 * 5°C, 26°C)
    # T_low = max(Tn - 0.3 * 5°C, 16°C)

    ## ABSENT
    # Temperature range:
    # T_low = 10°C
    # T_up = 35°C
