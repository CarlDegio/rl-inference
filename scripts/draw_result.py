import matplotlib.pyplot as plt
import json

import numpy as np


# 使用matplotlib，读取log_debug_0文件夹下，metrics.json文件中，"rewards"键的值

def read_json(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


def draw_episode_reward(reward_data):
    plt.figure()
    plt.plot(reward_data, label="episode-real-rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.show()


def draw_episode_planer_reward(planer_reward_data):
    plt.figure()
    data_mean = np.array([float(i["mean"]) for i in planer_reward_data])
    data_std = np.array([float(i["std"]) for i in planer_reward_data])
    plt.plot(data_mean, label="planer-traj-predict-reward", color="red")
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Predict Reward Curve")
    plt.ylim(-20, 40)
    plt.legend()
    plt.show()

def draw_episode_planer_information(planer_information_data):
    plt.figure()
    data_mean = np.array([float(i["mean"]) for i in planer_information_data])
    data_std = np.array([float(i["std"]) for i in planer_information_data])
    plt.plot(data_mean, label="planer-traj-predict-information", color="green")
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Information Gain")
    plt.title("Predict Information Curve")
    plt.ylim(0, 6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dir_name = "log_tactile_push_ball_0/"
    file_path = "../" + dir_name + "metrics.json"
    data = read_json(file_path)

    draw_episode_reward(data["rewards"])
    draw_episode_planer_reward(data["reward_stats"])
    draw_episode_planer_information(data["info_stats"])
