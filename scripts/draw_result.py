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

def draw_episode_reward_mean(reward_data):
    reward_buffer=[]
    reward_mean=[]
    for i in reward_data:
        if len(reward_buffer)>20:
            reward_buffer.pop(0)
        reward_buffer.append(i)
        reward_mean.append(np.mean(reward_buffer))

    plt.figure()
    plt.plot(reward_mean, label="20episodes-rewards-mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Mean Curve")
    plt.legend()
    plt.show()
def draw_success_rate(reward_data, success_reward):
    success_buffer = []
    success_rate = []
    for i in reward_data:
        if len(success_buffer) > 20:
            success_buffer.pop(0)
        success_buffer.append(i)
        success = 0
        for j in success_buffer:
            if j > success_reward:
                success += 1
        success_rate.append(success / len(success_buffer))
    plt.figure()
    plt.plot(success_rate, label="20episodes-success-rate-mean")
    plt.xlabel("Episode")
    plt.ylabel("success rate")
    plt.title("success rate")
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
    plt.ylim(-40, 30)
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
    # plt.ylim(0, 6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dir_name = "log_tactile_push_ball_0/"
    file_path = "../" + dir_name + "metrics.json"
    data = read_json(file_path)

    draw_episode_reward(data["rewards"])
    draw_success_rate(data["rewards"],1)
    draw_episode_reward_mean(data["rewards"])
    draw_episode_planer_reward(data["reward_stats"])
    draw_episode_planer_information(data["info_stats"])
