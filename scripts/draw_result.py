import matplotlib.pyplot as plt
import json

import numpy as np

plt.rcParams['font.family'] = ['SimHei']  # chinese
plt.rcParams['axes.unicode_minus'] = False

# 使用matplotlib，读取log_debug_0文件夹下，metrics.json文件中，"rewards"键的值
title_font_size = 16  #ppt 20, paper 16
label_font_size = 14  #ppt 18, paper 14
legend_font_size = 12 #ppt 14, paper 12


def read_json(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


def draw_episode_reward(reward_data):
    plt.figure()
    plt.plot(reward_data, label="回合奖励")
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("奖励", fontsize=label_font_size)
    plt.title("逐回合奖励曲线", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.show()


def draw_episode_reward_mean(reward_data):
    reward_buffer = []
    reward_mean = []
    for i in reward_data:
        if len(reward_buffer) > 20:
            reward_buffer.pop(0)
        reward_buffer.append(i)
        reward_mean.append(np.mean(reward_buffer))

    plt.figure()
    plt.plot(reward_mean, label="平滑奖励")
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("奖励", fontsize=label_font_size)
    plt.title("平滑奖励曲线", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.show()


def draw_compare_episode_reward_mean(rl_inference_reward, sac_reward):
    rl_inference_reward_buffer = []
    rl_inference_reward_mean = []
    for i in rl_inference_reward:
        if len(rl_inference_reward_buffer) > 20:
            rl_inference_reward_buffer.pop(0)
        rl_inference_reward_buffer.append(i)
        rl_inference_reward_mean.append(np.mean(rl_inference_reward_buffer))

    sac_reward_buffer = []
    sac_reward_mean = []
    sac_reward_ax = []
    for i in sac_reward:
        sac_reward_ax.append(i[0])
        if len(sac_reward_buffer) > 20:
            sac_reward_buffer.pop(0)
        sac_reward_buffer.append(i[1])
        sac_reward_mean.append(np.mean(sac_reward_buffer))
    plt.figure()
    plt.plot(rl_inference_reward_mean, label="主动推理强化学习奖励")
    plt.plot(sac_reward_ax, sac_reward_mean, label="SAC强化学习奖励")
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("奖励", fontsize=label_font_size)
    plt.title("平滑奖励曲线", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.xlim([0, 1000])
    plt.show()


def draw_compare_episode_success_rate(rl_inference_reward, sac_reward, success_reward):
    rl_inference_success_buffer = []
    rl_inference_success_rate = []
    for i in rl_inference_reward:
        if len(rl_inference_success_buffer) > 20:
            rl_inference_success_buffer.pop(0)
        rl_inference_success_buffer.append(i)
        success = 0
        for j in rl_inference_success_buffer:
            if j > success_reward:
                success += 1
        rl_inference_success_rate.append(success / len(rl_inference_success_buffer))

    sac_success_buffer = []
    sac_success_rate = []
    sac_reward_ax = []
    for i in sac_reward:
        sac_reward_ax.append(i[0])
        if len(sac_success_buffer) > 20:
            sac_success_buffer.pop(0)
        sac_success_buffer.append(i[1])
        success = 0
        for j in sac_success_buffer:
            if j > success_reward:
                success += 1
        sac_success_rate.append(success / len(sac_success_buffer))

    plt.figure()
    ax = plt.gca()
    plt.plot(rl_inference_success_rate, label="主动推理强化学习成功率")
    plt.plot(sac_reward_ax, sac_success_rate, label="SAC强化学习成功率")
    plt.plot([90, 90], [-0.05, 1], linestyle='--', color='red')
    plt.plot([687, 687], [-0.05, 1], linestyle='--', color='red')
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("成功率", fontsize=label_font_size)
    plt.title("成功率曲线", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)

    xticks = ax.get_xticks()
    xticks = np.concatenate([xticks, [90, 687]])
    xticks.sort()
    ax.set_xticks(xticks)
    ax.get_xticklabels()[2].set_color("red")
    ax.get_xticklabels()[4].set_color("red")

    plt.xlim([0, 1000])
    plt.ylim([-0.05, 1.05])
    plt.show()


def draw_sac_compare():
    file_path = "../" + dir_name + "SAC_incline_dense_sde_2e5_record-tag-reward.json"
    sac_raw_data = read_json(file_path)
    sac_reward = []
    for i in sac_raw_data:
        sac_reward.append([i[1] / 80, i[2]])
    draw_compare_episode_reward_mean(data["rewards"], sac_reward)
    draw_compare_episode_success_rate(data["rewards"], sac_reward, 1)


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
    plt.plot(success_rate, label="成功率")
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("成功率", fontsize=label_font_size)
    plt.title("成功率", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.show()


def draw_episode_planer_reward(planer_reward_data):
    plt.figure()
    data_mean = np.array([float(i["mean"]) for i in planer_reward_data])
    data_std = np.array([float(i["std"]) for i in planer_reward_data])
    plt.plot(data_mean, label="规划过程的预测奖励分布", color="red")
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, alpha=0.2)
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("奖励", fontsize=label_font_size)
    plt.title("逐回合的规划预测奖励分布", fontsize=title_font_size)
    plt.ylim(-40, 30)
    plt.legend(fontsize=legend_font_size)
    plt.show()


def draw_episode_planer_information(planer_information_data):
    plt.figure()
    data_mean = np.array([float(i["mean"]) for i in planer_information_data])
    data_std = np.array([float(i["std"]) for i in planer_information_data])
    plt.plot(data_mean, label="规划过程的期望参数增益", color="green")
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, alpha=0.2)
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("期望参数增益", fontsize=label_font_size)
    plt.title("逐回合的规划过程期望参数增益", fontsize=label_font_size)
    # plt.ylim(0, 6)
    plt.legend(fontsize=legend_font_size)
    plt.show()

def draw_tactile_compare(tactile_reward,success_reward):
    no_tactile_dir_name = "log_push_ball_incline_dense_0.4reset/"
    no_tactile_file_path = "../" + no_tactile_dir_name + "metrics.json"
    no_tactile_raw_data = read_json(no_tactile_file_path)
    no_tactile_reward=no_tactile_raw_data["rewards"]

    no_tactile_reward_buffer = []
    no_tactile_reward_mean = []
    no_tactile_success_buffer = []
    no_tactile_success_rate = []
    for i in no_tactile_reward:
        if len(no_tactile_reward_buffer) > 20:
            no_tactile_reward_buffer.pop(0)
            no_tactile_success_buffer.pop(0)
        no_tactile_reward_buffer.append(i)
        if i > success_reward:
            no_tactile_success_buffer.append(1)
        else:
            no_tactile_success_buffer.append(0)
        no_tactile_success_rate.append(np.mean(no_tactile_success_buffer))
        no_tactile_reward_mean.append(np.mean(no_tactile_reward_buffer))

    tactile_reward_buffer = []
    tactile_reward_mean = []
    tactile_success_buffer = []
    tactile_success_rate = []
    for i in tactile_reward:
        if len(tactile_reward_buffer) > 20:
            tactile_reward_buffer.pop(0)
            tactile_success_buffer.pop(0)
        tactile_reward_buffer.append(i)
        if i > success_reward:
            tactile_success_buffer.append(1)
        else:
            tactile_success_buffer.append(0)
        tactile_success_rate.append(np.mean(tactile_success_buffer))
        tactile_reward_mean.append(np.mean(tactile_reward_buffer))

    plt.figure(1)
    ax = plt.gca()
    plt.plot(tactile_reward_mean, label="有tactile的主动推理强化学习平滑奖励曲线")
    plt.plot(no_tactile_reward_mean, label="无tactile的主动推理强化学习平滑奖励曲线")
    plt.plot([-5, 115], [37.28, 37.28], linestyle=':', color='red')
    plt.plot([-5, 115], [47.619, 47.619], linestyle=':', color='red')
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("奖励", fontsize=label_font_size)
    plt.title("有无tactile传感器学习平滑奖励曲线的对比", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.xlim([-5, 115])
    plt.show()

    plt.figure(2)
    plt.plot(tactile_success_rate, label="有tactile的主动推理强化学习成功率")
    plt.plot(no_tactile_success_rate, label="无tactile的主动推理强化学习成功率")
    plt.plot([-5, 115], [1.0, 1.0], linestyle=':', color='red')
    plt.plot([-5, 115], [0.85, 0.85], linestyle=':', color='red')
    plt.xlabel("回合", fontsize=label_font_size)
    plt.ylabel("成功率", fontsize=label_font_size)
    plt.title("有无tactile传感器学习成功率的对比", fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.xlim([-5, 115])
    plt.show()


if __name__ == "__main__":
    dir_name = "log_tactile_push_ball_vec/"
    file_path = "./" + dir_name + "metrics.json"
    data = read_json(file_path)

    # draw_tactile_compare(data["rewards"], 1)
    # draw_sac_compare()

    draw_episode_reward(data["rewards"])
    # draw_success_rate(data["rewards"], 1)
    # draw_episode_reward_mean(data["rewards"])
    # draw_episode_planer_reward(data["reward_stats"])
    # draw_episode_planer_information(data["info_stats"])
