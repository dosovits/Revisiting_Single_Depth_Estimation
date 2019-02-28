from minos.lib.RoomSimulator import RoomSimulator
from minos.config import sim_config

import imageio
import numpy as np
import csv
import os

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

steps_per_episode = 50
num_episodes = 1000
resolution = (640,480)
env_name = "pointgoal_mp3d_m"

simargs = sim_config.get(env_name)
simargs["resolution"] = resolution
# simargs["episodes_per_scene_train"] = 50
simargs["modalities"].append("depth")
simargs["outputs"].append("depth")
simargs["observations"]["depth"] = True
print(simargs)
sim = RoomSimulator(simargs)
sim.reset()

csv_file = open("train.csv", "w")
csv_writer = csv.writer(csv_file)

for nepisode in range(num_episodes):
    print("Episode  {}/{}".format(nepisode, num_episodes))
    sim.reset()
    episode_folder = os.path.join("train", "{:05}".format(nepisode))
    make_dir(episode_folder)
    for nstep in range(steps_per_episode):
        full_state = sim.step([0,1,0])
        image = full_state['observation']['sensors']['color']['data']
        depth = full_state['observation']['sensors']['depth']['data']
        # print("Image shape", image.shape, ", Depth shape", depth.shape)

        image = np.reshape(image, (image.shape[1], image.shape[0], -1))
        depth = np.reshape(depth, (depth.shape[1], depth.shape[0], -1))

        # print("Image shape", image.shape, ", Depth shape", depth.shape)
        image_filename = os.path.join(episode_folder, "{:03}_color.jpg".format(nstep))
        depth_filename = os.path.join(episode_folder, "{:03}_depth.png".format(nstep))
        imageio.imwrite(image_filename, image[:,:,:3])
        imageio.imwrite(depth_filename, (np.clip(depth/10., 0., 1.) * 255.).astype(np.uint8))
        csv_writer.writerow([os.path.join("data", "mp3d", image_filename), os.path.join("data", "mp3d", depth_filename)])

csv_file.close()

# {'simulator': 'room_simulator', 'num_simulators': 1, 'modalities': ['color', 'measurements', 'depth'], 'outputs': ['color', 'measurements', 'rewards', 'terminals', 'depth'], 'resolution': (640, 480), 'frame_skip': 1, 'host': 'localhost', 'log_action_trace': False, 'auto_start': True, 'collision_detection': {'mode': 'navgrid'}, 'navmap': {'refineGrid': True, 'autoUpdate': True, 'allowDiagonalMoves': True, 'reverseEdgeOrder': False}, 'reward_type': 'dist_time', 'observations': {'color': True, 'forces': False, 'audio': False, 'normal': False, 'depth': True, 'objectId': False, 'objectType': False, 'roomType': False, 'roomId': False, 'map': False}, 'color_encoding': 'rgba', 'scene': {'dataset': 'mp3d', 'replaceModels': {'326': '327', '331': '327', '502': '502_2', '622': '625', '122': '122_2', '133': '133_2', '214': '214_2', '246': '246_2', '247': '247_2', '73': '73_2', '756': '756_2', '757': '757_2', '758': '758_2', '759': '759_2', '760': '760_2', '761': '761_2', '762': '762_2', '763': '763_2', '764': '764_2', '768': '768_2', '769': '769_2', '770': '770_2', 's__1762': 's__1762_2', 's__1763': 's__1763_2', 's__1764': 's__1764_2', 's__1765': 's__1765_2', 's__1766': 's__1766_2', 's__1767': 's__1767_2', 's__1768': 's__1768_2', 's__1769': 's__1769_2', 's__1770': 's__1770_2', 's__1771': 's__1771_2', 's__1772': 's__1772_2', 's__1773': 's__1773_2'}, 'defaultModelFormat': None, 'defaultSceneFormat': None}, 'config': '', 'color_mode': 'GRAY', 'maps': ['MAP01'], 'switch_maps': False, 'game_args': '', 'task': 'point_goal', 'goal': {'type': 'position', 'position': 'random', 'radius': 0.25}, 'scenes_file': '/home/adosovit/work/toolboxes/2019/navigation-benchmark/3rdparty/minos/minos/config/../data/scenes.mp3d.csv', 'states_file': '/home/adosovit/work/toolboxes/2019/navigation-benchmark/3rdparty/minos/minos/config/../data/episode_states.mp3d.csv.bz2', 'roomtypes_file': '/home/adosovit/work/toolboxes/2019/navigation-benchmark/3rdparty/minos/minos/config/../data/roomTypes.suncg.csv', 'num_episodes_per_restart': 1000, 'num_episodes_per_scene': 100, 'max_states_per_scene': 10, 'episodes_per_scene_test': 1, 'episodes_per_scene_train': 10, 'episode_schedule': 'train', 'measure_fun': <minos.lib.util.measures.MeasureDistDirTime object at 0x7f402026fe48>, 'nonserializable': ['measure_fun', 'scene_filter', 'episode_filter'], 'agent': {'radialClearance': 0.2}, 'scene_filter': <function <lambda> at 0x7f4020be0c80>, 'episode_filter': <function <lambda> at 0x7f40202767b8>, 'objective_size': 4, 'logdir': 'logs/2019_02_18_11_40_03'}
