from collections import defaultdict, namedtuple
import json
import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dpt.runtime import path_to_data


SceneRow = namedtuple(
    "SceneRow", ["scene", "pedestrian", "start", "end", "fps", "tag"]
)
SceneRow.__new__.__defaults__ = (None,) * len(SceneRow._fields)

TrackRow = namedtuple(
    "TrackRow",
    [
        "frame",
        "pedestrian",
        "x",
        "y",
        "h",
        "w",
        "l",
        "rot_z",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "prediction_number",
        "scene_id",
    ],
)
TrackRow.__new__.__defaults__ = (None,) * len(TrackRow._fields)


class JrdbReader:
    """Read the scene and track records used by the JRDB experiments."""

    def __init__(self, input_file):
        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = {}
        self._read_file(input_file)

    def _read_file(self, input_file):
        with open(input_file, "r", encoding="utf-8") as reader:
            for line in reader:
                item = json.loads(line)

                track = item.get("track")
                if track is not None:
                    row = TrackRow(
                        track["f"],
                        track["p"],
                        track["x"],
                        track["y"],
                        track["h"],
                        track["w"],
                        track["l"],
                        track["rot_z"],
                        track["bb_left"],
                        track["bb_top"],
                        track["bb_width"],
                        track["bb_height"],
                        track.get("prediction_number"),
                        track.get("scene_id"),
                    )
                    self.tracks_by_frame[row.frame].append(row)
                    continue

                scene = item.get("scene")
                if scene is not None:
                    row = SceneRow(
                        scene["id"],
                        scene["p"],
                        scene["s"],
                        scene["e"],
                        scene.get("fps"),
                        scene.get("tag"),
                    )
                    self.scenes_by_id[row.scene] = row

    def scenes(self, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if sample is not None:
            scene_ids = list(scene_ids)
            scene_ids = random.sample(
                scene_ids, int(len(scene_ids) * sample)
            )
        for scene_id in scene_ids:
            yield self._scene(scene_id)

    @staticmethod
    def _paths(primary_pedestrian, track_rows):
        primary_path = []
        other_paths = defaultdict(list)
        for row in track_rows:
            if row.pedestrian == primary_pedestrian:
                primary_path.append(row)
            else:
                other_paths[row.pedestrian].append(row)
        return [primary_path] + list(other_paths.values())

    @staticmethod
    def paths_to_xy(paths):
        frames = {row.frame for row in paths[0]}
        pedestrians = {
            row.pedestrian
            for path in paths
            for row in path
            if row.frame in frames
        }
        paths = [
            path for path in paths if path[0].pedestrian in pedestrians
        ]
        frames = sorted(frames)
        frame_to_index = {
            frame: index for index, frame in enumerate(frames)
        }

        xy = np.full((len(frames), len(pedestrians), 8), np.nan)
        for pedestrian_index, path in enumerate(paths):
            for row in path:
                if row.frame not in frame_to_index:
                    continue
                entry = xy[frame_to_index[row.frame]][pedestrian_index]
                entry[:4] = (row.x, row.y, 0.0, 0.0)
                entry[4:] = (
                    row.bb_left,
                    row.bb_top,
                    row.bb_width,
                    row.bb_height,
                )
        return xy

    def _scene(self, scene_id):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise KeyError(f"Scene {scene_id} was not found")

        track_rows = [
            row
            for frame in range(scene.start, scene.end + 1)
            for row in self.tracks_by_frame.get(frame, [])
        ]
        return scene_id, self._paths(scene.pedestrian, track_rows)


def _read_split(data_root, split, sample=1.0):
    scenes = []
    split_dir = os.path.join(data_root, str(split).strip("/"))
    files = [
        filename
        for filename in os.listdir(split_dir)
        if filename.endswith(".ndjson")
    ]
    for filename in files:
        reader = JrdbReader(os.path.join(split_dir, filename))
        stem = os.path.splitext(filename)[0]
        scenes.extend(
            (stem, scene_id, paths)
            for scene_id, paths in reader.scenes(sample=sample)
        )
    return scenes


def _drop_incomplete_neighbors(xy):
    xy_by_person = np.transpose(xy, (1, 0, 2))
    keep = np.ones(xy_by_person.shape[0], dtype=bool)
    for person_index in range(1, xy_by_person.shape[0]):
        if any(
            np.isnan(xy_by_person[person_index, frame, 0])
            for frame in range(9)
        ):
            keep[person_index] = False
    return np.transpose(xy_by_person[keep], (1, 0, 2))


def _drop_distant_neighbors(xy, radius=6):
    squared_distance = np.sum(
        np.square(xy[:, :, :2] - xy[:, 0:1, :2]), axis=2
    )
    keep = np.nanmin(squared_distance, axis=0) < radius**2
    return xy[:, keep]


def load_jrdb_split(split):
    samples = []
    data_root = path_to_data("jrdb_2dbox")
    for _, _, paths in _read_split(data_root, split, sample=1.0):
        scene = JrdbReader.paths_to_xy(paths)
        scene = _drop_incomplete_neighbors(scene)
        scene = _drop_distant_neighbors(scene)

        scene = scene.reshape(scene.shape[0], scene.shape[1], -1, 4)
        scene = np.transpose(scene, (1, 0, 2, 3))
        mask = np.ones(scene.shape[:-1])
        samples.append((np.asarray(scene), np.asarray(mask)))
    return samples


def collate_batch(batch):
    joints, masks = zip(*batch)
    person_padding = [torch.zeros(item.shape[0]) for item in joints]
    return (
        pad_sequence(joints, batch_first=True),
        pad_sequence(masks, batch_first=True),
        pad_sequence(
            person_padding, batch_first=True, padding_value=1
        ).bool(),
    )


def batch_process_coords(
    coords,
    masks,
    padding_mask,
    config,
    modality_selection="traj+2dbox",
    training=False,
):
    joints = coords.to(config["DEVICE"])
    masks = masks.to(config["DEVICE"])
    input_frames = config["TRAIN"]["input_track_size"]
    output_frames = config["TRAIN"]["output_track_size"]

    joints[:, :, :, 0] = (
        joints[:, :, :, 0]
        - joints[:, 0:1, input_frames - 1:input_frames, 0]
    )
    joints[:, :, :, 1:] = (
        joints[:, :, :, 1:]
        - joints[:, :, input_frames - 1:input_frames, 1:]
    ) * 0.25

    if training:
        joints[:, :, :, 0, :3] = random_rotate_trajectories(
            joints[:, :, :, 0, :3]
        )
    elif modality_selection == "traj":
        joints[:, :, :, 1:] = 0
    elif modality_selection != "traj+2dbox":
        raise ValueError(f"Unknown modality selection: {modality_selection}")

    batch_size, num_people, num_frames, num_modalities, feature_size = (
        joints.shape
    )
    joints = joints.transpose(1, 2).reshape(
        batch_size,
        num_frames,
        num_people * num_modalities,
        feature_size,
    )
    masks = masks.transpose(1, 2).reshape(
        batch_size, num_frames, num_people * num_modalities
    )

    prediction_end = input_frames + output_frames
    return (
        joints[:, :input_frames].float(),
        masks[:, :input_frames].float(),
        joints[:, input_frames:prediction_end].float(),
        masks[:, input_frames:prediction_end].float(),
        padding_mask.float(),
    )


def random_rotate_trajectories(trajectories):
    batch_size = trajectories.shape[0]
    angles = torch.deg2rad(torch.rand(batch_size) * 360)
    rotation = torch.zeros(
        batch_size, 3, 3, device=trajectories.device
    )
    rotation[:, 0, 0] = torch.cos(angles)
    rotation[:, 0, 1] = -torch.sin(angles)
    rotation[:, 1, 0] = torch.sin(angles)
    rotation[:, 1, 1] = torch.cos(angles)
    rotation[:, 2, 2] = 1
    rotated = torch.bmm(
        trajectories.reshape(batch_size, -1, 3).float(), rotation
    )
    return rotated.reshape(trajectories.shape)


class JrdbDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", track_size=21):
        self.samples = []
        for joints, masks in load_jrdb_split(split):
            joints = torch.from_numpy(joints)
            masks = torch.from_numpy(masks)
            num_frames = joints.shape[1]
            for start in range(0, num_frames - track_size + 1, track_size):
                end = start + track_size
                self.samples.append(
                    (joints[:, start:end], masks[:, start:end])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def create_dataset(dataset_name, logger, **kwargs):
    if logger is not None:
        logger.info(f"Loading dataset {dataset_name}")
    if dataset_name != "jrdb_2dbox":
        raise ValueError(f"Unsupported dataset: '{dataset_name}'")
    return JrdbDataset(**kwargs)


def get_datasets(dataset_names, config, logger):
    track_size = (
        config["TRAIN"]["input_track_size"]
        + config["TRAIN"]["output_track_size"]
    )
    return [
        create_dataset(
            dataset_name,
            logger,
            split="train",
            track_size=track_size,
        )
        for dataset_name in dataset_names
    ]


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def dataloader_for(dataset, config, **kwargs):
    shuffle = kwargs.pop("shuffle", False)
    pin_memory = kwargs.pop("pin_memory", True)
    prefetch_factor = kwargs.pop("prefetch_factor", 4)
    num_workers = int(config["TRAIN"]["num_workers"])

    generator = torch.Generator()
    generator.manual_seed(config["SEED"])
    options = dict(
        dataset=dataset,
        batch_size=config["TRAIN"]["batch_size"],
        num_workers=num_workers,
        collate_fn=collate_batch,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        options.update(
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=generator,
            prefetch_factor=prefetch_factor,
        )
    options.update(kwargs)
    return DataLoader(**options)


def validation_dataloader_for(dataset, config, **kwargs):
    shuffle = kwargs.pop("shuffle", False)
    pin_memory = kwargs.pop("pin_memory", True)
    prefetch_factor = kwargs.pop("prefetch_factor", 2)
    num_workers = max(0, int(config["TRAIN"]["num_workers"] // 2))

    generator = torch.Generator()
    generator.manual_seed(config["SEED"])
    options = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_batch,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    options.update(kwargs)
    return DataLoader(**options)
