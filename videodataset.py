import json
from pathlib import Path

from .loader import VideoLoader
import paddle.fluid as fluid
import numpy as np
import copy
from PIL import Image


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDataset(object):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                 root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):

        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.subset = subset

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            # clip = [self.spatial_transform(np.array(img)) for img in clip]
            clip = [self.spatial_transform(img) for img in clip]
        clip = np.stack(clip, 0).transpose([3, 0, 1, 2])

        return clip

    def reader(self):
        if self.subset == 'training':
            np.random.shuffle(self.data)
        for index in range(len(self.data)):
            path = self.data[index]['video']
            if isinstance(self.target_type, list):
                target = [self.data[index][t] for t in self.target_type]
            else:
                target = self.data[index][self.target_type]

            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            clip = self.__loading(path, frame_indices)

            if self.target_transform is not None:
                target = self.target_transform(target)

            yield clip, target
        return self.reader


class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                # clip = [self.spatial_transform(np.array(img)) for img in clip]
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(np.stack(clip, 0).transpose([3, 0, 1, 2]))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def reader(self):
        for index in range(len(self.data)):
            path = self.data[index]['video']

            video_frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                video_frame_indices = self.temporal_transform(video_frame_indices)

            clips, segments = self.__loading(path, video_frame_indices)

            if isinstance(self.target_type, list):
                target = [self.data[index][t] for t in self.target_type]
            else:
                target = self.data[index][self.target_type]

            if 'segment' in self.target_type:
                if isinstance(self.target_type, list):
                    segment_index = self.target_type.index('segment')
                    targets = []
                    for s in segments:
                        targets.append(copy.deepcopy(target))
                        targets[-1][segment_index] = s
                else:
                    targets = segments
            else:
                targets = [target for _ in range(len(segments))]

            yield clips, targets
        return self.reader

def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    return batch_clips, batch_targets


