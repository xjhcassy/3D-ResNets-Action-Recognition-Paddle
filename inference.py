import time
import json
from collections import defaultdict

import numpy as np
import paddle.fluid as fluid
from utils import AverageMeter
from datasets.videodataset import collate_fn

def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = fluid.layers.topk(outputs,
                                            k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.shape[0]):
        video_results.append({
            'label': class_names[locs.numpy()[i].item()],
            'score': sorted_scores.numpy()[i].item()
        })

    return video_results


def inference(data_loader, model, result_path, class_names, no_average,
              output_topk):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time()

    for batch_id, data in enumerate(data_loader()):
        data_time.update(time.time() - end_time)

        inputs, targets = collate_fn(data)
        video_ids, segments = zip(*targets)
        inputs = fluid.dygraph.to_variable(np.array(inputs).astype('float32'))

        outputs = model(inputs)
        for j in range(outputs.shape[0]):
            results['results'][video_ids[j]].append({
                'segment': segments[j],
                'output': outputs[j]
            })

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_id + 1) % 100 == 0:
            print('[{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                batch_id + 1,
                batch_time=batch_time,
                data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = fluid.layers.stack(video_outputs)
            average_scores = fluid.layers.reduce_mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)

    print('inference done')

