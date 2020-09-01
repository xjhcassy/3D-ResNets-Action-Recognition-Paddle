from pathlib import Path
import json
import random
import os
import time
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import ReduceLROnPlateau, MultiStepDecay
import paddle
from opts import parse_opts
from model import (generate_model, load_pretrained_model,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ScaleValue, ToArray,
                                PickFirstChannels)
from hapi.vision.transforms.transforms import ColorJitter
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger
import inference
from training import train_epoch
from validation import val_epoch


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    print(opt)
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    spatial_transform.append(ToArray())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   spatial_transform, temporal_transform)
    train_loader = paddle.batch(train_data.reader, batch_size=opt.batch_size)


    train_logger = Logger(opt.result_path / 'train.log',
                            ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        opt.result_path / 'train_batch.log',
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            learning_rate=opt.learning_rate, mode='min', patience=opt.plateau_patience)
    else:
        scheduler = MultiStepDecay(learning_rate=opt.learning_rate,
                                             milestones=opt.multistep_milestones)

    optimizer = fluid.optimizer.MomentumOptimizer(
        learning_rate=scheduler,
        momentum=opt.momentum,
        parameter_list=model_parameters,
        use_nesterov=opt.nesterov,
        regularization=fluid.regularizer.L2Decay(regularization_coeff=opt.weight_decay)
    )

    return (train_loader, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToArray(),
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    val_loader = paddle.batch(val_data.reader, batch_size=opt.batch_size)

    val_logger = Logger(opt.result_path / 'val.log',
                        ['epoch', 'loss', 'acc'])

    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
        spatial_transform.append(ToArray())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, spatial_transform,
        temporal_transform)

    inference_loader = paddle.batch(inference_data.reader, batch_size=opt.inference_batch_size)

    return inference_loader, inference_data.class_names

def resume_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    para_dict, opt_dict = fluid.dygraph.load_dygraph(str(resume_path))
    model.set_dict(para_dict)
    return model

def resume_train_utils(resume_path, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    begin_epoch = int(str(resume_path).split('_')[1])
    para_dict, opt_dict = fluid.dygraph.load_dygraph(str(resume_path))

    if optimizer is not None and opt_dict is not None:
        optimizer.set_dict(opt_dict)
    if scheduler is not None and 'LR_Scheduler' in opt_dict:
        scheduler.set_dict(opt_dict['LR_Scheduler'])
    return begin_epoch, optimizer, scheduler

def save_checkpoint(save_file_path, model, optimizer):
    model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    fluid.dygraph.save_dygraph(model_state_dict, save_file_path + '/resModel')
    fluid.dygraph.save_dygraph(opt_state_dict, save_file_path + '/resModel')


def main(opt):
    place = fluid.CPUPlace() if opt.no_cuda else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        print(place)
        random.seed(opt.manual_seed)
        np.random.seed(opt.manual_seed)
        prog = fluid.default_main_program()
        prog.global_seed(opt.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(opt.manual_seed)

        model = generate_model(opt)
        if opt.pretrain_path:
            model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                          opt.n_finetune_classes)
        
        if opt.resume_path is not None:
            model = resume_model(opt.resume_path, model)
        
        if opt.pretrain_path:
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
        else:
            parameters = model.parameters()

        if not opt.no_train:
            (train_loader, train_logger, train_batch_logger,
             optimizer, scheduler) = get_train_utils(opt, parameters)
            if opt.resume_path is not None:
                opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                    opt.resume_path, optimizer, scheduler)
                if opt.overwrite_milestones:
                    scheduler.milestones = opt.multistep_milestones

        if not opt.no_val:
            val_loader, val_logger = get_val_utils(opt)

        best_acc =0.88
        for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                train_epoch(epoch, train_loader, model, optimizer, scheduler,
                        train_logger, train_batch_logger)
               
                if epoch % opt.checkpoint == 0:
                    save_file_path = str(opt.result_path) + 'save_{}_{}_{}'.format(epoch, opt.train_crop, opt.batch_size)
                    save_checkpoint(save_file_path, model, optimizer)
            
            if not opt.no_val:
                prev_val_loss, val_acc = val_epoch(epoch, val_loader, model,
                                      val_logger)
            
            if not opt.no_train and opt.lr_scheduler == 'multistep':
                scheduler.epoch()
            elif not opt.no_train and opt.lr_scheduler == 'plateau':
                scheduler.step(prev_val_loss)

            if not opt.no_val:          
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_file_path = str(opt.result_path) + 'save_{}_{}_best_val_acc'.format(epoch, opt.train_crop)
                    save_checkpoint(save_file_path, model, optimizer)     

            if not opt.no_train:
                current_lr = optimizer.current_step_lr()
                print("current val_loss is %s, current lr is %s" % (prev_val_loss.numpy()[0], current_lr))
        

        if opt.inference:
            inference_loader, inference_class_names = get_inference_utils(opt)
            inference_result_path = opt.result_path / '{}_{}.json'.format(
                opt.inference_subset, opt.train_crop)

            inference.inference(inference_loader, model, inference_result_path,
                                inference_class_names, opt.inference_no_average,
                                opt.output_topk)


if __name__ == '__main__':
    opt = get_opt()
    main(opt)



