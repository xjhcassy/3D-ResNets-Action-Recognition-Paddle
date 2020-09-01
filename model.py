from models import resnet
import paddle.fluid as fluid

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append(v)

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        with fluid.dygraph.guard():
            para_dict, _ = fluid.dygraph.load_dygraph(str(pretrain_path))
            model.set_dict(para_dict)
            if model_name == 'densenet':
                model.classifier = fluid.dygraph.Linear(model.last_feature_size,
                                                 n_finetune_classes, act='softmax')
            else:
                model.fc = fluid.dygraph.Linear(model.last_feature_size,
                                         n_finetune_classes, act='softmax')

    return model


