import time
import os
import sys
import numpy as np
import paddle.fluid as fluid
from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch,
                data_loader,
                model,
                optimizer,
                scheduler,
                epoch_logger,
                batch_logger,
                ):
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch_id, data in enumerate(data_loader()):
        inputs = np.array([x[0] for x in data]).astype('float32')
        targets = np.array([[x[1]] for x in data]).astype('int')
        inputs = fluid.dygraph.to_variable(inputs)
        targets = fluid.dygraph.to_variable(targets)
        targets.stop_gradient = True


        # 计算网络输出结果
        outputs = model(inputs)

        # 计算网络输出和标签的交叉熵损失
        loss = fluid.layers.cross_entropy(outputs, targets)
        avg_loss = fluid.layers.mean(loss)

        # 计算网络预测精度
        # acc =  calculate_accuracy(outputs, targets)
        acc = fluid.layers.accuracy(input=outputs, label=targets)
        losses.update(avg_loss.numpy()[0], inputs.shape[0])
        accuracies.update(acc.numpy()[0], inputs.shape[0])

        
        avg_loss.backward()
        optimizer.minimize(avg_loss)
        optimizer.clear_gradients()


        # if batch_id % 10 == 0:
        #     print("Loss at epoch {} step {}: {}, acc: {}".format(epoch, batch_id, avg_loss.numpy()[0], acc.numpy()[0]))

        if (batch_id + 1) % 10 == 0:
            print('Epoch: [{0}] Step: [{1}]\t'
                'Loss {Loss.val:.5f} ({Loss.avg:.5f})\t'
                'Acc {Acc.val:.5f} ({Acc.avg:.5f})'.format(epoch,
                                                            batch_id + 1,
                                                            Loss=losses,
                                                            Acc=accuracies))
        current_lr = optimizer.current_step_lr()
        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': batch_id + 1,
                'iter': (epoch - 1) * 128 + (batch_id + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })
    

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

        
