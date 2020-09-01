import time
import sys
import numpy as np
import paddle.fluid as fluid
from utils import AverageMeter, calculate_accuracy
from datasets.videodataset import collate_fn

def val_epoch(epoch,
              data_loader,
              model,
              logger,
              ):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    total_loss = 0
    for batch_id, data in enumerate(data_loader()):
        inputs, targets = collate_fn(data)
        inputs = fluid.dygraph.to_variable(np.array(inputs).astype('float32'))
        targets = fluid.dygraph.to_variable(np.array(targets).astype('int')[:, np.newaxis])
        targets.stop_gradient = True

        # 计算网络输出结果
        outputs = model(inputs)
        # 计算网络预测精度
        loss = fluid.layers.cross_entropy(outputs, targets)
        avg_loss = fluid.layers.mean(loss)
        total_loss += avg_loss
        acc = fluid.layers.accuracy(input=outputs, label=targets)

        accuracies.update(acc.numpy()[0], inputs.shape[0])
        losses.update(avg_loss.numpy()[0], inputs.shape[0])

        if (batch_id + 1) % 10 == 0:
            print('Epoch: [{0}] Step: [{1}]\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                  'Acc {acc.val:.5f} ({acc.avg:.5f})'.format(
                epoch,
                batch_id + 1,
                loss=losses,
                acc=accuracies))

        # if batch_id % 10 == 0:
        #     print("Loss at step {}: {}, acc: {}".format(batch_id, avg_loss.numpy()[0], acc.numpy()[0]))
    print("验证集准确率为:{}".format(accuracies.avg))

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return total_loss / (batch_id + 1) , accuracies.avg

