import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')

# train_loader
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    # criterion指的是评价指标
    criterion = hlpr.task.criterion

    model.train()

    # data 里面含有图像数据(input,tensor类型)和标签(labels,tensor类型)
    for i, data in enumerate(train_loader):
        # 感觉是获取
        batch = hlpr.task.get_batch(i, data)
        # 把梯度设置成0，在计算反向传播的时候一般都会这么操作，原因未知
        model.zero_grad()
        # 主要进行攻击的代码
        # 可以看blind backdoor xxx
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()
        # 使用optimizer.step()之后，模型才会更新
        optimizer.step()
        # 打印的函数
        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        for i, data in enumerate(hlpr.task.test_loader):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)


def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    # 获得global的模型
    global_model = hlpr.task.model
    # 获得local的模型
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()
    # tqdm是python的进度条库，基本是基于对象迭代
    for user in tqdm(round_participants):
        # 将参数从global_model复制到local_model
        hlpr.task.copy_params(global_model, local_model)
        # 一个对象，会保存当前状态，并根据梯度更新参数
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            # 如果是恶意的用户，则执行进攻的训练
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            # 如果是非恶意的用户，则执行非进攻的训练
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
        # 然后来更新global的模型
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        # 如果用户是恶意用户，还会更新梯度
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        # 存疑，感觉是积累当前的权重变化
        hlpr.task.accumulate_weights(weight_accumulator, local_update)
    # 所有用户完成之后，更新全局的模型
    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':
    # 将所有的命令行参数都解读到parser
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()
    # 第二个参数设定的.yaml最重要，name只是确认你创建的文件的名字
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name
    # Helper读取params
    helper = Helper(params)
    logger.warning(create_table(params))
    
    try:
        # 参数fl来自于cifar_fed.yaml
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper)
    except (KeyboardInterrupt, RuntimeError):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
