import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, optimizer, epoch, batch_num, batch_num_val, train_dataloader, val_dataloader, total_epoch, cuda, loss_history):
    total_loss = 0.0
    val_total_loss = 0.0

    total_acc = 0.0
    val_total_acc = 0.0

    # 开始训练
    model.train()
    print('Start Train')

    with tqdm(total=batch_num, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):
            if iteration >= batch_num:
                break
            images, labels = batch
            if cuda:
                images = images.to(device)
                labels = labels.to(device)

            # 优化器梯度置零
            optimizer.zero_grad()

            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            predict_y = torch.max(outputs, dim=1)[1]
            total_acc += torch.eq(predict_y, labels).sum().item()

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'total_acc': total_acc / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # 开始验证
    model.eval()
    print('Start Validation')
    with tqdm(total=batch_num_val, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            if iteration >= batch_num_val:
                break
            images, labels = batch

            # 验证流程不需要梯度
            with torch.no_grad():
                if cuda:
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                predict_y = torch.max(outputs, dim=1)[1]

                val_total_acc += torch.eq(predict_y, labels).sum().item()
                val_total_loss += loss.item()

            pbar.set_postfix(**{'val_total_loss': val_total_loss / (iteration + 1),
                                'total_acc': val_total_acc / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(epoch,total_loss / batch_num, val_total_loss / batch_num_val, total_acc / batch_num, val_total_acc / batch_num_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / batch_num, val_total_loss / batch_num_val))
    print('Total Acc: %.4f || Val Acc: %.4f ' % (total_acc / batch_num, val_total_acc / batch_num_val))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Val_Loss%.4f-Val_Acc%.4f.pth' % (
        (epoch + 1), val_total_loss / batch_num_val, val_total_acc / batch_num_val))
