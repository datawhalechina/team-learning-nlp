import os
import time

import torch
import torch.nn.functional as F
from sklearn import metrics

class Solver(object):
    """
    """
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        self.model_choose = args.model_choose
        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.epochs = args.epochs
        # save and load model

        self.save_folder = os.path.join(args.save_folder, args.model_choose)
        is_exists = os.path.exists(self.save_folder)
        if not is_exists:
            os.mkdir(self.save_folder)
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        # self.model_path = args.model_path
        self.model_path = args.model_path

        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_lr = args.visdom_lr
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)
            self.optimizer.set_visdom(self.visdom_lr, self.vis)

        self._reset()


    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_acc = float("-inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss, _ = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # file_path = os.path.join(self.save_folder, self.model_path)  # + str(epoch+1)
            # torch.save(self.serialize(self.model, self.optimizer, epoch + 1,
            #                                 self.LFR_m, self.LFR_n,
            #                                 tr_loss=self.tr_loss,
            #                                 cv_loss=self.cv_loss),
            #            file_path)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss, val_acc = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_acc > self.best_val_acc:
            #if val_loss != self.best_val_loss:
                self.best_val_acc = val_acc
                file_path = os.path.join(self.save_folder, self.model_path)#+ str(epoch+1)
                torch.save(serialize(self.model, self.optimizer, epoch + 1,
                                                self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)
                print('######################################################')

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        sum_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        print('batch length:', len(data_loader))
        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        total_correct = 0
        total_sen = 0
        labels_cat = torch.empty(0, dtype=torch.int64)
        predicted_cat = torch.empty(0, dtype=torch.int64)

        target_names = ['aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots', 'chips',
                  'chocolate', 'drinks', 'fries', 'grapes', 'gummies', 'ice-cream',
                  'jelly', 'noodles', 'pickles', 'pizza', 'ribs', 'salmon',
                  'soup', 'wings']  # 这里这里这里#
        for i, (data) in enumerate(data_loader):
            padded_input, input_lengths, labels = data
            if len(input_lengths) <= 1:
                continue
            total_sen = total_sen + padded_input.size(0)

            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            labels = labels.cuda()

            pred = self.model(padded_input, input_lengths)
            model_out = pred[0]

            loss = F.cross_entropy(model_out, labels, reduction='sum')
            sum_loss = sum_loss + loss.item()
            loss = loss / padded_input.size(0)
            pred_res = model_out.max(1)[1]
            gold = labels.contiguous().view(-1)
            n_correct_res = pred_res.eq(gold)
            n_correct_res = n_correct_res.sum().item()
            total_correct = total_correct + n_correct_res

            predicted_cat = torch.cat((predicted_cat, pred_res.cpu()), -1)
            labels_cat = torch.cat((labels_cat, labels.cpu()), -1)

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # if self.model_choose == 'speaker_classify' and not cross_valid:
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')


        print('n_correct:', total_correct)
        print('total_sen:', total_sen)
        print('acc:', total_correct/total_sen)
        print('每个batch的平均损失相加:', total_loss / (i + 1))
        print('每个batch的损失相加后再平均:', sum_loss / total_sen)
        print(metrics.classification_report(labels_cat, predicted_cat, target_names=target_names, digits=4))
        return sum_loss / total_sen, total_correct/total_sen


#@staticmethod
def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
    package = {
        # Low Frame Rate Feature
        'LFR_m': LFR_m,
        'LFR_n': LFR_n,
        # state
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if tr_loss is not None:
        package['tr_loss'] = tr_loss
        package['cv_loss'] = cv_loss
    return package


