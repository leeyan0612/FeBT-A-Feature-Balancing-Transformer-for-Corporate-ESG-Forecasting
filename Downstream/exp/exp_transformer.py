import pickle
import sklearn.metrics as metrics
import pandas
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models import Transformer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, accuarcy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time

import warnings
import torch.nn as nn
from sklearn.metrics import recall_score

warnings.filterwarnings('ignore')


# Exp类
class Exp(Exp_Basic):
    def __init__(self, args):
        super(Exp, self).__init__(args)

    def _build_model(self):
        # 根据前面设计的参数选择算法模型
        model_dict = {
            'transformer': Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # 是否实验多GPU或GPU
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 根据数据参数读取数据
    def _get_data(self, flag):
        args = self.args
        prex = str(self.args.h_dim) + self.args.pf
        # 选择数据生成类（用于定义怎么构建每一个样本数据）
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'all': Dataset_Custom,
        }
        # 选择出数据生成类
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1  # 时间特征处理参数判断
        # 判断是训练、测试、预测，对应设定DataLoader参数
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            with open('mydata/test_data.pkl'.format(self.args.sed, prex, args.rd), 'rb') as f:
                data = pickle.load(f)
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = True
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        elif flag == 'val':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
            with open('mydata/val_data.pkl'.format(self.args.sed, prex, args.rd), 'rb') as f:
                data = pickle.load(f)
        else:
            shuffle_flag = True
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
            with open('mydata/train_data.pkl'.format(self.args.sed, prex, args.rd), 'rb') as f:
                data = pickle.load(f)
        # 传入参数到数据生成类中处理
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            data=data
        )
        print(flag, len(data_set))
        # DataLoader数据加载器
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    # 优化器
    # def _select_optimizer(self):
    #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    #     return model_optim

    def _select_optimizer(self):
        adapter_params = list(self.model.enc_embedding.adapter1.parameters()) + \
                         list(self.model.enc_embedding.adapter2.parameters()) + \
                         list(self.model.enc_embedding.adapter3.parameters()) + \
                         list(self.model.enc_embedding.adapter4.parameters()) + \
                         list(self.model.dec_embedding.adapter1.parameters()) + \
                         list(self.model.dec_embedding.adapter2.parameters()) + \
                         list(self.model.dec_embedding.adapter3.parameters()) + \
                         list(self.model.dec_embedding.adapter4.parameters())

        model_optim = optim.Adam(adapter_params, lr=self.args.learning_rate)
        return model_optim

    # 定义loss函数
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    # 做验证
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()  # 说明对模型进行测试
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            len_ = len(batch_x)
            # 每次获取batch_size批次大小数据进行输入
            pred, true, _ = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            true = true.long()
            pred = pred.view(len_, self.args.c_out)
            true = true.view(128).long()
            loss = criterion(pred, true)
            total_loss.append(loss.item())  # 保存loss
        total_loss = np.average(total_loss)  # 对训练一次计算一个平均loss
        self.model.train()
        return total_loss

    # 训练
    def train(self, setting):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.enc_embedding.adapter1.parameters():
            param.requires_grad = True
        for param in self.model.enc_embedding.adapter2.parameters():
            param.requires_grad = True
        for param in self.model.enc_embedding.adapter3.parameters():
            param.requires_grad = True
        for param in self.model.enc_embedding.adapter4.parameters():
            param.requires_grad = True
        for param in self.model.dec_embedding.adapter1.parameters():
            param.requires_grad = True
        for param in self.model.dec_embedding.adapter2.parameters():
            param.requires_grad = True
        for param in self.model.dec_embedding.adapter3.parameters():
            param.requires_grad = True
        for param in self.model.dec_embedding.adapter4.parameters():
            param.requires_grad = True
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # 判断是否有checkpoints目录，没有则创建
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # 获取当前时间进行计算一个epoch花费的时间
        time_now = time.time()
        # 获取数据加载的次数
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 提前停止，若loss一段时间没有下降就停止
        # 给模型设置优化器
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # 给模型设置loss
        # 是否使用自动、混合精度加速训练
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # 开始迭代
        accs = []
        mapes = []
        at_lists = []
        pfi_lists = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 用于记录次数
            train_loss = []  # 定义空列表存储loss
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                len_ = len(batch_x)
                # 按batch_-sizes一次数据进行前向传播
                iter_count += 1
                # 本次梯度清零
                model_optim.zero_grad()
                # 处理没batch_size数据函数，输入到transformer模型中计算输入
                pred, true, _ = self._process_one_batch(train_data, batch_x, batch_y,
                                                        batch_x_mark, batch_y_mark)
                pred = pred.view(len_, self.args.c_out)
                true = true.view(len_).long()
                loss = criterion(pred, true)  # 根据每次transformer的输出和真实值作loss
                train_loss.append(loss.item())
                # 打印训练信息
                if (i + 1) % 30 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # 是否使用自动、混合精度加速训练进行梯度更新和反向传播更新参数
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 做反向更新和梯度更新
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 对一个epoch的loss求评价cv
            train_loss = np.average(train_loss)
            # 对该epoch训练完的模型做验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # 对该epoch训练完的模型做测试
            test_loss = self.vali(test_data, test_loader, criterion)
            acc, mape, at_list = self.test(setting)
            accs.append(acc)
            mapes.append(mape)
            at_lists.append(at_list)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(-acc, self.model, path)
            # 判断是否提前停止
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 调整学习率
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
        # 保存模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, accs, mapes, at_lists

    # 对模型测试
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')  # 获取测试数据
        # 模型测试
        self.model.eval()
        # 定义预测值和真实值存储列表
        preds = []
        trues = []
        rela_errs = []
        correct_count = 0  # 正确预测的数量
        correct_count2 = 0
        total_count = 0
        gl_list = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            len_ = len(batch_x)
            # 一次获取batch_size大小的数据进行测试，并返回预测值和真实值
            pred, true, global_feature_importances = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark,
                                                                             batch_y_mark)
            gl_list.append(global_feature_importances.tolist())
            pred = pred.view(len_, self.args.c_out)
            true = true.view(128).long()
            predicted_classes = torch.argmax(pred, dim=1).detach().cpu().numpy()
            true_classes = true.detach().cpu().numpy()
            preds.append(predicted_classes)
            trues.append(true_classes)
        array = np.array(gl_list)
        # 计算每列的平均值
        column_means = array.mean(axis=0)
        # 如果你需要结果为列表形式
        column_means_list = column_means.tolist()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        acc = accuarcy(preds, trues)
        recall = recall_score(trues, preds, average='macro')
        print('Recall: ', recall)
        print('acc10:{}'.format(acc))
        return acc, recall, column_means_list

    def predict(self, setting, load=False):
        # 读取数据
        pred_data, pred_loader = self._get_data(flag='pred')
        # 加载训练的模型
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        # 定义空列表进行存储预测值
        preds = []
        # 进行未来值预测
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            # 一次获取batch_size大小的数据进行测试，并返回预测值和真实值
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())  # 存储预测值

        preds = np.array(preds)  # 数据类型转换
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # 数据降维

        # 保存预测结果为csv
        preds = preds.reshape(preds.shape[0], preds.shape[1])

        # 数据保存
        df_true = pandas.DataFrame(preds.reshape(-1, 1), columns=['feture value'])
        df_true.to_csv('./result/future.csv')

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # 对传入的batch_size的特征数据转为float型并指定计算设备
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        # 对传入的batch_size的时间特征数据转为float型并指定计算设备
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # 构建deconder输入
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # 模型预测（即输入数据计算输出）
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    # transformer模型
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # transformer模型
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                res = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = res[0]
                global_feature_importances = res[1]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 是否要反归一化
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y, global_feature_importances
