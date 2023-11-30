import argparse
import os
import numpy as np
import torch
from exp.exp_transformer import Exp
from utils.metrics import plot_accs, plot_rec, plot_att, plot_pfi, plot_mape

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1337)
torch.manual_seed(1337)
# 忽略警告
import warnings

warnings.filterwarnings("ignore")
# 如果没有这个目录，则创建这个目录
if not os.path.isdir('result'):
    os.mkdir('result')

parser = argparse.ArgumentParser(description='[transformer] Long Sequences Forecasting')  # 描述
parser.add_argument('--model', type=str, required=False, default='transformer', help='[transformer]')  # 可选的模型
parser.add_argument('--data', type=str, required=False, default='all', help='data')  # 数据名
parser.add_argument('--root_path', type=str, default='./mydata/', help='root path of the data file')  # 数据目录
parser.add_argument('--data_path', type=str, default='filtered_data_list.pkl', help='data file')
# 预测任务：S为单变量预测，M为多输出多输出，MS为多输入单输出
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # 要预测的目标列
# 对enconder编码的时间特征评率
parser.add_argument('--freq', type=str, default='y',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')  # 模型检查点
# enconder输入序列长度2
parser.add_argument('--seq_len', type=int, default=6, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=3, help='start token length of Informer decoder')  # deconder输入的长度
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')  # 预测长度

parser.add_argument('--enc_in', type=int, default=96,
                    help='encoder input size')  # 编码器输入特征维度大小,这个对应'datam':{'data':'datam.csv','T':'close','M':[321,321,321],'S':[1,1,1],'MS':[5,5,1]},中的设置，可直接在后面设置
parser.add_argument('--dec_in', type=int, default=96,
                    help='decoder input size')  # 解码器输入特征维度大小这个对应'datam':{'data':'datam.csv','T':'close','M':[321,321,321],'S':[1,1,1],'MS':[5,5,1]},中的设置，可直接在后面设置
parser.add_argument('--c_out', type=int, default=3,
                    help='output size')  # 输出维度大小，这个根据自己的预测任务来改，比如多输入单输出的就为1，这个对应'datam':{'data':'datam.csv','T':'close','M':[321,321,321],'S':[1,1,1],'MS':[5,5,1]},中的设置，可直接在后面设置
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')  # Emberding维度，把所以维度emberding到同一维度
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')  # 注意力头数
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')  # 编码器层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')  # 解码器层数

parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')  # 输出全连接层的维数
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')  # 注意力因子
parser.add_argument('--padding', type=int, default=0, help='padding type')  # padding填充类型，默认为0
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')  # dropout占比
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')  # 时间特征可选参数
parser.add_argument('--activation', type=str, default='gelu', help='activation')  # 激活函数
parser.add_argument('--output_attention', action='store_true', default='True',
                    help='whether to output attention in ecoder')  # 是否输出编码器注意力结果
parser.add_argument('--do_predict', action='store_true', default=False,
                    help='whether to predict unseen future data')  # 是否做未来预测
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder',
                    default=True)  # 解码器特征混合

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # 多线程，Windows运行不可修改
parser.add_argument('--itr', type=int, default=1, help='experiments times')  # 进行实验的次数
parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')  # 每次实验训练的次数
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size of train input data')  # 训练时取数据按多少样本进行取并进行一次网络参数更新
parser.add_argument('--patience', type=int, default=30,
                    help='early stopping patience')  # loss不更新早停止，如果经过patience次loss不变，则停止训练
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')  # 学习率
parser.add_argument('--des', type=str, default='test', help='exp description')  #
parser.add_argument('--loss', type=str, default='mse', help='loss function')  # loss函数
parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')  # 调整学习率
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                    default=False)  # 是否使用自动、混合精度加速训练
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)  # 对输出数据进行反归一化

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')  # 是否使用GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu')  # 单个gpu时对应的设备id，单个gpu为0
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)  # 多块gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')  # 多gpu的id
parser.add_argument('-h_dim', type=int, default=16, help='device ids of multile gpus')  # 多gpu的id
parser.add_argument('-pf', type=str, default="评级", help='device ids of multile gpus')  # 多gpu的id
parser.add_argument('-sed', type=int, default=102, help='device ids of multile gpus')  # 多gpu的id
parser.add_argument('-rd', type=int, default=0, help='device ids of multile gpus')  # 多gpu的id
# 获取以上参数
args = parser.parse_args()
# 判断gpu是否可用
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# 判断多gpu是否使用
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    # 数据的格式（换成自己的数据需要改，如'CCF':{'data':'CCF.csv','T':'close(目标特征)','M':[5,5,5]（输入5个特征，输出5个特征）,'S':[1,1,1],'MS':[12,12,1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},

}
# 处理数据参数
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)
acc_list = []
acc5_list = []
recall_list = []
for i in range(5):
    args.rd = 0
    # 实例transformer算法
    Exp = Exp
    # 实验开始
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_mx{}_{}_{}'.format(args.model,
                                                                                                   args.data,
                                                                                                   args.features,
                                                                                                   args.seq_len,
                                                                                                   args.label_len,
                                                                                                   args.pred_len,
                                                                                                   args.d_model,
                                                                                                   args.n_heads,
                                                                                                   args.e_layers,
                                                                                                   args.d_layers,
                                                                                                   args.d_ff,
                                                                                                   args.factor,
                                                                                                   args.embed,
                                                                                                   args.mix, args.des,
                                                                                                   ii)
        # 传入设置的参数
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model, accs, mapes, at_lists = exp.train(setting)
        # 打开文件
        with open('columns.txt', 'r') as f:
            # 读取文件内容
            columns_str = f.read()
        # 将字符串转换为列表
        columns_list = columns_str.split(', ')
        transposed_lists = list(map(list, zip(*at_lists)))  # 训练
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        acc, recall, column_means_list = exp.test(setting)  # 测试
        acc_list.append(acc)
        recall_list.append(recall)
        # 预测，需要把do_predict设置为True
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
print(acc_list)
print(recall_list)
