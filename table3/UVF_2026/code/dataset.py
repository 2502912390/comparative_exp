import pickle
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

class dataset():
    def __init__(self, datadir="./Datasets"):
        self.datadir = datadir

        self.views = ['view1', 'view2', 'view3', 'view4']

        # Reading Data. Note that this is a sample data
        self.path_to_data = self.datadir + "/sample_data/"
        self.num_class = None

    def data(self):
        X_train = {}
        y_train = {}
        X_test = {}
        y_test = {}

        for name_of_view in self.views:
            # view-specific training data
            X_train_temp = pickle.load(open(self.path_to_data + name_of_view + "/" +
                                            name_of_view + "_X_train.p", "rb"), encoding='latin1')

            y_train_temp = pickle.load(open(self.path_to_data + name_of_view + "/" +
                                            name_of_view + "_y_train.p", "rb"), encoding='latin1')

            # view-specific test data
            X_test_temp = pickle.load(open(self.path_to_data + name_of_view + "/" +
                                           name_of_view + "_X_test.p", "rb"), encoding='latin1')
            y_test_temp = pickle.load(open(self.path_to_data + name_of_view + "/" +
                                           name_of_view + "_y_test.p", "rb"), encoding='latin1')

            # 将数据转为torch.Tensor类型
            X_train[name_of_view] = torch.from_numpy(X_train_temp.toarray()).type(torch.float)
            y_train[name_of_view] = torch.from_numpy(y_train_temp)

            X_test[name_of_view] = torch.from_numpy(X_test_temp.toarray()).type(torch.float)
            y_test[name_of_view] = torch.from_numpy(y_test_temp)

        for i in range(len(y_train["view1"])):
            if y_train["view1"][i] == -1:
                y_train["view1"][i] = 0

        for i in range(len(y_test["view1"])):
            if y_test["view1"][i] == -1:
                y_test["view1"][i] = 0

        return X_test, y_test["view1"], X_train, y_train["view1"]


class sample_data(dataset):
    pass

class dataset2(Dataset):
    """
    load multi-view data
    """

    def __init__(self, datadir="./Datasets", train=True):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(dataset2, self).__init__()
        self.datadir = datadir
        # Reading Data. Note that this is a sample data
        self.num_class = None
        self.root = datadir + "/handwritten_6views/"
        self.train = train
        data_path = self.root + 'handwritten_6views.mat'

        dataset = sio.loadmat(data_path)
        view_number = int((len(dataset) - 5) / 2)
        self.X = dict()
        if train:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']

        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x


class handwritten_6views(dataset2):
    pass

class dataset3():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir

        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']

        # Reading Data. Note that this is a sample data
        self.path_to_data = self.datadir + "/tinyimagenet200/"
        self.num_class = 200

    def data(self):
        X_train = {}
        X_test = {}

        X_train['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121train_X.npy', "rb")))
        X_train['view2'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3train_X.npy', "rb")))
        X_train['view3'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2train_X.npy', "rb")))
        X_train['view4'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50train_X.npy', "rb")))
        X_train['view5'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontrain_X.npy', "rb")))
        y_train = torch.tensor(np.load(open(self.path_to_data + 'train_Y.npy', "rb")))

        X_test['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121test_X.npy', "rb")))
        X_test['view2'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3test_X.npy', "rb")))
        X_test['view3'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2test_X.npy', "rb")))
        X_test['view4'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50test_X.npy', "rb")))
        X_test['view5'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontest_X.npy', "rb")))
        y_test = torch.tensor(np.load(open(self.path_to_data + 'test_Y.npy', "rb")))

        return X_train, y_train, X_test, y_test

class tinyimagenet200(dataset3):
    pass


class YoutubeFace():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        self.path_to_data = self.datadir + "/YoutubeFace/view/"
        self.num_class = 31

    def data(self):
        X = {}
        for i, view in enumerate(self.views):
            X[view] = torch.tensor(np.load(open(self.path_to_data + f'v{i}.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)

        # X_train, y_train = transformed(X_train, y_train)
        return X_train, y_train, X_test, y_test



def transformed(X, y):
    X_transform = {}
    for view, samples in X.items():
        max_label = torch.max(y).item() + 1
        X_by_label = [[] for _ in range(max_label)]

        for i in range(len(samples)):
            label = y[i].item()
            X_by_label[label].append(samples[i])

        X_transform[view] = torch.cat([torch.stack(samples) for samples in X_by_label if samples], dim=0)
    y_transform, _ = torch.sort(y)
    return X_transform, y_transform


class nus_wide():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7']
        self.path_to_data = self.datadir + "/nus_wide/view/"
        self.num_class = 10

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'BoW_int.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CH.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CM55.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CORR.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_EDH.npy', "rb")), dtype=torch.float)
        X['view6'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_WT.npy', "rb")), dtype=torch.float)
        X['view7'] = torch.tensor(np.load(open(self.path_to_data + 'tags1k.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        # X, y = transformed(X, y)
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test




def stratified_split_data(X, y, test_size=0.2):


    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=12)
    idx_split = 0
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(X['view1'], y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    
    X_train = {}
    X_test = {}


    for view in X.keys():
        X_train[view] = X[view][train_idxs[idx_split]]
        X_test[view] = X[view][test_idxs[idx_split]]

    y_train = y[train_idxs[idx_split]]
    y_test = y[test_idxs[idx_split]]

    return X_train, y_train, X_test, y_test

class dataset6():
    def __init__(self, datadir="../Datasets"):
        print("ChemBook-10k")
        self.datadir = datadir

        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7', 'view8', 'view9', 'view10']

        # Reading Data. Note that this is a sample data
        self.path_to_data = self.datadir + "/ChemBook-10k/"
        self.num_class = None

    def data(self):
        X_train = {}
        X_test = {}

        X_train['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121train_X.npy', "rb")))
        X_train['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169train_X.npy', "rb")))
        X_train['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201train_X.npy', "rb")))
        X_train['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3train_X.npy', "rb")))
        X_train['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2train_X.npy', "rb")))
        X_train['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletrain_X.npy', "rb")))
        X_train['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18train_X.npy', "rb")))
        X_train['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34train_X.npy', "rb")))
        X_train['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50train_X.npy', "rb")))
        X_train['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontrain_X.npy', "rb")))

        y_train = torch.tensor(np.load(open(self.path_to_data + 'train_Y.npy', "rb")))

        X_test['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121test_X.npy', "rb")))
        X_test['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169test_X.npy', "rb")))
        X_test['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201test_X.npy', "rb")))
        X_test['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3test_X.npy', "rb")))
        X_test['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2test_X.npy', "rb")))
        X_test['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletest_X.npy', "rb")))
        X_test['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18test_X.npy', "rb")))
        X_test['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34test_X.npy', "rb")))
        X_test['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50test_X.npy', "rb")))
        X_test['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontest_X.npy', "rb")))

        y_test = torch.tensor(np.load( open( self.path_to_data + 'test_Y.npy', "rb")))

        return X_train, y_train, X_test, y_test

class chembook(dataset6):
    pass

class dataset7():
    def __init__(self, datadir="../Datasets"):
        print("ChEMBL-10k")
        self.datadir = datadir

        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7', 'view8', 'view9', 'view10']

        # Reading Data. Note that this is a sample data
        self.path_to_data = self.datadir + "/ChEMBL-10k/view/"
        self.num_class = None

    def data(self):
        X_train = {}
        X_test = {}

        X_train['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121train_X.npy', "rb")))
        X_train['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169train_X.npy', "rb")))
        X_train['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201train_X.npy', "rb")))
        X_train['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3train_X.npy', "rb")))
        X_train['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2train_X.npy', "rb")))
        X_train['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletrain_X.npy', "rb")))
        X_train['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18train_X.npy', "rb")))
        X_train['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34train_X.npy', "rb")))
        X_train['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50train_X.npy', "rb")))
        X_train['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontrain_X.npy', "rb")))

        y_train = torch.tensor(np.load(open(self.path_to_data + 'train_Y.npy', "rb")))

        X_test['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121test_X.npy', "rb")))
        X_test['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169test_X.npy', "rb")))
        X_test['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201test_X.npy', "rb")))
        X_test['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3test_X.npy', "rb")))
        X_test['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2test_X.npy', "rb")))
        X_test['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletest_X.npy', "rb")))
        X_test['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18test_X.npy', "rb")))
        X_test['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34test_X.npy', "rb")))
        X_test['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50test_X.npy', "rb")))
        X_test['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontest_X.npy', "rb")))

        y_test = torch.tensor(np.load( open( self.path_to_data + 'test_Y.npy', "rb")))

        return X_train, y_train, X_test, y_test

class chembl(dataset7):
    pass

class dataset8():
    def __init__(self, datadir="../Datasets"):
        print("PubChem-10k")
        self.datadir = datadir

        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7', 'view8', 'view9', 'view10']

        # Reading Data. Note that this is a sample data
        self.path_to_data = self.datadir + "/PubChem-10k/view/"
        self.num_class = None

    def data(self):
        X_train = {}
        X_test = {}

        X_train['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121train_X.npy', "rb")))
        X_train['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169train_X.npy', "rb")))
        X_train['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201train_X.npy', "rb")))
        X_train['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3train_X.npy', "rb")))
        X_train['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2train_X.npy', "rb")))
        X_train['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletrain_X.npy', "rb")))
        X_train['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18train_X.npy', "rb")))
        X_train['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34train_X.npy', "rb")))
        X_train['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50train_X.npy', "rb")))
        X_train['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontrain_X.npy', "rb")))

        y_train = torch.tensor(np.load(open(self.path_to_data + 'train_Y.npy', "rb")))

        X_test['view1'] = torch.tensor(np.load(open(self.path_to_data + 'desnet121test_X.npy', "rb")))
        X_test['view2'] = torch.tensor(np.load(open(self.path_to_data + 'desnet169test_X.npy', "rb")))
        X_test['view3'] = torch.tensor(np.load(open(self.path_to_data + 'desnet201test_X.npy', "rb")))
        X_test['view4'] = torch.tensor(np.load(open(self.path_to_data + 'InceptionV3test_X.npy', "rb")))
        X_test['view5'] = torch.tensor(np.load(open(self.path_to_data + 'MobileNetV2test_X.npy', "rb")))
        X_test['view6'] = torch.tensor(np.load(open(self.path_to_data + 'NASNetMobiletest_X.npy', "rb")))
        X_test['view7'] = torch.tensor(np.load(open(self.path_to_data + 'resnet18test_X.npy', "rb")))
        X_test['view8'] = torch.tensor(np.load(open(self.path_to_data + 'resnet34test_X.npy', "rb")))
        X_test['view9'] = torch.tensor(np.load(open(self.path_to_data + 'resnet50test_X.npy', "rb")))
        X_test['view10'] = torch.tensor(np.load(open(self.path_to_data + 'Xceptiontest_X.npy', "rb")))

        y_test = torch.tensor(np.load( open( self.path_to_data + 'test_Y.npy', "rb")))

        return X_train, y_train, X_test, y_test



class AWA():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7']
        self.path_to_data = self.datadir + "/AWA1/view/"
        self.num_class = 50

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'cq-hist.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'lss-hist.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'phog-hist.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'res101.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'rgsift-hist.npy', "rb")), dtype=torch.float)
        X['view6'] = torch.tensor(np.load(open(self.path_to_data + 'sift-hist.npy', "rb")), dtype=torch.float)
        X['view7'] = torch.tensor(np.load(open(self.path_to_data + 'surf-hist.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        # X, y = transformed(X, y)
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test



class Reuters5():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        self.path_to_data = self.datadir + "/Reuters5noisy/view/"
        self.num_class = 6

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'EN.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'FR.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'GR.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'IT.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'SP.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        # X, y = transformed(X, y)
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test




class Reuters3():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        self.path_to_data = self.datadir + "/Reuters3noisy/view/"
        self.num_class = 6

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'EN.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'FR.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'GR.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'IT.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'SP.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        # X, y = transformed(X, y)
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test




class VoxCeleb():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        self.path_to_data = self.datadir + "/audio/view/"
        self.num_class = 1251

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'wav_ecapa.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'wav_fbank.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'wav_mfcc.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'wav_resnet.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'wav_spec.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        # X, y = transformed(X, y)
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test









#
# class AdditionModel(nn.Module):
#     def __init__(self, input_size):
#         super(AdditionModel, self).__init__()
#
#         # 将不同视图的维度统一
#         self.weidu = 128
#
#         self.fc_view1 = nn.Linear(input_size[0], self.weidu)
#         self.fc_view2 = nn.Linear(input_size[1], self.weidu)
#         self.fc_view3 = nn.Linear(input_size[2], self.weidu)
#         self.fc_view4 = nn.Linear(input_size[3], self.weidu)
#         self.fc_view5 = nn.Linear(input_size[4], self.weidu)
#         # self.fc_view6 = nn.Linear(input_size[5], self.weidu)
#         # self.fc_view7 = nn.Linear(input_size[6], self.weidu)
#         # self.fc_view8 = nn.Linear(input_size[7], self.weidu)
#         # self.fc_view9 = nn.Linear(input_size[8], self.weidu)
#         # self.fc_view10 = nn.Linear(input_size[9], self.weidu)
#
#         self.cls = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.weidu, 1251),
#         )
#
#     def forward(self, x):
#
#
#         x1 = self.fc_view1(x["view1"])
#         x2 = self.fc_view2(x["view2"])
#         x3 = self.fc_view3(x["view3"])
#         x4 = self.fc_view4(x["view4"])
#         x5 = self.fc_view5(x["view5"])
#         # x6 = self.fc_view6(x["view6"])
#         # x7 = self.fc_view7(x["view7"])
#
#         o = x1 + x2 + x3 + x4 + x5
#
#         o = self.cls(o)
#         return o
#
# dataset = dataset12()
# X_train, y_train, X_test, y_test = dataset.data()
#
#
# # 获取输入尺寸
# input_size = [X_train[key].size(1) for key in X_train.keys()]
# print(input_size)
#
#
# model = AdditionModel(input_size)
#
# # 损失函数
# loss_fn = nn.CrossEntropyLoss()
#
# # 优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练的轮数
# epochs = 50
#
# # 训练过程
# for epoch in range(epochs):
#     print("------第{}轮训练------".format((epoch + 1)))
#     optimizer.zero_grad()
#     output = model(X_train)
#
#     loss = loss_fn(output.squeeze(), y_train.long())
#
#     _, predicted_labels = torch.max(output.data, 1)
#     correct_predictions = (predicted_labels.squeeze() == y_train).sum().item()
#
#     total_samples = len(y_train)
#
#     accuracy = correct_predictions / total_samples
#     print("训练集的Acc:{}".format(accuracy))
#     print(f'训练集的Loss: {loss.item()}\n')
#
#
#     loss.backward()
#     optimizer.step()
#
#     # 测试
#     with torch.no_grad():
#         output = model(X_test)
#         loss = loss_fn(output.squeeze(), y_test.long())
#
#         _, predicted_labels = torch.max(output.data, 1)
#         correct_predictions = (predicted_labels.squeeze() == y_test).sum().item()
#
#         total_samples = len(y_test)
#
#         accuracy = correct_predictions / total_samples
#
#         # 假设 y_test 和 predicted_labels 是 CUDA 张量
#         y_test_cpu = y_test.cpu()
#         predicted_labels_cpu = predicted_labels.cpu()
#
#
#         print(f'测试集Loss: {loss.item()}\n')
#         print("测试集的Acc:{}".format(accuracy))
#         # 计算精确度（Precision）、召回率（Recall）和 F1 分数（F1 Score）
#         precision = precision_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
#         recall = recall_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
#         f1 = f1_score(y_test_cpu.numpy(), predicted_labels_cpu.numpy(), average='weighted')
#
#         print("Precision:", precision, "Recall:", recall, "F1 Score:", f1)














