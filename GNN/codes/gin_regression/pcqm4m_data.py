import os
import os.path as osp

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset
import shutil

RDLogger.DisableLog('rdApp.*')


class MyPCQM4MDataset(Dataset):

    def __init__(self, root):
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'
        super(MyPCQM4MDataset, self).__init__(root)

        filepath = osp.join(root, 'raw/data.csv.gz')
        data_df = pd.read_csv(filepath)
        self.smiles_list = data_df['smiles']
        self.homolumogap_list = data_df['homolumogap']

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.move(osp.join(self.root, 'pcqm4m_kddcup2021/raw/data.csv.gz'), osp.join(self.root, 'raw/data.csv.gz'))

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, homolumogap = self.smiles_list[idx], self.homolumogap_list[idx]
        graph = smiles2graph(smiles)
        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        y = torch.Tensor([homolumogap])
        num_nodes = int(graph['num_nodes'])
        data = Data(x, edge_index, edge_attr, y, num_nodes=num_nodes)
        return data

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
        return split_dict


if __name__ == "__main__":
    dataset = MyPCQM4MDataset('dataset')
    from torch_geometric.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for batch in tqdm(dataloader):
        pass
