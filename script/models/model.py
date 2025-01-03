# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from torch.nn import Sequential, Linear, ReLU, GRU, Dropout

import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set #NNConv (implemented MPNN); Set2Set 
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import ChebConv

import math
import torch.nn.init as init
from GIGN import GIGN


class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_size, n_output, dropout=0.2):
        super(GCNModel, self).__init__()
        self.gcn_conv1 = GCNConv(num_features, num_features)
        self.gcn_conv2 = GCNConv(num_features, num_features*2)
        self.gc_fc1 = nn.Linear(num_features*3, hidden_size)
        self.gc_fc2 = nn.Linear(hidden_size, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature, edge_index, weight, protein_batch):
        graph_x = self.gcn_conv1(feature, edge_index, weight)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn_conv2(graph_x, edge_index, weight)
        graph_x = F.relu(graph_x)
        graph_x = torch.cat([graph_x, feature], dim=1)
        gc = global_mean_pool(graph_x, protein_batch)
        gc = self.gc_fc1(gc)
        gc = F.relu(gc)
        gc = self.dropout(gc)
        gc_out = self.gc_fc2(gc)
        gc_out = self.dropout(gc_out)
        return gc_out

class GCNModel02(nn.Module):
    def __init__(self, num_features, hidden_size, n_output, dropout=0.2):
        super(GCNModel02, self).__init__()
        self.gcn_conv1 = GCNConv(num_features, num_features)
        self.gcn_conv2 = GCNConv(num_features, num_features*2)
        self.gcn_conv3 = GCNConv(num_features*2, num_features*4)
        self.gc_fc1 = nn.Linear(num_features*5, hidden_size)
        self.gc_fc2 = nn.Linear(hidden_size, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature, edge_index, weight, protein_batch):
        graph_x = self.gcn_conv1(feature, edge_index, weight)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn_conv2(graph_x, edge_index, weight)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn_conv2(graph_x, edge_index, weight)
        graph_x = F.relu(graph_x)
        graph_x = torch.cat([graph_x, feature], dim=1)
        gc = global_mean_pool(graph_x, protein_batch)
        gc = self.gc_fc1(gc)
        gc = F.relu(gc)
        gc = self.dropout(gc)
        gc_out = self.gc_fc2(gc)
        gc_out = self.dropout(gc_out)
        return gc_out


class ChebModel(nn.Module):
    def __init__(self, num_features, hidden_size, n_output, K, dropout=0.2):
        super(ChebModel, self).__init__()
        self.cheb_conv1 = ChebConv(num_features, num_features, K)
        self.cheb_conv2 = ChebConv(num_features, num_features*2, K)
        # self.cheb_conv3 = ChebConv(num_features*2, num_features*4, K)
        self.gc_fc1 = nn.Linear(num_features*3, hidden_size)
        self.gc_fc2 = nn.Linear(hidden_size, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature, edge_index, protein_batch):
        graph_x = self.cheb_conv1(feature, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = self.cheb_conv2(graph_x, edge_index)
        graph_x = F.relu(graph_x)
        # graph_x = self.gcn_conv3(graph_x, edge_index, weight)
        # graph_x = F.relu(graph_x)
        graph_x = torch.cat([graph_x, feature], dim=1)
        gc = global_mean_pool(graph_x, protein_batch)
        gc = self.gc_fc1(gc)
        gc = F.relu(gc)
        gc = self.dropout(gc)
        gc_out = self.gc_fc2(gc)
        gc_out = self.dropout(gc_out)
        return gc_out

class ChebModel02(nn.Module):
    def __init__(self, num_features, hidden_size, n_output, K, dropout=0.2):
        super(ChebModel02, self).__init__()
        self.cheb_conv1 = ChebConv(num_features, num_features, K)
        self.cheb_conv2 = ChebConv(num_features, num_features*2, K)
        self.cheb_conv3 = ChebConv(num_features*2, num_features*4, K)
        self.gc_fc1 = nn.Linear(num_features*5, hidden_size)
        self.gc_fc2 = nn.Linear(hidden_size, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature, edge_index, protein_batch):
        graph_x = self.cheb_conv1(feature, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = self.cheb_conv2(graph_x, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = self.cheb_conv3(graph_x, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = torch.cat([graph_x, feature], dim=1)
        gc = global_mean_pool(graph_x, protein_batch)
        gc = self.gc_fc1(gc)
        gc = F.relu(gc)
        gc = self.dropout(gc)
        gc_out = self.gc_fc2(gc)
        gc_out = self.dropout(gc_out)
        return gc_out


class GATModel(nn.Module):
    def __init__(self, num_features, hidden_size, n_output, dropout=0.2):
        super(GATModel, self).__init__()
        self.gcn_conv1 = GATConv(num_features, num_features)
        self.gcn_conv2 = GATConv(num_features, num_features*2)
        self.gcn_conv3 = GATConv(num_features*2, num_features*4)
        self.gc_fc1 = nn.Linear(num_features*4, hidden_size)
        self.gc_fc2 = nn.Linear(hidden_size, n_output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature, edge_index, protein_batch):
        graph_x = self.gcn_conv1(feature, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn_conv2(graph_x, edge_index)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn_conv3(graph_x, edge_index)
        graph_x = F.relu(graph_x)
        gc = global_mean_pool(graph_x, protein_batch)
        gc = self.gc_fc1(gc)
        gc = F.relu(gc)
        gc = self.dropout(gc)
        gc_out = self.gc_fc2(gc)
        gc_out = self.dropout(gc_out)
        return gc_out

class CNN_2D(nn.Module):
    def __init__(self, output_dim):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=16, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, output_dim)
        # self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layer, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(num_features, hidden_dim, num_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden):
        gru_out, hidden = self.gru(x, hidden)
        output = self.fc(gru_out)
        output = self.dropout(output)
        output = output.view(output.shape[0], -1)
        return output
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layer, batch_size, self.hidden_dim)

class BIGRUModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layer, output_dim, drop_prob=0.1):
        super(BIGRUModel, self).__init__()
        self.bigru = nn.GRU(num_features, hidden_dim, num_layer, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
    
    def forward(self, x):
        bigru_out, _ = self.bigru(x)
        output = self.fc(bigru_out)
        output = output.view(output.shape[0], -1)
        return output


class Transforer_Encoder(nn.Module):
    def __init__(self, num_features, num_layer, hidden_dim, output_dim, num_heads, fc_middle_dim, dropout=0.2):
        super(Transforer_Encoder, self).__init__()

        self.encoder = make_model(num_features, num_layer, hidden_dim, output_dim, num_heads, dropout)
        self.fc1 = nn.Linear(hidden_dim, fc_middle_dim)
        self.fc2 = nn.Linear(fc_middle_dim * num_features, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encodedSeq, encodedSeq_mask):
       
        seq_mask = encodedSeq_mask.unsqueeze(1)
        seq_encoded = self.encoder(encodedSeq, seq_mask)
        seq_ln = self.fc1(seq_encoded)
        seq_ln = self.relu(seq_ln)
        seq_ln = seq_ln.view(seq_ln.shape[0], -1)
        seq_ln = self.fc2(seq_ln)
        seq_ln = self.dropout(seq_ln)
        return seq_ln
        

    def create_mask(self, src):
        mask_array = np.zeros((src.shape[0], src.shape[1]), dtype=np.int)
        for idx, item in enumerate(src):
            temp = np.zeros(src.shape[1])
            for i_idx, ele in enumerate(item):
                if ele > 0:
                    temp[i_idx] = 1
            mask_array[idx] = torch.LongTensor(temp)
        return mask_array

class MPNN(torch.nn.Module):
  def __init__(self, node_dim=75, dim=64, output_dim=8, dropout=0.2):
    super(MPNN, self).__init__()
    #super(Net, self).__init__(self, input_dimensions,...)

    self.lin0 = torch.nn.Linear(node_dim, dim) #[n-node_dim]
    nn = Sequential(Linear(6, 128),  #[edge_input_dimensions]
                    ReLU(), 
                    Dropout(dropout),
                    Linear(128, dim*dim)) #NN for edge features
    #self.conv = NNConv(dim, dim, nn, aggr = 'add') #agger='add'/n_features or dim size?? [64,64]
    self.conv = NNConv(in_channels=dim, out_channels=dim, nn=nn, aggr='add')
    self.gru = GRU(dim, dim)

    self.set2set = Set2Set(dim, processing_steps=3, num_layers=1) #return data twice dimensionality of input data [64,128]
    self.lin1 = torch.nn.Linear(2*dim, dim)
    self.lin2 = torch.nn.Linear(dim, output_dim)
    #self.lin2 = torch.nn.Linear(dim, 2) #for 2 labels classifications

  def forward(self, x, edge_index, edge_attr, batch): #one of the batches of data

    out = F.relu(self.lin0(x)) #[2430,64]
    h = out.unsqueeze(0) #[1,2430,64]

    for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr)) #[2430,64]
            out, h = self.gru(m.unsqueeze(0), h)
            #out[1,2430,64] and h[1,2430,64]
            out = out.squeeze(0) #[2430,64]

    out = self.set2set(out, batch) #[101,128]
    out = F.relu(self.lin1(out)) #[101,64]
    out = self.lin2(out)#[101,1]

    #2dimensional classification
    #out1 = F.log_softmax(out, dim=1) #for classification 0 or 1
    return out #reshape[101]

class MPNN_Pro(torch.nn.Module):
  def __init__(self, node_dim=75, dim=64, output_dim=8, dropout=0.2):
    super(MPNN_Pro, self).__init__()
    #super(Net, self).__init__(self, input_dimensions,...)

    self.lin0 = torch.nn.Linear(node_dim, dim) #[n-node_dim]
    nn = Sequential(Linear(1, 128),  #[edge_input_dimensions]
                    ReLU(), 
                    Dropout(dropout),
                    Linear(128, dim*dim)) #NN for edge features
    #self.conv = NNConv(dim, dim, nn, aggr = 'add') #agger='add'/n_features or dim size?? [64,64]
    self.conv = NNConv(in_channels=dim, out_channels=dim, nn=nn, aggr='add')
    self.gru = GRU(dim, dim)

    self.set2set = Set2Set(dim, processing_steps=3, num_layers=1) #return data twice dimensionality of input data [64,128]
    self.lin1 = torch.nn.Linear(2*dim, dim)
    self.lin2 = torch.nn.Linear(dim, output_dim)
    #self.lin2 = torch.nn.Linear(dim, 2) #for 2 labels classifications

  def forward(self, x, edge_index, edge_attr, batch): #one of the batches of data
    out = F.relu(self.lin0(x)) #[2430,64]
    h = out.unsqueeze(0) #[1,2430,64]
    
    for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr)) #[2430,64]
            out, h = self.gru(m.unsqueeze(0), h)
            #out[1,2430,64] and h[1,2430,64]
            out = out.squeeze(0) #[2430,64]

    out = self.set2set(out, batch) #[101,128]
    out = F.relu(self.lin1(out)) #[101,64]
    out = self.lin2(out)#[101,1]

    #2dimensional classification
    #out1 = F.log_softmax(out, dim=1) #for classification 0 or 1
    return out #reshape[101]



class SiameseNetwork_sequence(nn.Module):
    def __init__(self, trans_feature, num_layer, num_heads, hidden_dim, fc_middle_dim, output_dim, dropout, device):

        super(SiameseNetwork_sequence, self).__init__()
        self.transFeature = Transforer_Encoder(trans_feature, num_layer, hidden_dim, output_dim, num_heads, fc_middle_dim, dropout).to(device)
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.transFeature.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

   
    # Shared feature extractor
    def forward_feature(self, input, device):
        seqVec = input
        seqMask = self.transFeature.create_mask(seqVec)
        seqMask = torch.from_numpy(seqMask).to(device)
        output = self.transFeature(seqVec, seqMask)
        return output
       
    def forward(self, input1, input2, device):
        output1 = self.forward_feature(input1, device)
        output2 = self.forward_feature(input2, device)
        diff_output = output1 - output2
        output = self.fc(diff_output)
        output = self.dropout(output)
        return output


class SiameseNetwork_ligand(nn.Module):
    def __init__(self, ligand_feature_dim, dim, middle_dim, output_dim, dropout, device):

        super(SiameseNetwork_ligand, self).__init__()
        self.ligandFeature = MPNN(ligand_feature_dim, dim, middle_dim, dropout).to(device)
        self.fc = nn.Linear(middle_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ligandFeature.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Shared feature extractor
    def forward_feature(self, input):
        x, edge_index, edge_attr, batch = input
        output = self.ligandFeature(x, edge_index, edge_attr, batch)
        return output
       
    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        diff_otuput = output1 - output2
        output = self.fc(diff_otuput)
        output = self.dropout(output)
        return output


class SiameseNetwork_contactMap(nn.Module):
    def __init__(self, map_feature_dim, dim, middle_dim, output_dim, dropout, device):

        super(SiameseNetwork_contactMap, self).__init__()
        self.contactFeature = MPNN_Pro(map_feature_dim, dim, middle_dim, dropout).to(device)
        self.fc = nn.Linear(middle_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.contactFeature.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Shared feature extractor
    def forward_feature(self, input):
        x, edge_index, edge_attr, batch = input
        output = self.contactFeature(x, edge_index, edge_attr, batch)
        return output
       
    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output2 = self.forward_feature(input2)
        diff_otuput = output1 - output2
        output = self.fc(diff_otuput)
        output = self.dropout(output)
        return output



class SiameseNetwork_interaction(nn.Module):
    def __init__(self, inter_node_dim, middle_dim, output_dim, dropout, device):

        super(SiameseNetwork_interaction, self).__init__()
        self.interactionFeature = GIGN(inter_node_dim, middle_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.interactionFeature.apply(self.init_weights)
        self.middle_fc = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
        self.interactionFeature.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Shared feature extractor
    def forward_feature(self, input):
        output = self.interactionFeature(input).unsqueeze(1)
        return output
       
    def forward(self, input1, input2):
        output1 = self.forward_feature(input1)
        output1 = self.middle_fc(output1)
        output2 = self.forward_feature(input2)
        output2 = self.middle_fc(output2)
        diff_otuput = output1 - output2
        output = self.dropout(diff_otuput)
        return output


class SiameseNetworkCombinationInter(nn.Module):
    def __init__(self, args, device):
        super(SiameseNetworkCombinationInter, self).__init__()
        self.ligand_feature_dim = args['ligand_feature_dim']
        self.map_feature_dim = args['map_feature_dim']
        self.trans_feature = args['trans_feature']
        self.inter_feature = args['inter_feature']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.fc_middle_dim = args['fc_middle_dim']
        self.dim = args['dim']
        self.middle_dim = args['middle_dim']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']

        # self.siamese_sequence = SiameseNetwork_sequence(self.trans_feature, self.num_layer, self.num_heads, self.hidden_dim, self.fc_middle_dim, self.output_dim, self.dropout, device)
        self.siamese_ligand = SiameseNetwork_ligand(self.ligand_feature_dim, self.dim, self.middle_dim, self.output_dim, self.dropout, device)
        self.siamese_map = SiameseNetwork_contactMap(self.map_feature_dim, self.dim, self.middle_dim, self.output_dim, self.dropout,device)
        self.siamese_interaction = SiameseNetwork_interaction(self.inter_feature, self.middle_dim, self.output_dim, self.dropout,device)
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dim*3, self.output_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim*2, self.output_dim*2),
        )
        self.fc2 = nn.Linear(self.output_dim*2, 1)

    def forward(self, ligand_exp_input, ligand_dock_input,
                      map_exp_input, map_dock_input,
                      inter_exp_input, inter_af_input,
                      device):
        ligandSia = self.siamese_ligand(ligand_exp_input, ligand_dock_input)
        mapSia = self.siamese_map(map_exp_input, map_dock_input)
        # seqSia = self.siamese_sequence(seq_exp_input, seq_dock_input, device)
        interSia = self.siamese_interaction(inter_exp_input, inter_af_input)
        output = torch.cat((ligandSia, mapSia, interSia), 1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = output.squeeze()
        return output



class SiameseNetworkCombination(nn.Module):
    def __init__(self, args, device):
        super(SiameseNetworkCombination, self).__init__()
        self.ligand_feature_dim = args['ligand_feature_dim']
        self.map_feature_dim = args['map_feature_dim']
        self.trans_feature = args['trans_feature']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.fc_middle_dim = args['fc_middle_dim']
        self.dim = args['dim']
        self.middle_dim = args['middle_dim']
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']

        self.siamese_sequence = SiameseNetwork_sequence(self.trans_feature, self.num_layer, self.num_heads, self.hidden_dim, self.fc_middle_dim, self.output_dim, self.dropout, device)
        self.siamese_ligand = SiameseNetwork_ligand(self.ligand_feature_dim, self.dim, self.middle_dim, self.output_dim, self.dropout, device)
        self.siamese_map = SiameseNetwork_contactMap(self.map_feature_dim, self.dim, self.middle_dim, self.output_dim, self.dropout,device)

        self.fc = nn.Sequential(
            nn.Linear(self.output_dim*3, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, 1),
        )

    def forward(self, ligand_exp_input, ligand_dock_input,
                      map_exp_input, map_dock_input,
                      seq_exp_input, seq_dock_input,
                      inter_exp_input, inter_af_input,
                      device):
        ligandSia = self.siamese_ligand(ligand_exp_input, ligand_dock_input)
        mapSia = self.siamese_map(map_exp_input, map_dock_input)
        seqSia = self.siamese_sequence(seq_exp_input, seq_dock_input, device)
        
        output = torch.cat((ligandSia, mapSia, seqSia), 1)
        output = self.fc(output)
        output = output.squeeze()
        return output
