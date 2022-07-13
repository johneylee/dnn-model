import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import utils

# CTC model
class CTC_model(nn.Module):
  def __init__(self,input_dim=39, output_dim=256, num_class=40, n_layer=2):
    super(CTC_model, self).__init__()
    self.lstm = nn.LSTM(input_dim, output_dim, num_layers=n_layer, batch_first=True)
    self.fc = nn.Linear(output_dim, output_dim)
    self.clf = nn.Linear(output_dim, num_class)

  def forward(self, x):
    x,_ = self.lstm(x)
    x = self.fc(x)
    x = self.clf(x)
    return F.log_softmax(x, dim=-1).transpose(0,1)

##################################################################################

# LAS moddel
class LAS(nn.Module):
  def __init__(self, l_indim, l_outdim, l_layer, s_indim, s_outdim, s_class, s_rnnlayer, l_drop=0, s_drop=0):
    super(LAS, self).__init__()
    
    # sub network : listenr, speller
    self.listener = Listener(l_indim, l_outdim, l_layer)
    self.speller = Speller(s_indim, s_outdim, s_class, s_rnnlayer, l_outdim)
    
  def forward(self, x, ground_truth=None, tf_rate=0.9, train=True):
    # get acoustic feature h
    h = self.listener(x)
    
    # get word prediction using acoustic feature h
    pred, att , gt= self.speller(h, ground_truth, tf_rate, train)
    return pred, att, gt

# Prymidal LSTM
class LSTM(nn.Module):
  def __init__(self, in_dim, out_dim, num_ly, dropout_rate=0.0):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(in_dim, out_dim, batch_first=True, dropout=dropout_rate)
    self.num_ly = num_ly

  def forward(self, x):
    batch, time, feat = x.size()
    if self.num_ly != 0:
        odd_index=[]
        even_index=[]
        for i in range(time//2):
          # reduce the time resolution by half
          odd_index.append(2*i)
          even_index.append(2*i+1)
        x = torch.cat((x[:,odd_index,:], x[:,even_index,:]),dim=-1)
    x, hidden = self.lstm(x)
    return x

# Listener network
class Listener(nn.Module):
  def __init__(self, input_dim, output_dim, n_layers, dropout_rate=0.0):
    super(Listener, self).__init__()
    modules = []
    # define n_layers of prymidal LSTM
    for i in range(n_layers):
        if i == 0:
          modules.append(LSTM(input_dim, output_dim, i, dropout_rate))
        else:
          modules.append(LSTM(output_dim*2, output_dim, i, dropout_rate))
    self.layers = nn.Sequential(*modules)
  
  def forward(self, x):
    x = self.layers(x)
    return x

# Speller network
class Speller(nn.Module):
  def __init__(self, in_dim, out_dim, num_class, rnn_layers, listener_dim):
    super(Speller, self).__init__()
    # define speller lstm
    self.lstm = nn.LSTM(in_dim, out_dim, num_layers=rnn_layers, batch_first=True)
    
    # define speller FC for character distribution
    self.character_distribution = nn.Linear(out_dim*2, num_class)
    
    # define attention network
    self.attention = Attention()
    self.max_step=100
    self.outdim = out_dim
    self.num_class = num_class
    
  # define step forward for sequence to sequence style network.
  def forward_step(self, input_word, last_hidden, listener_feat):
    # take input as the word of previous tiem step and hidden
    rnn_output, hidden = self.lstm(input_word, last_hidden)
    
    # get attention and context c_i from rnn_output s_i and listener features h
    attention_score, context = self.attention(rnn_output, listener_feat)
    
    # predict the word character from rnn_out s_i and context c_i
    con_feat = torch.cat((rnn_output, context),dim=-1)
    pred = F.softmax(self.character_distribution(con_feat), dim=-1)
    return pred, hidden, context, attention_score
  
  def forward(self, listener_feat, ground_truth=None, tf_rate=1.0, train=True):
    
    word_pred=[]
    att_scores=[]
    
    tf = True if tf_rate>0 else False
    
    # preprocess the labels : concat <sos> and <eos> to the label
    # in_gt : input labels for teacher forcing - <sos>+labels
    # out_gt : output labels for loss calculation - labels + <eos>
    in_gt, out_gt = self.labels_processing(ground_truth)
    
    # get the one-hot encoded labels
    in_gt_one = utils.onehot(in_gt, self.num_class) 
    out_gt_one = utils.onehot(out_gt, self.num_class) 

    # get the input of first - <sos>
    rnn_input, hidden = self.initialize(in_gt_one)
    
    if tf:
        max_step = out_gt.size(1)
    else:
        max_step = self.max_step
   
    # decode the sequence to sequence network
    for step in range(max_step):
        # forward each time step
        pred, hidden, context, att_score = self.forward_step(rnn_input, hidden, listener_feat)
        word_pred.append(pred)
        att_scores.append(att_score)
        if tf:
            # need to implement the adapt tf-rate version
            # make next step input with ground truth for teacher forcing
            rnn_input = torch.cat((in_gt_one[:,step,:].unsqueeze(1), context), dim=-1)
        else:
            # make next step input with predicted output of current step
            rnn_input = torch.cat((pred, context), dim=-1)
        
        # stop criteiron for decoding
        #if the predicted output is <eos>, stop decoding..
        #if not train and (torch.argmax(pred, dim=-1)==42):
            #print('find <eos>')
            #return torch.cat(word_pred, dim =1), torch.cat(att_scores, dim=1), out_gt
            
    return torch.cat(word_pred, dim =1), torch.cat(att_scores, dim=1), out_gt

  # function for making first input
  def initialize(self, in_gt):
    batch_size = in_gt.size(0)
    in_word = in_gt[:,0,:].view(batch_size, 1,-1)
    
    context = torch.zeros((batch_size, 1, self.outdim)).to(in_gt.device)
    rnn_input = torch.cat((in_word, context), dim=-1)
    hidden = None
    return rnn_input, hidden

  # function for label processing : add <sos>, <eos>
  def labels_processing(self, ground_truth):
        tensor = torch.tensor((), dtype=ground_truth.dtype).to(ground_truth.device)
        tensor = tensor.new_full((len(ground_truth),1),41)  ### <sos>
        in_gt = torch.cat((tensor, ground_truth), dim=-1)
        
        tensor = torch.tensor((), dtype=ground_truth.dtype).to(ground_truth.device)
        tensor = tensor.new_full((len(ground_truth),1),42)  ### <eos>
        out_gt = torch.cat((ground_truth, tensor), dim=-1)
        
        return in_gt, out_gt
        
# Attention netowork
class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()
    # dot product based attention method
    
  def forward(self, decoder_state, listener_feat):
    """
    Args:
      decoder_state : N * 1 * D
      listner_feat  : N * T_i * D
    """
    attention_scores = torch.bmm(decoder_state, listener_feat.transpose(1,2))
    attention_scores = F.softmax(attention_scores,dim=-1)
    attention_out = torch.bmm(attention_scores, listener_feat)
    return attention_scores, attention_out
    
    
