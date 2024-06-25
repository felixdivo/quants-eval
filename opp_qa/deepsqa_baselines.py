from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import torch.nn.functional as F

# Baseline model: Prior and Prior-Q solution
def baseline_prior_q(test_data_path, data_ind=None):
    if data_ind is None:
        df = pd.read_pickle(test_data_path)
    else:
        df = pd.read_pickle(test_data_path)
        df = df.iloc[data_ind]
    
    ans_dict_q = {}

    for i in df.question_family_index.unique():
        answer_count_i = df[df.question_family_index==i]['answer'].value_counts()
        ans_dict_q[str(i)] = answer_count_i.keys()[0]
        
    ans_prior = df.answer.value_counts().keys()[0]
        
    return ans_dict_q, ans_prior

def run_baseline_prior(test_data_path, dict_prior_q, dict_prior, data_ind=None):
    if data_ind is None:
        df = pd.read_pickle(test_data_path)
    else:
        df = pd.read_pickle(test_data_path)
        df = df.iloc[data_ind]
    df = df.reset_index()

    prior_ans = np.full(df.shape[0], dict_prior)
    prior_result = (prior_ans == df.answer.values)

    prior_q_ans = np.full(df.shape[0], dict_prior)

    for q_family_i in dict_prior_q.keys():
        id_i = df[df.question_family_index == int(q_family_i)].index
        prior_q_ans[id_i] = dict_prior_q[q_family_i]
    
    prior_q_result = (prior_q_ans == df.answer.values)

    return prior_result, prior_q_result

# Baseline model: DL based methods: CNN, LSTM, Stacked Attention, etc.
class Word2VecModel(nn.Module):
    def __init__(self, embedding_matrix, num_words, embedding_dim, num_hidden_lstm, output_dim, dropout_rate):
        super(Word2VecModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embedding_dim, num_hidden_lstm, batch_first=True, dropout=dropout_rate, num_layers=2)
        self.fc = nn.Linear(num_hidden_lstm, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        # x, _ = self.lstm(x)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Get the last hidden state of the last LSTM layer
        x = self.fc(x)
        x = self.relu(x)
        return x

class SensoryModel(nn.Module):
    def __init__(self, dim, channel, num_feat_map, num_hidden_lstm, output_dim, dropout_rate):
        super(SensoryModel, self).__init__()
        self.conv1 = nn.Conv2d(channel, num_feat_map, kernel_size=(1, 3), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(num_feat_map, num_feat_map, kernel_size=(1, 3), padding='same')
        self.lstm = nn.LSTM(num_feat_map * dim, num_hidden_lstm, batch_first=True)
        self.fc = nn.Linear(num_hidden_lstm, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = rearrange(x, "b channel var seq -> b seq (channel var)")
        # LSTM returns output and hidden state, we use the hidden state
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Get the last hidden state of the last LSTM layer
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class BaselineSQA(nn.Module):
    def __init__(self, embedding_matrix, num_words, embedding_dim, num_hidden_lstm, output_dim, dropout_rate, sen_dim, sen_channel, num_feat_map, num_classes, model_type='cnn_lstm_mul'):
        super(BaselineSQA, self).__init__()
        self.model_type = model_type
        if model_type == 'lstm':
            self.lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, num_hidden_lstm, output_dim, dropout_rate)
        elif model_type == 'cnn':
            self.sen_model = SensoryModel(sen_dim, sen_channel, num_feat_map, num_hidden_lstm, output_dim, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)

        self.tanh = nn.Tanh()
        self.dense1 = nn.LazyLinear(output_dim)
        self.dense2 = nn.Linear(output_dim, num_classes)

        if model_type in ['cnn_lstm_mul', 'cnn_lstm_cat', 'deepsqa', 'deepsqa2']:
            self.lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, num_hidden_lstm, output_dim, dropout_rate)
            self.sen_model = SensoryModel(sen_dim, sen_channel, num_feat_map, num_hidden_lstm, output_dim, dropout_rate)
            
            self.relu = nn.ReLU()

    def forward(self, x_input, y_input):
        if self.model_type == "lstm":
            lstm_result = self.lstm_model(y_input)
        elif self.model_type == "cnn":
            sen_result = self.sen_model(x_input)
        else:
            sen_result = self.sen_model(x_input)
            lstm_result = self.lstm_model(y_input)

        if self.model_type == 'cnn_lstm_mul':
            merged = sen_result * lstm_result
            d1 = self.tanh(self.dense1(merged))
            d2 = self.dense2(d1)
            return d2

        elif self.model_type == 'cnn_lstm_cat':
            merged = torch.cat((sen_result, lstm_result), dim=1)
            d1 = self.tanh(self.dense1(merged))
            d2 = self.dense2(d1)
            return d2

        elif self.model_type == 'deepsqa':
            merged = torch.cat((sen_result, lstm_result), dim=1)
            d1 = self.relu(self.dense1(merged))
            dp1 = self.dropout(d1)
            d2 = self.dense2(dp1)
            return d2

        elif self.model_type == 'deepsqa2':
            merged_1 = sen_result * lstm_result
            merged = torch.cat((sen_result, lstm_result, merged_1), dim=1)
            d1 = self.relu(self.dense1(merged))
            dp1 = self.dropout(d1)
            d2 = self.dense2(dp1)
            return d2

        elif self.model_type == 'lstm':
            d1 = self.tanh(self.dense1(lstm_result))
            d2 = self.dense2(d1)
            return d2

        elif self.model_type == 'cnn':
            d1 = self.tanh(self.dense1(sen_result))
            d2 = self.dense2(d1)
            return d2

        else:
            raise ValueError("Wrong model type!")

# Stacked Attention Model
class ShowAskAttendAnswer(nn.Module):
    def __init__(self, vocab_size, num_glimpses=2, n=14):
        super(ShowAskAttendAnswer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(300, 1024, batch_first=True)
        self.conv1 = nn.Conv2d(2048 + 1024, 512, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, num_glimpses, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(n, n))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 + 2048, 1024)
        self.fc2 = nn.Linear(1024, 3000)

    def forward(self, image_input, question_input):
        question_embedding = self.embedding(question_input)
        question_embedding = self.tanh(question_embedding)
        question_embedding = self.dropout(question_embedding)
        question_lstm, _ = self.lstm(question_embedding)
        question_tile = question_lstm.unsqueeze(1).repeat(1, image_input.size(1), 1)
        question_tile = question_tile.view(question_tile.size(0), image_input.size(1), image_input.size(2), -1)
        concatenated_features1 = torch.cat((image_input, question_tile), dim=1)
        concatenated_features1 = self.dropout(concatenated_features1)
        attention_conv1 = self.conv1(concatenated_features1)
        attention_relu = self.relu(attention_conv1)
        attention_relu = self.dropout(attention_relu)
        attention_conv2 = self.conv2(attention_relu)
        attention_maps = self.softmax(attention_conv2)
        image_attention = self.glimpse(attention_maps, image_input)
        concatenated_features2 = torch.cat((image_attention, question_lstm), dim=1)
        concatenated_features2 = self.dropout(concatenated_features2)
        fc1 = self.fc1(concatenated_features2)
        fc1_relu = self.relu(fc1)
        fc1_relu = self.dropout(fc1_relu)
        fc2 = self.fc2(fc1_relu)
        return fc2

    def glimpse(self, attention_maps, image_features, num_glimpses=2, n=14):
        glimpse_list = []
        for i in range(num_glimpses):
            glimpse_map = attention_maps[:, i, :, :].unsqueeze(1)
            glimpse_tile = glimpse_map.repeat(1, image_features.size(1), 1, 1)
            weighted_features = image_features * glimpse_tile
            weighted_average = self.avg_pool(weighted_features)
            weighted_average = self.flatten(weighted_average)
            glimpse_list.append(weighted_average)
        return torch.cat(glimpse_list, dim=1)


class SANModel(nn.Module):
    def __init__(self, embedding_matrix, num_words, embedding_dim, seq_length, 
                 num_hidden_lstm, output_dim, dropout_rate, sen_dim, sen_win_len, 
                 sen_channel, num_feat_map, num_classes, num_glimpses=2, n=1):
        super(SANModel, self).__init__()
        self.lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, 
                                        num_hidden_lstm, output_dim, dropout_rate)
        self.sen_model = SensoryModel(sen_dim, sen_win_len, sen_channel, num_feat_map, 
                                      num_hidden_lstm, output_dim, dropout_rate)
        
        self.num_glimpses = num_glimpses
        self.n = n
        
        self.attention_conv1 = nn.Conv2d(2048 + 1024, 512, kernel_size=(1, 1))
        self.attention_conv2 = nn.Conv2d(512, num_glimpses, kernel_size=(1, 1))
        self.fc1 = nn.Linear(1024 + 2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_input, y_input):
        sen_result = self.sen_model(x_input)
        lstm_result = self.lstm_model(y_input)
        
        question_tile = lstm_result.unsqueeze(1).repeat(1, self.n*self.n, 1)
        question_tile = question_tile.view(question_tile.size(0), self.n, self.n, -1)
        
        concatenated_features1 = torch.cat((sen_result, question_tile), dim=1)
        concatenated_features1 = self.dropout(concatenated_features1)
        
        attention_relu = F.relu(self.attention_conv1(concatenated_features1))
        attention_relu = self.dropout(attention_relu)
        
        attention_maps = F.softmax(self.attention_conv2(attention_relu), dim=1)
        
        image_attention = self.glimpse(attention_maps, sen_result)
        
        concatenated_features2 = torch.cat((image_attention, lstm_result), dim=1)
        concatenated_features2 = self.dropout(concatenated_features2)
        
        fc1_relu = F.relu(self.fc1(concatenated_features2))
        fc1_relu = self.dropout(fc1_relu)
        
        fc2_softmax = F.softmax(self.fc2(fc1_relu), dim=1)
        
        return fc2_softmax

    def glimpse(self, attention_maps, image_features):
        glimpse_list = []
        for i in range(self.num_glimpses):
            glimpse_map = attention_maps[:, i, :, :].unsqueeze(1)
            glimpse_tile = glimpse_map.repeat(1, image_features.size(1), 1, 1)
            weighted_features = image_features * glimpse_tile
            weighted_average = F.avg_pool2d(weighted_features, kernel_size=(self.n, self.n))
            weighted_average = weighted_average.view(weighted_average.size(0), -1)
            glimpse_list.append(weighted_average)
        return torch.cat(glimpse_list, dim=1)