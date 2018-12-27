import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import random
import math
import os
import numpy as np
from performancePlot import computeDistance

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        emb_dim = input_dim

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional= True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # embedded = [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim= emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional= True)

        self.out = nn.Linear(2*hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
       
        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.1):
        # src = [sent len, batch size]
        # trg = [sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the zero tokens
        input = torch.zeros(batch_size,dtype=torch.long).to(self.device)

        for t in range(0, max_len):

            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]

            input = (trg[t] if teacher_force else top1)

        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    total = 0
    correct = 0
    predicted_dis = 0
    optimal_dis = 0
    for i, (src, trg) in enumerate(iterator):
        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [sent len, batch size]
        # output = [sent len, batch size, output dim]

        # reshape to:
        # trg = [(sent len - 1) * batch size]
        # output = [(sent len - 1) * batch size, output dim]
        loss = criterion(output.view([-1,output.shape[2]]), trg.reshape(num_robots*BATCH_SIZE))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total += trg.size(1)

        _, predicted = torch.max(output.data, 2)

        for vec in range(trg.size(1)):
            correct += (predicted[:,vec] == (trg[:,vec])).all().item()
        predicted_dis += computeDistance(torch.transpose(src, 0, 1), torch.transpose(predicted, 0, 1))
        optimal_dis += computeDistance(torch.transpose(src, 0, 1), torch.transpose(trg, 0, 1))
        if i+1%10 == 0:
            print(predicted_dis, optimal_dis)

        epoch_loss += loss.item()
    print(correct, total, len(iterator))
    return epoch_loss / len(iterator), correct / total, \
           predicted_dis/total, optimal_dis/total


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    total = 0
    correct = 0
    predicted_dis = 0
    optimal_dis = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)

            output = model(src, trg, 0)  # turn off teacher forcing

            loss = criterion(output.view([-1, output.shape[2]]), trg.reshape(num_robots * BATCH_SIZE))

            epoch_loss += loss.item()
            total += trg.size(1)

            _, predicted = torch.max(output.data, 2)

            for vec in range(trg.size(1)):
                correct += (predicted[:, vec] == (trg[:, vec])).all().item()
            predicted_dis += computeDistance(torch.transpose(src, 0, 1), torch.transpose(predicted, 0, 1))
            optimal_dis += computeDistance(torch.transpose(src, 0, 1), torch.transpose(trg, 0, 1))
            if i+1%10 == 0:
                print(predicted_dis, optimal_dis)
    return epoch_loss / len(iterator), correct/total, \
           predicted_dis/total, optimal_dis/total


def loading_data(num_robots):
    """
    (1): Load data from distanceMatrices.csv and assignmentMatrices.csv
    (2): Split data with the reference of number of robots
    :return: groups of training data and test data
    """
    import pandas

    print("Obtain training data")
    #distanceMatrices = np.loadtxt('distanceMatrices.csv', dtype=float)
    #assignmentMatrices = np.loadtxt('assignmentMatrices.csv', dtype=int)
    distanceMatrices1 = pandas.read_csv('../../16x16_SeqData/distanceMatrices_train_500w.csv',
                                       header=None,
                                       nrows=2000,
                                       sep=' ',
                                       dtype='float')
    distanceMatrices1 = distanceMatrices1.values

    distanceMatrices2 = pandas.read_csv('../../16x16_SeqData/distanceMatrices_train_300w.csv',
                                       header=None,
                                       nrows=1000,
                                       sep=' ',
                                       dtype='float')
    distanceMatrices2 = distanceMatrices2.values
    distanceMatrices = np.concatenate((distanceMatrices1,distanceMatrices2))
    assignmentMatrices1 = pandas.read_csv('../../16x16_SeqData/assignmentMatrices_train_500w.csv',
                                       header=None,
                                       nrows=2000,
                                       sep=' ',
                                       dtype='float')
    assignmentMatrices1 = assignmentMatrices1.values
    assignmentMatrices2 = pandas.read_csv('../../16x16_SeqData/assignmentMatrices_train_300w.csv',
                                       header=None,
                                       nrows=1000,
                                       sep=' ',
                                       dtype='float')
    assignmentMatrices2 = assignmentMatrices2.values
    assignmentMatrices  = np.concatenate((assignmentMatrices1,assignmentMatrices2))
    size0,size1  = distanceMatrices.shape()
    size2,size3  = assignmentMatrices.shape()
    print(str(size0)+" " + str(size1)+"  "+str(size2)+" "+str(size3))
    # y_train = to_categorical(y_train)
    N, M = assignmentMatrices.shape
    assert num_robots == M
    assignmentMatrices = assignmentMatrices.reshape(N, num_robots)

    # Create a MxNxM matrices,within which matrices[i,:,:] is the ground truth for model i
    N, M = distanceMatrices.shape
    distanceMatrices = distanceMatrices.reshape(N, num_robots, num_robots)

    NTrain = int(0.9*N)
    X_train = distanceMatrices[:NTrain, ] # the training inputs we will always use
    X_test = distanceMatrices[NTrain:, ] # for testing
    y_train = assignmentMatrices[:NTrain,:]
    y_test = assignmentMatrices[NTrain:,:]
    print("Obtain training data: robots: {}, samples: {}".format(num_robots, N))

    return torch.tensor(X_train,device= device).float(), torch.tensor(y_train,device= device).long(), \
           torch.tensor(X_test,device= device).float(), torch.tensor(y_test,device= device).long()

"""
Initialize model
"""
num_robots = 16
BATCH_SIZE = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is {0}'.format(device))

X_train, y_train, X_test, y_test = loading_data(num_robots = num_robots)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_iterator = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_iterator = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)


INPUT_DIM = num_robots
OUTPUT_DIM = num_robots
ENC_EMB_DIM = num_robots
DEC_EMB_DIM = num_robots
HID_DIM = 1024
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

training = True
"""
Train model
"""
if training:
    N_EPOCHS = 50
    CLIP = 10
    SAVE_DIR = 'models/bidirectional_16x16'
    res_train = []
    optimal_train = []
    train_acc_list = []

    if not os.path.isdir('{}'.format(SAVE_DIR)):
        os.makedirs('{}'.format(SAVE_DIR))

    for epoch in range(N_EPOCHS):

        train_loss, acc, avg_pred_dis, avg_optimal_dis = train(model, train_iterator, optimizer, criterion, CLIP)
        optimal_train.append(avg_optimal_dis)
        res_train.append(avg_pred_dis)
        train_acc_list.append(acc)	

        if (epoch+1) % 1 == 0:
            MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut1_model'+str(epoch+1)+'.pt')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(
            '| Epoch: {} | Train Loss: {} | Train PPL: {} | Train Accuracy: {}'.format(epoch+1, train_loss, math.exp(train_loss), acc))
    np.savetxt('./csv_16x16/train_distance.csv',res_train,delimiter=',',fmt='%f')
    np.savetxt('./csv_16x16/train_optimal_distance.csv',optimal_train,delimiter=',',fmt='%f')
    np.savetxt('./csv_16x16/train_acc.csv',train_acc_list,delimiter=',',fmt='%f')
else:
    """
    Test model
    """
    N_EPOCHS = 50
    res_train = []
    res = []
    optimal_train = []
    optimal = []
    test_acc_list = []

    for epoch in range(0, N_EPOCHS):
        SAVE_DIR = 'models/bidirectional_16x16'
        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut1_model' + str(epoch + 1) + '.pt')
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        #train_loss, train_acc, avg_tr_pred_dis, avg_tr_optimal_dis = evaluate(model, train_iterator, criterion)
        test_loss, test_acc, avg_pred_dis, avg_optimal_dis = evaluate(model, test_iterator, criterion)

        optimal_train.append(avg_optimal_dis)
        res_train.append(avg_pred_dis)
        #optimal.append(avg_tr_optimal_dis)
        #res.append(avg_tr_pred_dis)

        test_acc_list.append(test_acc)

        print('EPOCH: {} | Test acc: {} '.format(epoch+1,test_acc))#, train_acc))
    np.savetxt('./csv_16x16/train_distance.csv',res_train,delimiter=',',fmt='%f')
    np.savetxt('./csv_16x16/optimval_distance.csv',optimal_train,delimiter=',',fmt='%f')
    np.savetxt('./csv_16x16/test_acc.csv',test_acc_list,delimiter=',',fmt='%f')	
    #plotDistance(iterations=np.linspace(1, N_EPOCHS, N_EPOCHS), optimalDistance=np.asarray(optimal_train),
    #            totalDistances=np.asarray(res_train))
    #from matplotlib import pyplot as plt
    #plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS),test_acc_list)
    #plt.xlabel("test accuracy")
    #plt.show()
    #plotDistance(iterations=np.linspace(1,N_EPOCHS,N_EPOCHS), optimalDistance= np.asarray(optimal),
    #             totalDistances= np.asarray(res))

