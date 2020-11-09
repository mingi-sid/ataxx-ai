# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# %%
from ai import *
import ataxx
import ataxx.players

# %%
from torch.utils.data import Dataset, DataLoader
import pickle
import random
class GameDataset(Dataset):
    def __init__(self, record, filepath='game_record.pkl'):
        self.record = record
        try:
            with open(filepath, 'rb') as f:
                record_from_file = pickle.load(f)
                file_length = len(record_from_file['input'])
                use_num = min(file_length, 30000)
                sample_key = random.sample(range(file_length), use_num)
                record_sampled = {'input': [], 'value': [], 'policy': []}
                for key in sample_key:
                    record_sampled['input'].append(record_from_file['input'][key])
                    record_sampled['policy'].append(record_from_file['policy'][key])
                    record_sampled['value'].append(record_from_file['value'][key])
                self.record['input'] += record_sampled['input']
                self.record['policy'] += record_sampled['policy']
                self.record['value'] += record_sampled['value']
                assert(len(self.record['input']) == len(self.record['policy']))
                assert(len(self.record['input']) == len(self.record['value']))
                with open('game_record.pkl', 'wb') as f2:
                    pickle.dump(self.record, f2)
        except:
            with open('game_record.pkl', 'wb') as f2:
                pickle.dump(self.record, f2)

    def __len__(self):
        return len(self.record['input'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        return {'input': self.record['input'][idx],
                'value': self.record['value'][idx],
                'policy': self.record['policy'][idx]}

initial_state = [[1,0,0,0,0,0,2],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[2,0,0,0,0,0,1]]
def self_play(match_num=100, max_visit=100, c_exploration=1.0, minimax=False):
    game_record = {'input': [],
                'policy': [],
                'value': []}
    print('Matching')
    for i in range(match_num):
        AI1 = SidusPlayer(1, use_gpu=True)
        AI2 = SidusPlayer(2, use_gpu=True)
        if random.random() < 0.5:
            board = initial_state
        else:
            start_pos = [(random.randrange(7), random.randrange(7), 1) for j in range(2)]
            start_pos += [(random.randrange(7), random.randrange(7), 2) for j in range(2)]
            board = [[0 for x in range(7)] for y in range(7)]
            for x, y, p in start_pos:
                board[x][y] = p
        turn = 0
        max_turn = 128
        while True:
            move_1 = AI1.move(board, max_visit=max_visit, is_train=True, c_exploration=c_exploration)
            if move_1 != tuple():
                policy = get_move_key(move_1[0][0], move_1[0][1], move_1[1][0], move_1[1][1])
            else:
                policy = 24
            tensor_input_1 = list_to_tensor_input(board, 1)
            game_record['input'].append(tensor_input_1)
            game_record['policy'].append(torch.tensor(policy, dtype=torch.long))
            board = update_board(board, move_1)
            if is_end(board) or turn >= max_turn:
                if get_winner(board, 1, 2) == 1:
                    value_1st = 1
                else:
                    value_1st = -1
                v = value_1st
                while len(game_record['value']) < len(game_record['input']):
                    game_record['value'].append(torch.tensor([v], dtype=torch.float))
                    v = -v
                break
            turn += 1
            if minimax:
                move_2 = move_minimax(board, 2, 4)
            else:
                move_2 = AI2.move(board, max_visit=max_visit, is_train=True, c_exploration=c_exploration)
            if move_2 != tuple():
                policy = get_move_key(move_2[0][0], move_2[0][1], move_2[1][0], move_2[1][1])
            else:
                policy = 24
            tensor_input_2 = list_to_tensor_input(board, 2)
            game_record['input'].append(tensor_input_2)
            game_record['policy'].append(torch.tensor(policy, dtype=torch.long))
            board = update_board(board, move_2)
            if is_end(board) or turn >= max_turn:
                if get_winner(board, 1, 2) == 1:
                    value_1st = 1
                else:
                    value_1st = -1
                v = value_1st
                while len(game_record['value']) < len(game_record['input']):
                    game_record['value'].append(torch.tensor([v], dtype=torch.float))
                    v = -v
                break
            turn += 1
        print(i)
    # End of for
    return game_record
    
def move_minimax(board, player, depth=4):
    actions = {}
    for row in range(7):
        for col in range(7):
            if board[row][col] == int(player):
                moveable_positions = []
                for i in range(max(row-2, 0), min(row+3, 7)):
                    for j in range(max(col-2, 0), min(col+3, 7)):
                        if board[i][j] == 0:
                            moveable_positions.append((i, j))
                if moveable_positions:
                    actions[(row, col)] = moveable_positions
    if len(actions.keys()) == 0:
        return tuple()
    ataxx_board = ataxx.Board()
    for x in range(7):
        for y in range(7):
            ataxx_board.set(x, y, {2:ataxx.BLACK,1:ataxx.WHITE,0:ataxx.EMPTY}[board[x][y]])
    move = ataxx.players.alphabeta(ataxx_board, -999999, 999999, depth)
    if move == None:
        return tuple()
    if move.is_single():
        position = move.to_x, move.to_y
        piece = None
        for move_from in actions.keys():
            if position in actions[move_from]:
                piece = move_from
                break
    else:
        piece = move.fr_x, move.fr_y
        position = move.to_x, move.to_y
    return (piece, position)
    
def train(record, epoch_num=10):
    print('Training')
    datafeeder = GameDataset(record)
    randomsampler = torch.utils.data.RandomSampler(datafeeder, replacement=True, num_samples=20000)

    net = SidusAtaxxNet()
    net.load_state_dict(torch.load('model.pt'))
    net.train()
    net.cuda()

    criterion_pol = nn.NLLLoss()
    criterion_val = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.00001)

    for epoch in range(epoch_num):
        dataloader = DataLoader(datafeeder, batch_size=32, sampler=randomsampler)
        total_loss = 0
        loss_pol_sum = 0
        loss_val_sum = 0
        loss_val_max = -1
        loss_val_min = 1
        for i, data in enumerate(dataloader):
            #print(torch.sum(data['input'][0][0]))
            output = net(data['input'].to('cuda'))
            policy, value = output
            #print(value[0].detach().cpu().item(), data['value'][0])
            loss_pol = criterion_pol(policy, data['policy'].to('cuda'))
            loss_val = criterion_val(value, data['value'].to('cuda'))
            alpha = 1e0
            loss = 2 * (loss_pol + alpha * loss_val) / (1 + alpha)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_pol_sum += loss_pol.detach().cpu().item()
            loss_val_sum += loss_val.detach().cpu().item()
            total_loss += loss.detach().cpu().item()
            loss_val_float = loss_val.detach().cpu().item()
            if loss_val_float > loss_val_max:
                loss_val_max = loss_val_float
            if loss_val_float < loss_val_min:
                loss_val_min = loss_val_float
            print_interval = 100
            if i % print_interval == print_interval - 1:
                print('[%d, %5d] loss: %.8f\tpol: %.8f\tval: %.8f %.5f %.5f' %
                    (epoch + 1, i + 1, total_loss/print_interval, loss_pol_sum/print_interval, loss_val_sum/print_interval, loss_val_max, loss_val_min))
                total_loss = 0
                loss_pol_sum = 0
                loss_val_sum = 0
                loss_val_max = -1
                loss_val_min = 1
                torch.save({'epoch': epoch,
                            'step': i,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'loss': loss}, 'checkpoint_aginst.tar')
        torch.save(net.state_dict(), 'model.pt')

def compete(match_num, target_num):
    print('Competing')
    AI_champ = SidusPlayer(1, filepath='model_top.pt', use_gpu=True)
    AI_chall = SidusPlayer(2, filepath='model.pt', use_gpu=True)
    win_count = 0
    for i in range(match_num):
        turn = 0
        max_turn = 128
        board = initial_state
        while turn < max_turn:
            board_str_1 = [''.join([{0:'_',1:'O',2:'X'}[item] for item in row]) for row in board]
            move_1 = AI_champ.move(board, max_visit=500)
            board = update_board(board, move_1)
            if is_end(board) or turn >= max_turn:
                if i == 0:
                    for j in range(7):
                        print(board_str_1[j])
                    print()
                if get_winner(board, 1, 2) == 2:
                    win_count += 1
                    print('W')
                else:
                    print('_')
                break
                
            turn += 1
            board_str_2 = [''.join([{0:'_',1:'O',2:'X'}[item] for item in row]) for row in board]
            if i == 0:
                for j in range(7):
                    print(board_str_1[j] + '\t' + board_str_2[j])
                print()
            move_2 = AI_chall.move(board, max_visit=500)
            board = update_board(board, move_2)
            if is_end(board) or turn >= max_turn:
                if get_winner(board, 1, 2) == 2:
                    win_count += 1
                    print('W')
                else:
                    print('_')
                break
            turn += 1
        if match_num - (i+1) + win_count < target_num:
            break
        if win_count >= target_num:
            break
    if win_count >= target_num:
        torch.save(AI_chall.net.state_dict(), 'model_top.pt')
        return True
    return False


# %%
def main():
    count = 0

    while True:
        count += 1
        print('count', count)
        for i in range(2):
            game_record = self_play(20, max_visit=100, c_exploration=1.0, minimax=False)
            # Train
            train(game_record, 20)
        new_champ = compete(7, 5)
        if new_champ:
            print('selfplay {}: New champ'.format(count))
        else:
            print('selfplay {}: End'.format(count))


# %%
def print_dataset(record, idx):
    print('idx', idx)
    print(torch.sum(record['input'][idx][0] - record['input'][idx][1]))
    print(record['input'][idx][0] - record['input'][idx][1])
    pol = record['policy'][idx].item()
    print(pol // 7 // 7, (pol // 7) % 7, pol % 7)
    print(record['value'][idx])


# %%
empty_record = {'input': [],
                'policy': [],
                'value': []}


main()