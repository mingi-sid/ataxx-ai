import sys
import random
import time
start_time = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

cuda_ok = torch.cuda.is_available()
test = not __file__ in ['ai1.py', 'ai2.py'] and __name__ == "__main__"

class ResidualBlock(nn.Module):
    def __init__(self, size=7, channel=16):
        super(ResidualBlock, self).__init__()
        self.size = size
        self.channel = channel
        self.conv1a = nn.Conv2d(channel, channel//2, kernel_size=3, padding=1, bias=False)
        self.conv1b = nn.Conv2d(channel, channel//2, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2a = nn.Conv2d(channel, channel//2, kernel_size=3, padding=1, bias=False)
        self.conv2b = nn.Conv2d(channel, channel//2, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, input):
        xa = self.conv1a(input)
        xb = self.conv1b(input)
        x = torch.cat((xa, xb), dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        xa = self.conv2a(x)
        xb = self.conv2b(x)
        x = torch.cat((xa, xb), dim=1)
        x = self.bn2(x)
        x += input
        x = self.relu(x)
        return x

class SidusAtaxxNet(nn.Module):
    def __init__(self, size=7):
        self.size = size
        
        super(SidusAtaxxNet, self).__init__()
        self.conv1a = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.conv1b = nn.Conv2d(3, 4, kernel_size=5, padding=2, bias=False)
        self.conv1c = nn.Conv2d(3, 4, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(16)

        self.res_layers = nn.Sequential(*[ResidualBlock() for i in range(6)])

        self.conv_val = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val1 = nn.Linear(size*size, size*size)
        self.fc_val2 = nn.Linear(size*size, 1)

        self.conv_pol = nn.Conv2d(16, 3, kernel_size=1, bias=False)
        self.bn_pol = nn.BatchNorm2d(3)
        self.fc_pol = nn.Linear(3*size*size, 17*size*size)
        
    def forward(self, batch):
        x = batch.view(-1, 3, self.size, self.size)

        xa = self.relu(self.conv1a(x))
        xb = self.relu(self.conv1b(x))
        xc = self.relu(self.conv1b(x))
        x = torch.cat((xa, xb, xc), dim=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_layers(x)

        x_val = self.conv_val(x)
        x_val = self.bn_val(x_val)
        x_val = self.relu(x_val)
        x_val = x_val.view(-1, 1*self.size*self.size)
        x_val = self.fc_val1(x_val)
        x_val = self.relu(x_val)
        x_val = self.fc_val2(x_val)

        x_pol = self.conv_pol(x)
        x_pol = self.bn_pol(x_pol)
        x_pol = self.relu(x_pol)
        x_pol = x_pol.view(-1, 3*self.size*self.size)
        x_pol = self.fc_pol(x_pol)
        
        return F.log_softmax(x_pol, dim=1), torch.tanh(x_val)

def list_to_tuple(l):
    return tuple([tuple([item for item in row]) for row in l])

def tuple_to_list(t):
    return [[item for item in list(row)] for row in list(t)]

class SidusPlayer():
    def __init__(self, player_num, filepath='model.pt', use_gpu=False):
        self.player_num = player_num
        self.net = SidusAtaxxNet()
        if use_gpu:
            self.net.load_state_dict(torch.load(filepath))
        else:
            self.net.load_state_dict(torch.load(filepath, map_location='cpu'))
        self.gpu = use_gpu
        if self.gpu:
            self.net.cuda()
        self.net.eval()
        self.move_dict = {}
        self.state_dict = {}

    def move(self, board_list, max_visit=100, is_train=False, c_exploration=1.0):
        root = list_to_tuple(board_list)

        # Get policy, value
        tensor_input = list_to_tensor_input(board_list, self.player_num)
        if self.gpu:
            tensor_input = tensor_input.to('cuda')
        with torch.no_grad():
            policy, value = self.net(tensor_input)
        if self.gpu:
            policy = policy.cpu()
            value = value.cpu()
        prob = torch.exp(policy).view(-1)
        self.state_dict[(root, self.player_num)] = {'prob': prob, 'value': value}
        
        history = []
        current_state = (root, self.player_num)  # tuple(Board_tuple, player_to_move)
        current_player = self.player_num
        if max_visit == None:
            max_visit = 999999
        for i in range(max_visit):
            # Start from root
            history = []
            current_state = (root, self.player_num)
            current_player = self.player_num
            # Iterate until find leaf node
            while True:
                if test:
                    print_board(current_state[0])
                N_sum = 0
                if not current_state in self.move_dict:
                    # Add current state and possible moves to tree
                    self.move_dict[current_state] = {}
                    possible_moves = reversed(get_possible_moves(current_state[0], current_player))
                    if test:
                        #print(possible_moves)
                        pass
                    for move in possible_moves:
                        fr, to = move
                        move_key = get_move_key(fr[0], fr[1], to[0], to[1])
                        self.move_dict[current_state][move] = {'to': None, 'N': 0, 'Q': 0.0,
                                                               'P': prob[move_key], 'W': 0.0}
                current_move_dict = self.move_dict[current_state]
                for move in current_move_dict.keys():
                    N_sum += current_move_dict[move]['N']
                # Choose move with maximum U
                max_U = -10000000
                max_U_move = None
                for move in current_move_dict.keys():
                    Q = current_move_dict[move]['Q']
                    P = current_move_dict[move]['P']
                    N = current_move_dict[move]['N']
                    U = Q + c_exploration * P * sqrt(N_sum) / (1 + N)
                    if max_U < U:
                        max_U = U
                        max_U_move = move
                if max_U_move == None:  # No possible move
                    if is_end(current_state[0]):   # case: Game end
                        if get_winner(current_state[0], current_player, 3-current_player) == current_player:
                            v = 1
                        else:
                            v = -1
                        for history_state, history_move in reversed(history):
                            v = -v
                            self.move_dict[history_state][history_move]['N'] += 1
                            self.move_dict[history_state][history_move]['W'] += v
                            N = self.move_dict[history_state][history_move]['N']
                            W = self.move_dict[history_state][history_move]['W']
                            self.move_dict[history_state][history_move]['Q'] = W / N
                        if test:
                            print(i, sum([sum(row) for row in current_state[0]]))
                        break
                    else:   # case: Must pass
                        max_U_move = tuple()
                        self.move_dict[current_state][max_U_move] = {'to': None, 'N': 0, 'Q': 0.0,
                                                               'P': 1, 'W': 0.0}
                # Move
                if test:
                    print(max_U_move)
                history.append((current_state, max_U_move))
                if self.move_dict[current_state][max_U_move]['to'] != None:
                    next_state, next_player = self.move_dict[current_state][max_U_move]['to']
                else:
                    next_player = 3 - current_player
                    next_state = (update_board(current_state[0], max_U_move), next_player)
                    self.move_dict[current_state][max_U_move]['to'] = (next_state, next_player)
                current_state = next_state
                current_player = next_player
                # Get policy, value
                it_was_new = False
                if current_state in self.state_dict:
                    prob = self.state_dict[current_state]['prob']
                    value = self.state_dict[current_state]['value']
                else:
                    tensor_input = list_to_tensor_input(current_state[0], current_player)
                    if self.gpu:
                        tensor_input = tensor_input.to('cuda')
                    with torch.no_grad():
                        policy, value = self.net(tensor_input)
                    if self.gpu:
                        policy = policy.cpu()
                        value = value.cpu()
                    prob = torch.exp(policy).view(-1)
                    self.state_dict[current_state] = {'prob': prob, 'value': value}
                    it_was_new = True
                # Update parents
                for history_state, history_move in reversed(history):
                    value = -value
                    self.move_dict[history_state][history_move]['N'] += 1
                    self.move_dict[history_state][history_move]['W'] += value
                    N = self.move_dict[history_state][history_move]['N']
                    W = self.move_dict[history_state][history_move]['W']
                    self.move_dict[history_state][history_move]['Q'] = W / N
                if it_was_new:
                    break
            if max_visit == 999999:
                if time.time() - start_time > 8:
                    break
        # End of for i in range(max_visit)
        if not is_train:
            max_N = -1
            max_N_move = None
            move_from_root = self.move_dict[(root, self.player_num)]
            for move in move_from_root.keys():
                N = move_from_root[move]['N']
                if test:
                    print(move, N)
                if max_N < N:
                    max_N = N
                    max_N_move = move
            return max_N_move
        else:
            list_N = []
            move_from_root = self.move_dict[(root, self.player_num)]
            moves = list(move_from_root)
            for move in moves:
                N = move_from_root[move]['N']
                list_N.append(N)
            del self.state_dict[(root, self.player_num)]
            del self.move_dict[(root, self.player_num)]
            random_N_move = random.choices(moves, weights=list_N)[0]
            return random_N_move
# End of class SidusPlayer()

def print_board(board):
    result = "\n"
    for row in list(board):
        for item in list(row):
            result += str(item)
        result += '\n'
    print(result[:-1])

def get_possible_moves(board, player):
    split_actions = {}
    jump_actions = {}
    for row in range(7):
        for col in range(7):
            if board[row][col] == player:
                for dx, dy in [(-2,2),  (-1,2),  (0,2),  (1,2),  (2,2),
                               (-2,1),                           (2,1),
                               (-2,0),                           (2,0),
                               (-2,-1),                          (2,-1),
                               (-2,-2), (-1,-2), (0,-2), (1,-2), (2,-2)]:
                    if row + dx < 0 or row + dx >= 7 or col + dy < 0 or col + dy >= 7:
                        continue
                    if board[row + dx][col + dy] == 0:
                        if (row + dx, col + dy) in jump_actions:
                            jump_actions[(row + dx, col + dy)].append((row, col))
                        else:
                            jump_actions[(row + dx, col + dy)] = [(row, col)]
    for row in range(7):
        for col in range(7):
            if board[row][col] == player:
                for dx, dy in [(-1,1),  (0,1),  (1,1),
                               (-1,0),          (1,0),
                               (-1,-1), (0,-1), (1,-1)]:
                    if row + dx < 0 or row + dx >= 7 or col + dy < 0 or col + dy >= 7:
                        continue
                    if board[row + dx][col + dy] == 0:
                        if (row + dx, col + dy) in jump_actions:
                            del jump_actions[(row + dx, col + dy)]
                        split_actions[(row + dx, col + dy)] = [(row, col)]
    actions = []
    for to in jump_actions.keys():
        for fr in jump_actions[to]:
            actions.append((fr, to))
    for to in split_actions.keys():
        for fr in split_actions[to]:
            actions.append((fr, to))
    return actions
# End of def get_possible_moves()

_direction_map = {(0,2): 1,
                 (1,2): 2,
                 (2,2): 3,
                 (2,1): 4,
                 (2,0): 5,
                 (2,-1): 6,
                 (2,-2): 7,
                 (1,-2): 8,
                 (0,-2): 9,
                 (-1,-2): 10,
                 (-2,-2): 11,
                 (-2,-1): 12,
                 (-2,0): 13,
                 (-2,1): 14,
                 (-2,2): 15,
                 (-1,2): 16}
def get_move_direction(fr_x, fr_y, to_x, to_y):
    return _direction_map[(to_x-fr_x, to_y-fr_y)]
def get_move_key(fr_x, fr_y, to_x, to_y):
    is_jump = abs(fr_x - to_x) == 2 or abs(fr_y - to_y) == 2
    if is_jump:
        move_direction = get_move_direction(fr_x, fr_y, to_x, to_y)
    else:
        move_direction = 0
    key = move_direction * 7 * 7 + to_x * 7 + to_y
    return key

def is_end(board):
    found = {}
    for row in list(board):
        for cell in list(row):
            if not cell in found:
                found[cell] = True
    return len(found.keys()) <= 2

def get_winner(board, player, opponent):
    point_player = 0
    point_opponent = 0
    for row in list(board):
        for cell in list(row):
            if cell == player:
                point_player += 1
            elif cell == opponent:
                point_opponent += 1
    movable_player = get_possible_moves(board, player) != []
    movable_opponent = get_possible_moves(board, opponent) != []
    if movable_player and movable_opponent:
        pass
    elif movable_player:
        point_player = 7*7 - point_opponent
    elif movable_opponent:
        point_opponent = 7*7 - point_player
    if point_player > point_opponent:
        return player
    else:
        return opponent

def update_board(board_tuple, move):
    if move == tuple():
        return board_tuple
    fr, to = move
    fr_x, fr_y = fr
    to_x, to_y = to
    is_jump = abs(fr_x - to_x) == 2 or abs(fr_y - to_y) == 2
    player = board_tuple[fr_x][fr_y]
    board_list = tuple_to_list(board_tuple)
    if is_jump:
        board_list[fr_x][fr_y] = 0
        board_list[to_x][to_y] = player
    else:
        board_list[to_x][to_y] = player
    for dx, dy in [(-1,1),  (0,1),  (1,1),
                   (-1,0),          (1,0),
                   (-1,-1), (0,-1), (1,-1)]:
        if to_x + dx < 0 or to_x + dx >= 7 or to_y + dy < 0 or to_y + dy >= 7:
            continue
        if not board_list[to_x + dx][to_y + dy] in [0, player]:
            board_list[to_x + dx][to_y + dy] = player
    return list_to_tuple(board_list)


def list_to_tensor_input(board_list, current_player):
    map_dict = {current_player: 1, 3-current_player: -1, 0: 0}
    board_list = [[map_dict[item] for item in list(row)] for row in list(board_list)]
    board_tensor = torch.tensor(board_list, dtype=torch.float)
    board_player = F.relu(board_tensor)
    board_opponent = F.relu(-board_tensor)
    countdown = 0.5
    turn_tensor = torch.full(board_tensor.size(), countdown, dtype=torch.float)
    input_stack = (board_player, board_opponent, turn_tensor)
    return torch.stack(input_stack, dim=0)

if __name__ == "__main__" and not test:
    logfile_name = __file__[:-3] + 'log.txt'
    #f = open(logfile_name, 'a')

    input_str = sys.stdin.read()
    # 입력 예시
    # READY 1234567890.1234567 (입력시간)
    # "OK" 를 출력하세요.
    if input_str.startswith("READY"):
        #f.write("OK\n")
        #f.close()
        # 출력
        sys.stdout.write("OK")

    # 입력 예시
    # PLAY
    # 2 0 0 0 0 0 1
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 1 0 0 0 0 0 2
    # 1234567890.1234567 (입력시간)

    # AI의 액션을 출력하세요.
    # 출력 예시 : "0 0 2 2"
    elif input_str.startswith("PLAY"):
        player = __file__[-4]
        board = []
        actions = {} # { key: piece(start position), value: list of position(destination position) }

        # make board
        input_lines = input_str.split("\n")
        for i in range(7):
            board.append([int(x) for x in input_lines[i+1].split(" ")])
        for row in board:
            pass
            #f.write(' '.join([str(x) for x in row]))
            #f.write('\n')

        AI = SidusPlayer(player_num=int(player), use_gpu=cuda_ok)

        piece, position = AI.move(board, max_visit=None)
        
        #f.write(f"{piece[0]} {piece[1]} {position[0]} {position[1]}" + '\n')

        #f.close()
        # 출력
        sys.stdout.write(f"{piece[0]} {piece[1]} {position[0]} {position[1]}")
if __name__ == "__main__" and test:
    player = '2'
    board = []
    input_str = """PLAY
2 0 0 0 0 0 1
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
1 0 0 0 0 0 2
1234567890.1234567"""
    input_lines = input_str.split("\n")
    for i in range(7):
        board.append([int(x) for x in input_lines[i+1].split(" ")])

    AI = SidusPlayer(player_num=int(player))

    piece, position = AI.move(board, max_visit=100)
    print(piece, position)
