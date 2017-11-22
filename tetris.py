import numpy as np
import copy
import random

I = list()
rot = np.bool_([[True, True, True, True],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]])
I.append(rot)
rot = np.bool_([[True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False]])
I.append(rot)

O = list()
rot = np.bool_([[True, True, False, False],
                [True, True, False, False],
                [False, False, False, False],
                [False, False, False, False]])
O.append(rot)

S = list()
rot = np.bool_([[False, True, True, False],
                [True, True, False, False],
                [False, False, False, False],
                [False, False, False, False]])
S.append(rot)
rot = np.bool_([[True, False, False, False],
                [True, True, False, False],
                [False, True, False, False],
                [False, False, False, False]])
S.append(rot)

Z = list()
rot = np.bool_([[True, True, False, False],
                [False, True, True, False],
                [False, False, False, False],
                [False, False, False, False]])
Z.append(rot)
rot = np.bool_([[False, True, False, False],
                [True, True, False, False],
                [True, False, False, False],
                [False, False, False, False]])
Z.append(rot)

L = list()
rot = np.bool_([[True, False, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [False, False, False, False]])
L.append(rot)
rot = np.bool_([[False, False, True, False],
                [True, True, True, False],
                [False, False, False, False],
                [False, False, False, False]])
L.append(rot)

J = list()
rot = np.bool_([[False, True, False, False],
                [False, True, False, False],
                [True, True, False, False],
                [False, False, False, False]])
J.append(rot)
rot = np.bool_([[True, True, True, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False]])
J.append(rot)

T = list()
rot = np.bool_([[True, True, True, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, False, False, False]])
T.append(rot)
rot = np.bool_([[True, False, False, False],
                [True, True, False, False],
                [True, False, False, False],
                [False, False, False, False]])

T.append(rot)
rot = np.bool_([[False, True, False, False],
                [True, True, True, False],
                [False, False, False, False],
                [False, False, False, False]])
T.append(rot)
rot = np.bool_([[False, True, False, False],
                [True, True, False, False],
                [False, True, False, False],
                [False, False, False, False]])
T.append(rot)

class Piece:

    switch = {
        0: I,
        1: O,
        2: S,
        3: Z,
        4: L,
        5: J,
        6: T
    }

    def __init__(self, piece_id = None):
        if(piece_id is None):
            piece_id = random.randint(0, 6)
        self.piece_id = piece_id
        self.identity = self.switch[self.piece_id]
        self.rot = 0
        self.rotations = len(self.identity)


    def width(self, rot):
        for c in reversed(range(4)):
            for r in range(4):
                if(self.identity[rot][r][c]):
                    return c + 1


    def height(self, rot):
        for r in reversed(range(4)):
            for c in range(4):
                if (self.identity[rot][r][c]):
                    return r + 1


    def rotate(self):
        self.rot  += 1
        if(self.rot >= self.rotations):
            self.rot = 0

    def set_rotate(self, rotate_to):
        self.rot  = rotate_to

    def rotated_profile(self):
        return self.identity[self.rot]

class Board:

    def __init__(self):
        self.w, self.h = 10, 10;
        self.board = np.bool_([[False for c in range(self.w)] for r in range(self.h)])
        self.lines_cleared = 0

    def possible_actions(self, p):
        actions = list()
        for r in range(p.rotations):
            for c in (range(self.w + 1 - p.width(r))):
                actions.append(np.int_([r, c]))
        return actions

    # action comes as array of [rotation, column].
    # returns False if game is over after this action.
    def perform_action(self, action, p):
        rot = action[0]
        landing_col = action[1]
        p.set_rotate(rot)
        landing_height = -3

        while(landing_height + p.height(rot) < self.h and self.piece_fits(p, landing_height, landing_col)):
            landing_height += 1

        if(not self.piece_fits(p, landing_height, landing_col)):
            landing_height -= 1

        self.make_permanent(p, landing_height, landing_col)
        self.lines_cleared += self.clear_lines()

        if(landing_height < 0):
            return False
        else:
            return True


    def piece_fits(self, p, landing_height, landing_col):
        for c in range(4):
            for r in range(4):
                if(landing_height + r >= 0 and r < p.height(p.rot) and c < p.width(p.rot)):
                    if(self.board[landing_height + r][landing_col + c] and p.rotated_profile()[r][c]):
                        return False

        return True

    def make_permanent(self, p, landing_height, landing_col):
        for c in range(4):
            for r in range(4):
                if (landing_height + r >= 0 and r < p.height(p.rot) and c < p.width(p.rot)):
                    self.board[landing_height + r][landing_col + c] = (p.rotated_profile()[r][c] or self.board[landing_height + r][landing_col + c])

    def clear_lines(self):
        lines_cleared = 0
        full = False
        for r in range(self.h):
            for c in range(self.w):
                if(not self.board[r][c]):
                    full = False
                    break
                else:
                    full = True
            if(full):
                lines_cleared += 1
                self.clear_line(r)
        return lines_cleared

    def clear_line(self, linetoclear):
        # print("clearing line ", linetoclear)
        for r in reversed(range(linetoclear)):
            for c in range(self.w):
                self.board[r + 1][c] = self.board[r][c]
        for c in range(self.w):
            self.board[0][c] = False

def discounted_sum(r, discount):
    sum = 0
    for i in range(0, len(r)):
        if(r[i] >= 1):
            sum = sum + pow(discount, len(r) - i - 1) * r[i]

    return sum

#input is array of actions
#output is array of resulting boards and actions without the ones that end on gameover
def resulting_boards(actions, b, p):
    resulting_boards = []
    filtered_actions = []
    for action in actions:
        clone_board = copy.deepcopy(b)
        game_continues = clone_board.perform_action(action, p)
        if game_continues:
            resulting_boards.append(np.flip(clone_board.board, 0))
            filtered_actions.append(action)
    return resulting_boards, filtered_actions




import tetris_comparison_agent
numgames = 100000

def values(alternatives, ta):
    values = []
    for i in range(len(alternatives)):
        values.append(ta.value(alternatives[i].reshape(-1)))
    return values

def playGames(numgames, model, sess):
    ta = tetris_comparison_agent.TetrisAgent(model, sess)
    random.seed(1)
    mean_score = 0
    for game in range(numgames):
        b = Board();
        p = Piece();
        alternatives = []
        picked_action = []
        cumulated_reward = []
        reward = []

        reward_baseline = 0
        steps = 0
        exp_length = 5
        prev_lines_cleared = 0
        discount = 0.9
        train = False

        game_continues = True
        while(game_continues):
            prev_lines_cleared = b.lines_cleared
            # print("piece id: ", p.piece_id)
            possible_actions = b.possible_actions(p)
            possible_placements, possible_actions_filtered = resulting_boards(possible_actions, b, p)
            if len(possible_placements) == 0:
                while (len(reward) > 0):
                    index = len(reward)-1
                    alt_reward = np.zeros(len(alternatives[index]))
                    # alt_reward = np.array(values(alternatives[index]))
                    alt_reward[picked_action[index]] = max(alt_reward[picked_action[index]], discounted_sum(reward, discount))
                    cumulated_reward = alt_reward - reward_baseline
                    ta.add_to_experience_pool(alternatives[index], cumulated_reward, (max(cumulated_reward) > 0))
                    alternatives.pop()
                    reward.pop()
                break
            action_index = ta.pick(np.int_(possible_placements))
            game_continues = b.perform_action(possible_actions_filtered[action_index], p)
            #print(possible_actions_filtered[action_index])
            # print(1*b.board)
            # print("r:",b.lines_cleared - prev_lines_cleared)


            p = Piece();
            steps += 1
            if (not game_continues):
                continue

            #managing memory:
            reward.insert(0, b.lines_cleared - prev_lines_cleared)

            if(len(reward) > exp_length):
                reward.pop()

            picked_action.insert(0, action_index)
            if(len(picked_action) > exp_length):
                picked_action.pop()

            alternatives.insert(0, np.int_(possible_placements))
            if (len(alternatives) > exp_length):
                alternatives.pop()

            if(steps >= exp_length):
                alt_reward = np.zeros(len(alternatives[exp_length - 1]))
                # alt_reward = np.array(values(alternatives[exp_length - 1]))
                # alt_reward[picked_action[exp_length - 1]] = max(alt_reward[picked_action[exp_length - 1]], discounted_sum(reward))
                alt_reward[picked_action[exp_length - 1]] = discounted_sum(reward, discount)
                cumulated_reward = alt_reward - reward_baseline

            # if(sum(cumulated_reward) > 0):
            #     print(cumulated_reward)
            #     print(picked_action)
            #     print(action_index)
            #     for alt in alternatives[exp_length - 1]:
            #          print(1*(alt))
            #          print()
            # if(sum(cumulated_reward) > 0):
            if(len(alternatives) >= exp_length and train):
                ta.add_to_experience_pool(alternatives[exp_length - 1], cumulated_reward, (max(cumulated_reward) > 0))
            # print("____________________")

        # print("cleared lines: ", b.lines_cleared)
        mean_score = mean_score + b.lines_cleared
        if(game % 20 == 0 and train):
            ta.train_from_pool()

    return mean_score / numgames
