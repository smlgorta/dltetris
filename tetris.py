#!/usr/bin/env python
#BLOCK means Tetrimino
#Field means Board

import sys, random
import pygame as pyg
import copy
from pygame.locals import *

FIELD_SIZE = (10, 10)
FIELD_BG_COLOR = (0, 0, 0)
SQ_SIZE = 50
SQ_BORDER1_COLOR = (204, 204, 204)
SQ_BORDER2_COLOR = (0, 0, 0)
BLOCK_COLORS = (
    (0, 255, 255),  # I
    (0, 0, 255),    # J
    (255, 165, 0),  # L
    (255, 255, 0),  # O
    (0, 255, 0),    # S
    (160, 32, 240), # T
    (255, 0, 0)     # Z
)
FALL_DELAY = 500

BLOCK_DEF = (
    (((0, 1), (1, 1), (2, 1), (3, 1)), 4), # I
    (((0, 0), (0, 1), (1, 1), (2, 1)), 3), # J
    (((0, 1), (1, 1), (2, 0), (2, 1)), 3), # L
    (((0, 0), (1, 0), (0, 1), (1, 1)), 2), # O
    (((0, 1), (1, 0), (1, 1), (2, 0)), 3), # S
    (((0, 1), (1, 0), (1, 1), (2, 1)), 3), # T
    (((0, 0), (1, 0), (1, 1), (2, 1)), 3)  # Z
)

SQUARES = []

class Block:
    def __init__(self, field):
        self.field = field

    def create(self):
        self.type = random.randint(0, 6)
        self.c = [[False] * 4 for i in range(4)]
        for i in BLOCK_DEF[self.type][0]: self.c[i[0]][i[1]] = True
        self.size = BLOCK_DEF[self.type][1]
        self.sq = SQUARES[self.type]
        self.x, self.y = int((FIELD_SIZE[0] - self.size) / 2), 0
        self.rot = 0

    def collides(self, dx, dy):
        for i in range(4):
            for j in range(4):
                a, b = self.y + dy + j, self.x + dx + i
                if self.c[i][j] and (a < 0 or b < 0 or a >= FIELD_SIZE[1] or b >= FIELD_SIZE[0] or self.field.f[int(a)][int(b)]):
                    return True
        return False

    def curr_width(self):
        maxint = 0
        for i in range(len(self.c[0])):
            int = 0
            for j in range(len(self.c)):
                if(self.c[j][i]):
                    int=j+1 - self.displacement()
            if(int > maxint):
                maxint = int
        return maxint

    #empty spaces to the left side
    def displacement(self):
        minint = 100
        for i in range(len(self.c[0])):
            int = 0
            for j in range(len(self.c)):
                if(self.c[j][i]):
                    break
                else:
                    int+=1
            if(int < minint):
                minint = int
        return minint

    def move(self, dx, dy):
        if self.collides(dx, dy):
            if dy >= 1:
                if self.y <= 0:
                    return -1
                for i in range(4):
                    for j in range(4):
                        if self.c[i][j]:
                            self.field.f[int(self.y + dy + j - 1)][int(self.x + i)] = self.sq
            return 0

        self.x += dx
        self.y += dy
        return 1

    def set_col(self, col):
        self.x = col
        return 1

    def set_row(self, row):
        self.y = row
        return 1

    def remove(self):
        for i in range(4):
            for j in range(4):
                if self.c[i][j]:
                    self.field.f[int(self.y + j)][int(self.x + i)] = None
        return 1

    def set_rot(self, rot):
        self.c = [[False] * 4 for i in range(4)]
        for i in BLOCK_DEF[self.type][0]: self.c[i[0]][i[1]] = True
        for i in range(rot):
            self.rotate(True)

    def rotate(self, dir):
        d = [[False] * 4 for i in range(4)]
        self.rot += 1
        for i in range(4):
            for j in range(4):
                if dir:
                    d[self.size - j - 1][i] = self.c[i][j]
                else:
                    d[j][self.size - i - 1] = self.c[i][j]
        for i in range(4):
            for j in range(4):
                a, b = self.y + j, self.x + i
                if d[i][j] and (a < 0 or b < 0 or a >= FIELD_SIZE[1] or b >= FIELD_SIZE[0] or self.field.f[int(a)][int(b)]):
                    return
        self.c = d

    def draw(self, screen):
        for i in range(4):
            for j in range(4):
                if self.c[i][j]:
                    screen.blit(self.sq, ((self.x + i) * SQ_SIZE, (self.y + j) * SQ_SIZE))

class Field:
    def __init__(self):
        self.f = [[None] * FIELD_SIZE[0] for i in range(FIELD_SIZE[1])]

    def check_filled(self):
        for i in range(FIELD_SIZE[1]):
            while all(self.f[i]):
                del self.f[i]
                self.f.insert(0, [None] * FIELD_SIZE[0])

    def draw(self, screen):
        for i in range(FIELD_SIZE[1]):
            for j in range(FIELD_SIZE[0]):
                if self.f[i][j]:
                    screen.blit(self.f[i][j], (j * SQ_SIZE, i * SQ_SIZE))

    def get_board_int(self):
        boardint = [[0 for x in range(FIELD_SIZE[0])] for y in range(FIELD_SIZE[1])]
        for i in range(FIELD_SIZE[1]):
            for j in range(FIELD_SIZE[0]):
                if self.f[i][j] is None:
                    boardint[FIELD_SIZE[1] - i - 1][j] = 0
                else:
                    boardint[FIELD_SIZE[1] - i - 1][j] = 1
                # print(boardint[j][i], end="")
            # print("")
        return boardint

    def clone(self):
        c = Field()
        for i in range(FIELD_SIZE[1]):
            for j in range(FIELD_SIZE[0]):
                c.f[i][j] = self.f[i][j]
        return c

    def get_possible_placements(self, b: Block):
        placements = []
        cols = []
        rots = []
        numplacements = 0
        original_field = b.field
        original_col = b.x
        original_rot = b.rot
        b.field = copy.deepcopy(b.field)
        for rot in range(4):
            # print("width", b.curr_width())
            for col in range(FIELD_SIZE[0]-b.curr_width()+1):
                # print("col", col)
                numplacements += 1
                b.set_col(col-b.displacement())
                while b.move(0, 1) == 1: pass
                placements.append(b.field.get_board_int())
                cols.append(col)
                rots.append(rot)
                b.remove()
                b.field = copy.deepcopy(b.field)
                b.set_row(0)
            b.set_col(int((FIELD_SIZE[0] - b.size) / 2))
            b.rotate(True)
        b.field = original_field
        b.x = original_col
        b.set_rot(original_rot)
        # print("numplacements:",numplacements)
        return placements, cols, rots;

def main():
    import tetris_comparison_agent
    ta = tetris_comparison_agent.TetrisAgent()

    pyg.init()
    pyg.mouse.set_visible(0)
    pyg.display.set_caption('asdftris')

    screen = pyg.display.set_mode((FIELD_SIZE[0] * SQ_SIZE, FIELD_SIZE[1] * SQ_SIZE))

    for color in BLOCK_COLORS:
        i = pyg.Surface((SQ_SIZE, SQ_SIZE))
        i.fill(color)
        pyg.draw.line(i, SQ_BORDER1_COLOR, (0, 0), (0, SQ_SIZE-1))
        pyg.draw.line(i, SQ_BORDER1_COLOR, (0, 0), (SQ_SIZE-1, 0))
        pyg.draw.line(i, SQ_BORDER2_COLOR, (0, SQ_SIZE-1), (SQ_SIZE-1, SQ_SIZE-1))
        pyg.draw.line(i, SQ_BORDER2_COLOR, (SQ_SIZE-1, 0), (SQ_SIZE-1, SQ_SIZE-1))
        SQUARES.append(i)

    field = Field()

    block = Block(field)
    block.create()

    last_fall, over = 0, False
    while True:
        placements, cols, rots = field.get_possible_placements(block)
        print(len(placements), " placements")
        picked = ta.pick(placements[:34])
        print(picked, " picked")
        block.set_rot(rots[picked])
        block.set_col(cols[picked])
        while block.move(0, 1) == 1: pass
        # for event in pyg.event.get():
        #     if event.type == QUIT: return
        #     elif event.type == KEYDOWN:
        #         k = event.key
        #         if k == K_ESCAPE: return
        #         elif k == K_UP: block.rotate(True)
        #         elif k == K_LEFT: block.move(-1, 0)
        #         elif k == K_RIGHT: block.move(1, 0)
        #         elif k == K_DOWN:
        #             while block.move(0, 1) == 1: pass
        # if over: continue
        time = pyg.time.get_ticks()
        if time - last_fall >= FALL_DELAY:
            n = block.move(0, 1)
            if n == -1:
                over = True
            elif n == 0:
                field.check_filled()
                block.create()

            last_fall = time

        screen.fill(FIELD_BG_COLOR)
        block.draw(screen)
        field.draw(screen)
        pyg.display.flip()
        pyg.time.wait(1000)


if __name__ == '__main__':
    main()