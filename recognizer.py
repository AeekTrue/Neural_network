import pygame
import numpy as np
import sys
import network
pygame.init()

# variables
WIDTH = pygame.display.Info().current_w
HEIGHT = pygame.display.Info().current_h
# WIDTH, HEIGHT=800,600
FPS = 60
corner_left_up = (100, 100)
BACKGROUND = (0, 0, 0)
FIELD_COLOR = (100, 100, 100)
ACTIVE_UNIT_COLOR = (12,140,253)
FIELD_W, FIELD_H = 28, 28
UNIT_SIZE = 30
x_area = (corner_left_up[0], corner_left_up[0] + FIELD_W * UNIT_SIZE)  # кликабельная область по x
y_area = (corner_left_up[1], corner_left_up[1] + FIELD_H * UNIT_SIZE)  # кликабельная область по y

# init/create objs
# sc = pygame.display.set_mode((WIDTH,HEIGHT))
sc = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
clock = pygame.time.Clock()
f1 = pygame.font.Font(None, 36)


def draw_field():
    sc.fill(BACKGROUND)
    cur_y = corner_left_up[1]
    for y in range(FIELD_H):
        cur_x = corner_left_up[0]
        for x in range(FIELD_W):
            rectangle = pygame.Rect(cur_x, cur_y, UNIT_SIZE, UNIT_SIZE)
            pygame.draw.rect(sc, FIELD_COLOR, rectangle, 1)
            cur_x += UNIT_SIZE
        cur_y += UNIT_SIZE
    pygame.display.update()


def get_unit_coordinates(x, y):
    unit_x = (x - corner_left_up[0]) // UNIT_SIZE
    unit_y = (y - corner_left_up[1]) // UNIT_SIZE
    return unit_x, unit_y


def activate_unit(u_x, u_y, matrix):#зарисовывает клеточку и добавляет её в матрицу
    new_matrix = matrix
    new_matrix[u_x, u_y] = 0.75
    rectangle = pygame.Rect(corner_left_up[0] + u_x * UNIT_SIZE,
                            corner_left_up[1] + u_y * UNIT_SIZE,
                            UNIT_SIZE, UNIT_SIZE)
    pygame.draw.rect(sc, ACTIVE_UNIT_COLOR, rectangle)
    return new_matrix


def position_validator(x, y):
    """
    :return: 1 если мышь в области рисования, иначе 0
    """
    x_valid = x >= x_area[0] and x <x_area[1]
    y_valid = y >= y_area[0] and y <y_area[1]
    return  x_valid and y_valid


def get_picture():
    done = False
    picture = np.zeros((FIELD_W, FIELD_H))

    while not done:
        position_x, position_y = pygame.mouse.get_pos()
        button_l, button_c, button_r = pygame.mouse.get_pressed()
        if button_l and position_validator(position_x, position_y):
            unit_x, unit_y = get_unit_coordinates(position_x, position_y)
            picture = activate_unit(unit_x, unit_y, picture)
            pygame.display.update()
        for event in pygame.event.get():
            if event.type == 2 and event.key == 27:
                sys.exit()
            if event.type == 2 and event.key == 13:
                return  picture


# display objs
pygame.display.update()
struct = np.array([784, 16, 16, 10])
w, b = network.configurator.load_neural_network('F:\\Machine_learning_output\\90presents\\')
net = network.Network(struct, w, b)
answer = None
# main loop
while 1:

    draw_field()
    text_answer = f1.render(str(answer), 1, FIELD_COLOR)
    sc.blit(text_answer, (0, 0))
    pygame.display.update()

    picture = get_picture()
    input_activations = picture.ravel()
    output_activations = net.forward_feed(input_activations)
    answer = np.argmax(output_activations)
    for event in pygame.event.get():
        if event.type == 2 and event.key == 27:
            sys.exit()

    # edit objs and other
    # UPD display

    pygame.display.update()

    # pause

    clock.tick(FPS)
