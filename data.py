import pygame
import os
import random
import cv2
from PIL import ImageColor

pygame.init()


def draw_big_shapes(shape, win, big_color):
    if shape == 'circle':
        cx = random.randint(320, 380)
        cy = random.randint(320, 380)
        r = random.randint(250, 310)
        pygame.draw.circle(win, big_color, (cx, cy), r, 0)
    elif shape == 'rectangle':
        x = random.randint(0, 200)
        y = random.randint(0, 200)
        big = max(x, y)
        small = min(x, y)
        ref = 500 - big
        if (big == x):
            l = random.randint(ref, ref + 100)
            w = 690 - small + 10
        else:
            w = random.randint(ref, ref + 100)
            l = 690 - small
        pygame.draw.rect(win, big_color, (x, y, l, w), 0)
    elif shape == 'ellipse':
        x = random.randint(0, 200)
        y = random.randint(0, 200)
        big = max(x, y)
        small = min(x, y)
        ref = 500 - big
        if (big == x):
            l = random.randint(ref, ref + 100)
            w = 690 - small + 10
        else:
            w = random.randint(ref, ref + 100)
            l = 690 - small
        pygame.draw.ellipse(win, big_color, (x, y, l, w), 0)
    else:  # draw square
        x = random.randint(0, 200)
        y = random.randint(0, 200)
        pygame.draw.rect(win, big_color, (x, y, 500, 500), 0)


def draw_small_shapes(shape, win, small_color):
    if shape == 'circle':
        cx = random.randint(320, 380)
        cy = random.randint(320, 380)
        r = random.randint(100, 150)
        pygame.draw.circle(win, small_color, (cx, cy), r, 0)
    elif shape == 'rectangle':
        x = random.randint(250, 300)
        y = random.randint(250, 300)
        l = random.randint(150, 200)
        w = random.randint(150, 200)
        pygame.draw.rect(win, small_color, (x, y, l, w), 0)
    elif shape == 'square':
        x = random.randint(250, 300)
        y = random.randint(250, 300)
        l = random.randint(150, 200)
        pygame.draw.rect(win, small_color, (x, y, l, l), 0)
    elif shape == 'ellipse':
        x = random.randint(250, 300)
        y = random.randint(250, 300)
        l = random.randint(150, 200)
        w = random.randint(150, 200)
        pygame.draw.ellipse(win, small_color, (x, y, l, w), 0)
    else:  # triangle
        x1 = random.randint(220, 480)
        y1 = random.randint(200, 220)
        x2 = random.randint(220, 300)
        y2 = random.randint(400, 480)
        x3 = random.randint(400, 480)
        y3 = random.randint(400, 480)
        pygame.draw.polygon(win, small_color, [(x1, y1), (x2, y2), (x3, y3)], 0)


def create_imgs(data_map, num_imgs, set_type):
    num_combos = num_imgs
    white = (255, 255, 255)
    big_colors = {}
    small_colors = {}
    print(data_map['big']['colors'])
    for color in data_map['big']['colors']:
        big_colors[color] = ImageColor.getrgb(color)
    for color in data_map['small']['colors']:
        small_colors[color] = ImageColor.getrgb(color)

    X = 700
    Y = 700
    print("values: ", small_colors.values())

    win = pygame.display.set_mode((X, Y))

    pygame.display.set_caption('Shapes')

    win.fill(white)
    shape_combos = []
    for i in range(int(num_combos)):
        for big_color in big_colors.values():
            # print("big: ", big_color)
            for big_shape in data_map['big']['shapes']:
                for small_color in [x for x in small_colors.values() if x != big_color]:
                    # print("small: ", small_color)
                    for small_shape in data_map['small']['shapes']:
                        shape_combos.append({'big': (big_shape, big_color),
                                             'small': (small_shape, small_color)})
    for i, img in enumerate(shape_combos):
        draw_big_shapes(img['big'][0], win, img['big'][1])
        draw_small_shapes(img['small'][0], win, img['small'][1])
        im = 'shape' + str(i)
        fname = './dataset/raw/' + im + '.png'
        pygame.image.save(win, fname)

        orig_img = cv2.imread(fname, cv2.IMREAD_COLOR)
        resize_img = 'newshape' + str(i)
        fname = './dataset/{}/{}.png'.format(set_type, resize_img)
        new_img = cv2.resize(orig_img, (28, 28))
        cv2.imwrite(fname, new_img)
        os.remove('./dataset/raw/' + im + '.png')

        win.fill(white)


if __name__ == "__main__":
    while True:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()

                quit()

            pygame.display.update()


# main()
