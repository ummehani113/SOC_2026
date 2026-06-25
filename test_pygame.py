import pygame
import time

pygame.init()

screen = pygame.display.set_mode((600,400))
pygame.display.set_caption("Testing")

running = True

start = time.time()

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if time.time()-start > 5:
        running=False

pygame.quit()