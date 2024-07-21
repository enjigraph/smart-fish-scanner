import pygame
pygame.mixer.init()

def start():
    pygame.mixer.music.load('./voice/start.mp3')
    pygame.mixer.music.play()
    
def finish():
    pygame.mixer.music.load('./voice/finish.mp3')
    pygame.mixer.music.play()

def retry():
    pygame.mixer.music.load('./voice/retry.mp3')
    pygame.mixer.music.play()

def remove():
    pygame.mixer.music.load('./voice/remove.mp3')
    pygame.mixer.music.play()

    
