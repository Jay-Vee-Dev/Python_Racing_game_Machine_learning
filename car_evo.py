import pygame
import torch
import numpy as np
from ga.ga_utils import evaluate_population, evolve, set_flat_params
from tracks.track import Track
from entities.car import Car
import torch.nn as nn
import os

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        steer = self.tanh(out[0])
        throttle = self.sigmoid(out[1])
        return torch.tensor([steer, throttle])

# Ensure assets folder exists
os.makedirs("assets", exist_ok=True)

# Pygame setup
pygame.init()
SCREEN_W, SCREEN_H = 800, 600
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# Track
track = Track()

# GA parameters
POP_SIZE = 10
EVAL_STEPS = 1200

# Load previous best network if exists
best_path = "assets/best_car_net.pt"
best_model = Net()
if os.path.exists(best_path):
    best_model.load_state_dict(torch.load(best_path))
    best_flat = torch.cat([p.data.view(-1) for p in best_model.parameters()])
    population = [best_flat.clone() for _ in range(POP_SIZE)]
else:
    population = [torch.randn(sum(p.numel() for p in Net().parameters())) for _ in range(POP_SIZE)]

# Assign colors
colors = [Car.PREDEFINED_COLORS[i % len(Car.PREDEFINED_COLORS)] for i in range(POP_SIZE)]

gen = 0
while True:  # infinite learning
    gen += 1
    # Handle quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Evaluate
    fitnesses, cars = evaluate_population(population, track, Net, eval_steps=EVAL_STEPS, render=True, screen=screen, clock=clock)

    # Track fastest completion time
    completion_times = [car.time_alive for car in cars if car.finished]
    fastest_time = min(completion_times) if completion_times else None

    print(f"Gen {gen}  best {max(fitnesses):.2f}  avg {np.mean(fitnesses):.2f}  fastest_time {fastest_time}")

    # Save best network
    best_idx = int(np.argmax(fitnesses))
    best_flat = population[best_idx]
    best_model = Net()
    set_flat_params(best_model, best_flat)
    torch.save(best_model.state_dict(), best_path)

    # Evolve population
    population = evolve(population, fitnesses)
