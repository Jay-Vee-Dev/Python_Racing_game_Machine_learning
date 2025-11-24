import torch
import numpy as np
from entities.car import Car
import pygame

def get_flat_params(model):
    return torch.cat([p.data.view(-1).clone() for p in model.parameters()])

def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def mutate_flat(flat, std=0.1):
    return flat + torch.randn_like(flat) * std

def crossover(a, b):
    flat = a.clone()
    cp = np.random.randint(0, len(a))
    flat[cp:] = b[cp:]
    return flat

def evolve(population, fitnesses):
    # Sort by fitness descending
    idx = np.argsort(fitnesses)[::-1]
    elites = [population[i] for i in idx[:max(1,int(0.2*len(population)))]]
    new_pop = [e.clone() for e in elites]

    while len(new_pop) < len(population):
        elite_indices = np.arange(len(elites))
        ia, ib = np.random.choice(elite_indices, size=2, replace=True)
        a = elites[ia]
        b = elites[ib]

        child = crossover(a, b)
        child = mutate_flat(child)
        new_pop.append(child)

    return new_pop

def evaluate_population(population, track, NetClass, eval_steps=1200, render=False, screen=None, clock=None):
    cars = [Car(track=track, color_idx=i) for i in range(len(population))]
    models = []
    for flat in population:
        m = NetClass()
        set_flat_params(m, flat)
        models.append(m)

    fitnesses = [0.0]*len(population)
    alive = [True]*len(population)
    centerline = np.array(track.centerline)

    for t in range(eval_steps):
        if render and screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill((30,30,30))
            screen.blit(track.surface, (0,0))

        for i, car in enumerate(cars):
            if not car.alive or car.finished:
                alive[i] = False
                continue

            # Observations
            obs = car.sensors(track.surface)
            angle_to_center = (track.height/2 - car.y) / (track.width/2 - car.x + 1e-6)
            obs_vec = torch.tensor([obs[0], obs[1], obs[2], car.speed/6.0, angle_to_center], dtype=torch.float32)
            with torch.no_grad():
                out = models[i](obs_vec).numpy()
            steer, throttle = float(out[0]), float(out[1])
            car.step(steer, throttle, track.surface, finish_line=track.finish_line)

            # Base progress fitness
            car_pos = np.array([car.x, car.y])
            dists = np.linalg.norm(centerline - car_pos, axis=1)
            progress = len(centerline) - np.min(dists)
            fitness = progress + car.speed*5.0

            # Huge bonus if finished
            if car.finished:
                fitness += 5000.0  # large bonus for completing the track
                # optionally reward faster completion
                fitness += max(0, (eval_steps - car.time_alive) * 2.0)

            fitnesses[i] = fitness

            if render and screen:
                car.draw(screen)

        if render and screen:
            pygame.display.flip()
            clock.tick(60)

        if not any(alive):
            break

    return fitnesses, cars
