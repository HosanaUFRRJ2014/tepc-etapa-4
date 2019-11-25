
############
# to compile
# $ pythran -Ofast -march=native tsp_compute_single_threaded.py

# pythran export search_for_best(int, float list list, int, float, int, float, float, bool)
# pythran export signature(int list)


import math
import sys
import random as rd
from constants import NUMBER_OF_PROCESSORS

MASTER = 0
INITIAL_DATA_TAG = 11
UPDATE_BESTIES = 72


def dist(cities, i, j):
    """
    """
    dx = cities[i][0] - cities[j][0]
    dy = cities[i][1] - cities[j][1]
    return math.sqrt(dx**2 + dy**2)


def dist_path(cities, path):
    """
    """
    N = len(path)
    li_dist = [dist(cities, path[k], path[k+1]) for k in range(N - 1)]
    li_dist += [dist(cities, path[0], path[N-1])]
    return sum(li_dist)


def search_for_best(seed,
                    cities,
                    nb_step,
                    beta_mult=1.005,
                    accept_nb_step=100,
                    p1=0.2,
                    p2=0.6,
                    check_signature=False):
    """
    exported
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rd.seed(seed)
    if rank == MASTER:
        nb_city = len(cities)

        # init
        path = [k for k in range(len(cities))]

        beta = 1.0
        n_accept = 0
        best_energy = float('inf')
        energy = dist_path(cities, path)

        compute_energy = 0

        data = {
            'nb_city': nb_city,
            'path': path,
            'beta': beta,
            'n_accept': n_accept,
            'best_energy': best_energy,
            'energy': energy,
            'compute_energy': compute_energy
        }
        if check_signature:
            store = set()
            data['store'] = store
        comm.send(data, dest=1, tag=INITIAL_DATA_TAG)
    else:
        data = comm.recv(source=MASTER, tag=INITIAL_DATA_TAG)
        nb_city = data['nb_city']
        path = data['path']
        beta = data['beta'] + 2
        n_accept = data['n_accept']
        best_energy = data['best_energy']
        energy = data['energy']
        compute_energy = data['compute_energy']
        store = data.get('store')

    for step in range(int(nb_step/NUMBER_OF_PROCESSORS)):
        if n_accept == accept_nb_step:
            beta *= beta_mult
            n_accept = 0

        p = rd.uniform(0.0, 1.0)
        if p < p1:
            # reverse section i-j
            i = rd.randint(1, nb_city-2)
            j = rd.randint(i+1, nb_city-1)
            new_path = path[:i]+[path[k] for k in range(j, i-1, -1)]+path[j+1:]

        elif p < p2:
            # move i to j
            new_path = path[:]
            i = rd.randint(1, nb_city - 1)
            b = new_path.pop(i)
            j = rd.randint(1, nb_city - 2)
            new_path.insert(j, b)

        else:
            # swap i and j
            new_path = path[:]
            i = rd.randint(1, nb_city - 1)
            j = rd.randint(1, nb_city - 1)
            new_path[i] = path[j]
            new_path[j] = path[i]

        process = True
        if check_signature:
            uuid = signature(new_path)
            if uuid in store:
                process = False

        if process:
            new_energy = dist_path(cities, new_path)
            compute_energy += 1

            if check_signature:
                store.add(uuid)

            if rd.uniform(0.0, 1.0) < math.exp(-beta * (new_energy - energy)):
                n_accept += 1
                energy = new_energy
                path = new_path[:]
                if energy < best_energy:
                    best_energy = energy
                    best_path = path[:]

    if rank != MASTER:
        besties = {
            'best_energy': best_energy,
            'best_path': best_path
        }
        comm.send(besties, dest=MASTER, tag=UPDATE_BESTIES)
        # print('saindo do rank=', rank)
        exit(0)

    # Compara o melhor valor encontrado pelo worker e o melhor valor encontrado
    # pelo master
    if rank == MASTER:
        besties_from_worker = comm.recv(source=1, tag=UPDATE_BESTIES)
        print('besties from worker: ',besties_from_worker)
        print('besties in master: ', best_energy, best_path)
        if besties_from_worker.get('best_energy') < best_energy:
            best_energy = besties_from_worker.get('best_energy')
            best_path = besties_from_worker.get('best_path')

    if check_signature:
        store_stats = {
            'nb_step': nb_step,
            'store_len': len(store),
            'ratio': len(store)/nb_step,
            'compute_energy': compute_energy,
        }
        return (best_energy, best_path, store_stats)
    else:
        return (best_energy, best_path, {})


def signature(path):
    """
    """
    i = path.index(0)
    cand_1 = path[i:]+path[:i]
    cand_2 = cand_1[0:1]+cand_1[1:][::-1]

    if cand_1[1] < cand_2[1]:
        cand = cand_1
    else:
        cand = cand_2

    return tuple(cand)
