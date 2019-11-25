from tsp_wrapper import TSP


def main():
    nb_city = 100
    nb_run = int(8)
    nb_step = int(1e6)

    beta_mult = 1.02
    accept_nb_step = 100
    p1 = 0.2
    p2 = 0.8

    tsp = TSP(
        nb_city, nb_run, nb_step, beta_mult, accept_nb_step, p1, p2
    )
    tsp.generate_cities(seed=54321)
    # tsp.show_cities()
    solution = tsp.search_single_thread(
        'search',
        dated=False
    )
    print(solution)
    # tsp.show_results(nb_best=3, save=True)


if __name__ == "__main__":
    main()
