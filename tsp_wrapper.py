import os
import json
import concurrent.futures
import datetime as dt
import pandas as pd
import random as rd
import matplotlib.pyplot as pl

from IPython.display import display
from timeit import default_timer as timer

# from .tsp_compute_multi_threaded_omp import search_for_best_omp
# from .tsp_compute_single_threaded import search_for_best
from tsp_compute_single_threaded import search_for_best


class TSP:
    """
    """

    def __init__(self, nb_city, nb_run, nb_step, beta_mult, accept_nb_step, p1, p2):
        """
        """
        assert isinstance(nb_city, int) and nb_city > 0, 'nb_city must be an int > 0'
        assert isinstance(nb_run, int) and nb_run > 0, 'nb_run must be an int > 0'
        assert isinstance(nb_step, int) and nb_step > 0, 'nb_step must be an int > 0'
        assert isinstance(beta_mult, float) and beta_mult > 1, 'nb_step must be a float > 1'
        assert isinstance(accept_nb_step, int) and accept_nb_step > 0, 'nb_step must be an int > 0'
        assert isinstance(p1, float) and 0 < p1 < 1, 'p1 must be a probability >0 and <1'
        assert isinstance(p2, float) and 0 < p2 < 1, 'p1 must be a probability >0 and <1'
        assert p1 < p2, 'p1 must be <p2'

        self.params = {
            'nb_city': nb_city,
            'nb_run': nb_run,
            'nb_step': nb_step,
            'beta_mult': beta_mult,
            'accept_nb_step': accept_nb_step,
            'p1': p1,
            'p2': p2,
        }
        self.df_params = pd.DataFrame(pd.Series(self.params), columns=['value'])
        self.df_params.index.name = 'param'

    def show_params(self):
        """
        """
        display(self.df_params)

    def generate_cities(self, seed=123456):
        """
        """
        self.seed = seed
        rd.seed(seed)
        self.cities = [[rd.uniform(0.0, 1.0), rd.uniform(0.0, 1.0)]
                       for i in range(self.params['nb_city'])]

    def task(self,
             seed,
             cities,
             nb_step,
             beta_mult,
             accept_nb_step,
             p1,
             p2,
             check_signature):
        """
        """
        sol = search_for_best(seed, cities, nb_step, beta_mult,
                              accept_nb_step, p1, p2, check_signature)
        return sol

    def search_single_thread(self,
                             name,
                             dated=True,
                             max_workers=None,
                             check_signature=False):
        """
        """
        seed = rd.randint(0, 1e6)

        print('start search')
        t0 = timer()

        sol = self.task(
            seed,
            self.cities,
            self.params['nb_step'],
            self.params['beta_mult'],
            self.params['accept_nb_step'],
            self.params['p1'],
            self.params['p2'],
            check_signature,
        )

        t1 = timer()
        print('run time = {:.2f} s'.format(t1-t0))

        return sol

    # concurrent search at Python in search_for_best
    def search_concurrent(self,
                          name,
                          dated=True,
                          max_workers=None,
                          check_signature=False):
        """
        """
        rd.seed()

        self.name = name
        print('start search')
        t0 = timer()

        # Create a pool of processes
        # By default, one is created for each logical CPU
        # To see how many CPUs on macOS: `sysctl hw.physicalcpu hw.logicalcpu`
        # To see how many CPUs on Linux: `lscpu | grep -E '^Thread|^CPU\('`
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

            dic_future_tag = {}
            nb_run = self.params['nb_run']
            for k in range(nb_run):
                seed = rd.randint(0, 1e6)
                future = executor.submit(
                    self.task,
                    seed,
                    self.cities,
                    self.params['nb_step'],
                    self.params['beta_mult'],
                    self.params['accept_nb_step'],
                    self.params['p1'],
                    self.params['p2'],
                    check_signature,
                )
                dic_future_tag[future] = k

            li_sol = []
            for future in concurrent.futures.as_completed(dic_future_tag):
                tag = dic_future_tag[future]
                try:
                    li_sol.append(future.result())
                except Exception as exc:
                    print(f'task {tag} generated an exception: {exc}')
                else:
                    print(f'task {1+tag}/{nb_run} done')

        t1 = timer()
        print('run time = {:.2f} s'.format(t1-t0))

        ts = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        tag = '-'+ts if dated else ''
        nb_city = self.params['nb_city']
        self.res = {
            'name': name,
            'timestamp': ts,
            'tag': tag,
            'nb_step': self.params['nb_step'],
            'cities': self.cities,
            'solutions': [{'dist': e[0], 'path': e[1]} for e in li_sol],
            'store_stats': [e[2] for e in li_sol],
            'runtime': t1-t0,
        }

    # multi-threaded C++ search in search_for_best_omp
    # def search_omp(self,
    #                name,
    #                n_run=1,
    #                dated=True,
    #                check_signature=False):
    #     """
    #     """
    #     rd.seed()

    #     self.name = name
    #     print('start search')
    #     t0 = timer()

    #     li_sol = search_for_best_omp(self.params['nb_run'],
    #                                  self.cities,
    #                                  self.params['nb_step'],
    #                                  self.params['beta_mult'],
    #                                  self.params['accept_nb_step'],
    #                                  self.params['p1'],
    #                                  self.params['p2'],
    #                                  check_signature=check_signature,
    #                                  )

    #     t1 = timer()
    #     print('run time = {:.2f} s'.format(t1-t0))

    #     ts = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    #     tag = '-'+ts if dated else ''
    #     nb_city = self.params['nb_city']
    #     self.res = {
    #         'name': name,
    #         'timestamp': f'{ts}',
    #         'tag': f'{tag}',
    #         'nb_step': self.params['nb_step'],
    #         'cities': self.cities,
    #         'solutions': [{'dist': e[0], 'path': e[1]} for e in li_sol],
    #         'store_stats': [e[2] for e in li_sol],
    #         'runtime': t1-t0,
    #     }

    def save_results(self):
        """
        """
        if self.res is None:
            print('No results to save')
            return

        result_folder = self.res['name'] + self.res['tag']

        folder = os.path.join('dump', result_folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder, f'results.json')
        with open(path, 'w') as f:
            f.write(json.dumps(self.res))

        print(f'results saved to {path}')

    def load_results(self, name):
        """
        """
        path = os.path.join('dump', name, 'results.json')
        if not os.path.exists(path):
            print(f'File {path} does not exist')
            return False

        with open(path, 'r') as f:
            self.res = json.loads(f.read())
            self.name = name

        print(f'results loaded from {path}')
        return True

    def load_cities(self, name):
        """
        """
        if self.load_results(name):
            self.cities = self.res['cities']

    def show_results(self, name=None, nb_best=None, size=5, save=False):
        """
        """
        if name is not None:
            if not self.load_results(name):
                return

        df = pd.DataFrame(data=[e['dist'] for e in self.res['solutions']], columns=['distance'])
        df.index.name = 'solution'
        df = df.sort_values('distance')
        display(df)

        if nb_best != 0:

            li_sol = [self.res['solutions'][k] for k in list(df.index)]

            if nb_best is not None:
                li_sol = li_sol[:nb_best]

            # create sublplots
            S = len(li_sol)
            n = min(S, 3)
            m = S // n

            if S % n:
                m += 1

            with pl.style.context(('ggplot')):

                fig, ax = pl.subplots(m, n,
                                      sharex='col',
                                      sharey='row',
                                      figsize=(n*size, m*size)
                                      )

                for k, sol in enumerate(li_sol):
                    cities = [self.res['cities'][k] for k in sol['path']]
                    dist = sol['dist']
                    N = len(cities)

                    if m == 1 and n == 1:
                        axs = ax
                    elif m == 1:
                        axs = ax[k]
                    else:
                        i = k // n
                        j = k % n
                        axs = ax[i, j]

                    for c in range(N-1):
                        axs.plot([cities[c][0], cities[c+1][0]],
                                 [cities[c][1], cities[c+1][1]], 'bo-')
                    axs.plot([cities[0][0], cities[N-1][0]], [cities[0][1], cities[N-1][1]], 'bo-')
                    axs.set_title('Dist='+str(dist))
                    # axs.axis([0.0, 1.0, 0.0, 1.0])
                    axs.axis([-0.05, 1.05, -0.05, 1.05])

                    if save and k == len(li_sol)-1:
                        shortest = li_sol[0]['dist']
                        # fig_name = 'TSP_solution_{}_dist_{}_r'.format(N, shortest)
                        fig_name = 'result'.format(N, shortest)

                        result_folder = self.res['name'] + self.res['tag']
                        result_folder = os.path.join('dump', result_folder)
                        if not os.path.exists(result_folder):
                            os.makedirs(result_folder)

                        path = os.path.join(result_folder, fig_name+'.png')
                        pl.savefig(path)

    def show_cities(self, save=True, size=5):
        """
        """
        cities = self.cities
        N = len(cities)

        with pl.style.context(('ggplot')):
            fig, ax = pl.subplots(1, 1, figsize=(size, size))
            li_x = [e[0] for e in cities]
            li_y = [e[1] for e in cities]
            ax.scatter(li_x, li_y, color='r')
            ax.axis([-0.05, 1.05, -0.05, 1.05])
            ax.set_title(f'seed = {self.seed}')

            if save:
                result_folder = 'dump'
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)

                fig_name = f'random_cities_{self.seed}'
                path = os.path.join('dump', fig_name+'.png')
                pl.savefig(path)
