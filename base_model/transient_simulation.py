import json
import os
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream
from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from utils.rvms import idfStudent

stationary = True
nodes = 2  # n nodi
arrival_time = 35.0
arrival_time_morning = 14.0  # nodes = 3 min
arrival_time_afternoon = 5.0  # nodes = 4 min
arrival_time_evening = 14.0
arrival_time_night = 35.0  # nodes = 2 min


seeds = [987654321, 539458255, 482548808]
replicas = 64
sampling_frequency = 75

b = 128
k = 160
START = 8.0 * 60
STOP = 1000 * 12 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
p_size = 0.6
p_positive = 0.05


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    last = 0.0  # last arrival time
    index = 0.0  # jobs departed
    number = 0.0  # jobs in node
    stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.stat = Track()  # Track stats


def set_arrival_time(x):
    global arrival_time
    arrival_time = x


def get_arrival_time():
    return arrival_time


arr_est = 0


def select_node(from_tkt_queue):
    selectStream(select_node_stream)
    if from_tkt_queue:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    # Caso arrivo dall'esterno
    r = random()
    if r <= p_ticket_queue:
        global arr_est
        arr_est += 1
        return TICKET_QUEUE
    else:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1


def minimum(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return None
    elif a < b:
        return a
    else:
        return b


def next_event():
    time_event = []
    for i in range(1, len(node_list)):
        time_event.append(node_list[i].arrival)
        time_event.append(node_list[i].completion)

    time_event = sorted(time_event, key=lambda x: (x is None, x))

    for i in range(1, len(time_event)):
        if time_event[0] == node_list[i].arrival or time_event[0] == node_list[i].completion:
            return i


def get_arrival(y):
    # ---------------------------------------------
    # * generate the next arrival time from an Exponential distribution.
    # * --------------------------------------------

    selectStream(select_node_arrival)
    return Exponential(y)


def get_service(id_node):
    # --------------------------------------------
    # * generate the next service time
    # * --------------------------------------------
    # */
    if id_node == TICKET_QUEUE:
        selectStream(select_node_random)
        r = random()
        if r <= p_size:  # green pass
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(2, 1.5, 1, 3)  # green pass
            return service
        else:
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            return service
    else:
        selectStream(id_node + select_node_arcades)
        service = TruncatedNormal(15, 3, 10, 20)  # arcade game time
        return service


dict_list = []

select_node_positive = 75


def is_positive():
    selectStream(select_node_positive)
    r = random()
    if r <= p_positive:
        return True
    else:
        return False


def online_variance(n, mean, variance, x):
    delta = x - mean
    variance = variance + delta * delta * (n - 1) / n
    mean = mean + delta / n
    return mean, variance


def plot_stats_global():
    x = [i * sampling_frequency for i in range(0, len(dict_list[0]["avg_wait_system"]))]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    fig1 = plt.figure(figsize=(16, 9), dpi=400)
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels

    for i in range(0, len(dict_list)):
        plt.plot(x, [dict_list[i]["avg_wait_system"][j] for j in range(0, len(dict_list[i]["avg_wait_system"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none', figure=fig1)

    plt.legend(["seed = " + str(dict_list[0]["seed"]), "seed = " + str(dict_list[1]["seed"]),
                "seed = " + str(dict_list[2]["seed"])])

    plt.xlabel("Number of jobs")
    plt.ylabel("Avg wait system (minutes)")
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    if stationary:
        plt.savefig(fname=results_dir + "/transient_night_s", bbox_inches='tight')
    else:
        plt.savefig(fname=results_dir + "/transient_night_ns", bbox_inches='tight')
    plt.show()


def plot_stats():
    x = [i for i in range(0, len(batch_means_info["avg_delay_arcades"]))]  # in 0 global stats
    y = (batch_means_info["avg_delay_arcades"][:])  # in 0 global stats
    print(x)
    print(y)

    plt.errorbar(x, y, yerr=batch_means_info["w_arcades"][:], fmt='.', color='black',
                 ecolor='red', elinewidth=3, capsize=0)
    plt.tight_layout()

    plt.legend(["Gain"])
    plt.title("Avg delay system")
    plt.xlabel("Configuration")
    plt.ylabel("Gain function")
    plt.show()
    x1 = [i for i in range(0, len(job_list))]
    y1 = [i["delay_arcades"] for i in job_list]
    plt.errorbar(x1, y1, fmt='.')
    plt.show()


if __name__ == '__main__':
    for seed in seeds:

        # settings
        batch_means_info_struct = {
            "seed": 0,
            "n_nodes": 0,
            "lambda": 0.0,
            "b": 0,
            "k": 0,
            "job_list": [],
            "avg_wait_ticket": [],  # [elem 0-50, elem 50-100, ..]
            "std_ticket": [],
            "w_ticket": [],
            "avg_delay_arcades": [],
            "std_arcades": [],
            "w_arcades": [],
            "avg_wait_system": [],
            "std_system": [],
            "w_system": [],
            "final_wait_ticket": 0.0,
            "final_std_ticket": 0.0,
            "final_w_ticket": 0.0,
            "final_delay_arcades": 0.0,
            "final_std_arcades": 0.0,
            "final_w_arcades": 0.0,
            "correlation_delay_arcades": 0.0
        }

        batch_means_info = batch_means_info_struct
        batch_means_info["seed"] = seed
        batch_means_info["b"] = b
        batch_means_info["k"] = k
        batch_means_info["n_nodes"] = nodes - 1
        batch_means_info["lambda"] = 1.0 / arrival_time
        print(batch_means_info)
        node_list = [StatusNode(i) for i in range(nodes + 1)]  # in 0 global stats

        plantSeeds(seed)

        for replica in range(0, replicas):
            job_list = []
            batch_means_info["job_list"] = job_list
            for center in node_list:
                center.number = 0.0
                center.last = 0.0
                center.arrival = None
                center.completion = None
                center.index = 0.0
                center.stat.node = 0.0
                center.stat.queue = 0.0
                center.stat.service = 0.0

            batch_index = 0
            time.current = START
            arrival = START  # global temp var for getArrival function     [minutes]

            # initialization of the first arrival event
            arrival += get_arrival(arrival_time)
            node = node_list[select_node(False)]
            node.arrival = arrival
            min_arrival = arrival
            old_index = 0

            while node_list[0].index <= b * k:  # (node_list[0].number > 0)

                if node_list[0].index % sampling_frequency == 0 and node_list[0].index != 0 and old_index != node_list[
                    0].index:
                    old_index = node_list[0].index

                    old_index_arcades = 0
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            old_index_arcades += center.index
                    old_index_arcades = int(old_index_arcades)

                    if replica == 0:
                        batch_means_info["avg_wait_ticket"].append(
                            job_list[old_index_arcades - 1]["wait_ticket"])
                        batch_means_info["std_ticket"].append(0.0)

                        batch_means_info["avg_delay_arcades"].append(
                            job_list[old_index_arcades - 1]["delay_arcades"])
                        batch_means_info["std_arcades"].append(0.0)

                        batch_means_info["avg_wait_system"].append(
                            job_list[old_index_arcades - 1]["wait_system"])
                        batch_means_info["std_system"].append(0.0)
                    else:
                        batch_means_info["avg_wait_ticket"][batch_index], batch_means_info["std_ticket"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_wait_ticket"][batch_index],
                                                           batch_means_info["std_ticket"][batch_index],
                                                           job_list[old_index_arcades - 1]["wait_ticket"])
                        batch_means_info["avg_delay_arcades"][batch_index], batch_means_info["std_arcades"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_delay_arcades"][batch_index],
                                                           batch_means_info["std_arcades"][batch_index],
                                                           job_list[old_index_arcades - 1]["delay_arcades"])
                        batch_means_info["avg_wait_system"][batch_index], batch_means_info["std_system"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_wait_system"][batch_index],
                                                           batch_means_info["std_system"][batch_index],
                                                           job_list[old_index_arcades - 1]["wait_system"])

                    batch_index += 1

                node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
                time.next = minimum(node_to_process.arrival, node_to_process.completion)
                # Aggiornamento delle aree basate sul giro prima
                for i in range(0, len(node_list)):
                    if node_list[i].number > 0:
                        node_list[i].stat.node += (time.next - time.current) * node_list[i].number
                        node_list[i].stat.queue += (time.next - time.current) * (node_list[i].number - 1)
                        node_list[i].stat.service += (time.next - time.current)

                current_for_update = time.current
                time.current = time.next  # advance the clock

                if time.current == node_to_process.arrival:

                    node_to_process.number += 1
                    node_list[0].number += 1  # update system stat
                    arrival += get_arrival(arrival_time)
                    node_selected_pos = select_node(False)

                    if node_selected_pos != node_to_process.id:
                        node_to_process.arrival = INFINITY
                    node = node_list[node_selected_pos]

                    if node.arrival != INFINITY:
                        node.last = node.arrival
                        if node.last is not None and node_list[0].last is not None and node_list[0].last < node.last:
                            node_list[0].last = node.last
                    # update node and system last arrival time

                    # Controllo che l'arrivo sul nodo i-esimo sia valido. In caso negativo
                    # imposto come ultimo arrivo del nodo i-esimo l'arrivo precedentemente
                    # considerato
                    if arrival > STOP:
                        if node.arrival != INFINITY:
                            node.last = node.arrival
                        # update node and system last arrival time
                        if node_list[0].last < node.last:
                            node_list[0].last = node.last
                        node.arrival = INFINITY
                    else:
                        node.arrival = arrival

                    if node_to_process.number == 1:
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                else:
                    node_to_process.index += 1  # node stats update
                    node_to_process.number -= 1
                    if node_to_process.id != TICKET_QUEUE:  # system stats update
                        node_list[0].index += 1
                        node_list[0].number -= 1

                        #  Inserimento statistiche puntuali ad ogni completamento
                        actual_stats = {
                            "wait_ticket": 0.0,
                            "delay_arcades": 0.0,
                            "wait_system": 0.0
                        }
                        act_st = actual_stats
                        if node_list[0].index != 0:
                            act_st["wait_system"] = node_list[0].stat.node / node_list[0].index
                        if node_list[1].index != 0:
                            act_st["wait_ticket"] = node_list[1].stat.node / node_list[1].index
                        delay_arcades_avg = 0.0
                        for i in range(2, nodes + 1):
                            if node_list[i].index != 0:
                                delay_arcades_avg += (node_list[i].stat.queue / node_list[i].index)
                        delay_arcades_avg = delay_arcades_avg / (nodes - 1.0)
                        act_st["delay_arcades"] = delay_arcades_avg
                        job_list.append(act_st)

                    if node_to_process.number > 0:
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                    else:
                        node_to_process.completion = INFINITY

                    if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on
                        # ARCADE_i
                        if not is_positive():
                            arcade_node = node_list[select_node(True)]  # on first global stats

                            arcade_node.number += 1  # system stats don't updated
                            arcade_node.last = time.current

                            if arcade_node.number == 1:
                                arcade_node.completion = time.current + get_service(arcade_node.id)
                        else:
                            node_list[0].index += 1
                            node_list[0].number -= 1

                arrival_list = [node_list[n].arrival for n in range(1, len(node_list))]
                min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]

            #  Global batch means

        dict_list.append(batch_means_info)
        for i in range(0, len(batch_means_info["std_arcades"])):
            batch_means_info["std_arcades"][i] = sqrt(batch_means_info["std_arcades"][i] / replicas)
            batch_means_info["std_ticket"][i] = sqrt(batch_means_info["std_ticket"][i] / replicas)
            batch_means_info["std_system"][i] = sqrt(batch_means_info["std_system"][i] / replicas)
            if replicas > 1:
                LOC = 0.95
                u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
                t = idfStudent(replicas - 1, u)  # critical value of t
                w_ticket = t * batch_means_info["std_ticket"][i] / sqrt(replicas - 1)  # interval half width
                w_arcades = t * batch_means_info["std_arcades"][i] / sqrt(replicas - 1)  # interval half width
                w_system = t * batch_means_info["std_system"][i] / sqrt(replicas - 1)  # interval half width
                batch_means_info["w_ticket"].append(w_ticket)
                batch_means_info["w_arcades"].append(w_arcades)
                batch_means_info["w_system"].append(w_system)
        path = "stats_" + str(seed) + ".json"
        with open(path, 'w+') as json_file:
            json.dump(batch_means_info, json_file, indent=4)
        json_file.close()
        # plot_stats()

    plot_stats_global()
