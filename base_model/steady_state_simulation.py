import json
import statistics
from math import sqrt

from matplotlib import pyplot as plt

from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from utils.rvms import idfStudent

nodes = 4  # n nodi
arrival_time = 30.0
arrival_time_morning = 15.0
arrival_time_afternoon = 5.0
arrival_time_evening = 15.0
arrival_time_night = 25.0

b = 512
k = 64
batch_index = 0

seed = 123456789  # TODO: Controlla il seed migliore o forse era il multiplier?
START = 8.0 * 1440
STOP = 20 * 12 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
# ARCADE1 = 2
# ARCADE2 = 3
# ARCADE3 = 4
p_size = 0.6


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
    selectStream(5)  # TODO: scegliere gli streams sequenziali o no?
    if from_tkt_queue:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    # if r <= 1 / (nodes - 1):
    #     return ARCADE1
    # elif r <= 2 / (nodes - 1):
    #     return ARCADE2
    # else:
    #     return ARCADE3

    # Caso arrivo dall'esterno

    r = random()
    if r <= p_ticket_queue:
        global arr_est
        arr_est += 1
        # print(arr_est)
        return TICKET_QUEUE
    else:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    #  if r <= 1.0 / float(nodes - 1):
    #      # print(1.0 / float(nodes - 1))
    #      return ARCADE1
    #  elif r <= 2.0 / float(nodes - 1):
    #      # print(2.0 / float(nodes - 1))
    #      return ARCADE2
    #  else:
    #      return ARCADE3


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


# TODO: Provare l'esponenziale troncata e verificare i tempi di interarrivo
def get_arrival(y):
    # ---------------------------------------------
    # * generate the next arrival time from an Exponential distribution.
    # * --------------------------------------------

    selectStream(0)
    return Exponential(y)


def get_service(id_node):
    # --------------------------------------------
    # * generate the next service time
    # * --------------------------------------------
    # */
    if id_node == TICKET_QUEUE:
        selectStream(6)
        r = random()
        if r <= p_size:  # green pass
            selectStream(id_node + 10)
            service = TruncatedNormal(2, 1.5, 1, 3)  # green pass
            # print("Green pass: ", service)
            return service
        else:
            selectStream(id_node + 15)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            # print("covid test: ", service)
            return service
    else:
        selectStream(id_node + 10)
        service = TruncatedNormal(30, 9, 1, 100)  # arcade game time
        # print("arcade game time: ", service)
        return service


batch_means_info = {
    "seed": 0,
    "n_nodes": 0,
    "lambda": 0.0,
    "b": 0,
    "k": 0,
    "avg_wait_ticket": [],
    "std_ticket": [],
    "w_ticket": [],
    "avg_delay_arcades": [],
    "std_arcades": [],
    "w_arcades": [],
    "final_wait_ticket": 0.0,
    "final_std_ticket": 0.0,
    "final_w_ticket": 0.0,
    "final_delay_arcades": 0.0,
    "final_std_arcades": 0.0,
    "final_w_arcades": 0.0,
    "correlation": []
}

actual_stats = {
    "wait_ticket": 0.0,
    "delay_arcades": 0.0
}

job_list = []


def online_variance(n, mean, variance, x):
    delta = x - mean
    variance = variance + delta * delta * (n - 1) / n
    mean = mean + delta / n
    return mean, variance


def plot_stats():
    x = [i for i in range(0, len(batch_means_info["avg_delay_arcades"]))]  # in 0 global stats
    y = (batch_means_info["avg_delay_arcades"][:])  # in 0 global stats
    print(x)
    print(y)
    # plt.plot(x, y)

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
    # settings
    batch_means_info["seed"] = seed
    batch_means_info["b"] = b
    batch_means_info["k"] = k
    batch_means_info["n_nodes"] = nodes - 1
    batch_means_info["lambda"] = 1.0 / arrival_time
    node_list = [StatusNode(i) for i in range(nodes + 1)]  # in 0 global stats
    plantSeeds(seed)

    time.current = START
    arrival = START  # global temp var for getArrival function     [minutes]

    # initialization of the first arrival event
    arrival += get_arrival(arrival_time)
    node = node_list[select_node(False)]
    node.arrival = arrival
    min_arrival = arrival
    old_index = 0

    while node_list[0].index <= b * k:  # (node_list[0].number > 0)
        # print(node_list[0].index)
        if node_list[0].index % b == 0 and node_list[0].index != 0 and old_index != node_list[0].index:
            old_index = node_list[0].index
            avg_wait_ticket = 0.0
            avg_delay_arcades = 0.0
            std_ticket = 0.0
            std_arcades = 0.0
            n = 0
            mean = 0
            M2 = 0
            for i in range(b * batch_index, b * batch_index + b):
                # print("len job list: ", len(job_list), ", index: ",node_list[0].index, ", batch_index: ",batch_index,", begin for: ", b * batch_index, ", end for: ", b * batch_index + b, ", elem_index: ", i)
                n += 1
                #  avg calculation,  std calculation

                avg_wait_ticket, std_ticket = online_variance(n, avg_wait_ticket, std_ticket,
                                                              job_list[i]["wait_ticket"])
                avg_delay_arcades, std_arcades = online_variance(n, avg_delay_arcades, std_arcades,
                                                                 job_list[i]["delay_arcades"])

            std_ticket = statistics.variance([job_list[i]["wait_ticket"] for i in range(b * batch_index, b * batch_index + b)])
            std_arcades = statistics.variance([job_list[i]["delay_arcades"] for i in range(b * batch_index, b * batch_index + b)])
            std_ticket = sqrt(std_ticket)
            std_arcades = sqrt(std_arcades)
            #  calculate interval width
            LOC = 0.95
            u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
            t = idfStudent(n - 1, u)  # critical value of t
            print(std_arcades, t)
            w_ticket = t * std_ticket / sqrt(n - 1)  # interval half width
            w_arcades = t * std_arcades / sqrt(n - 1)  # interval half width
            #  update dictionary
            batch_means_info["avg_wait_ticket"].append(avg_wait_ticket)
            batch_means_info["avg_delay_arcades"].append(avg_delay_arcades)
            batch_means_info["std_ticket"].append(std_ticket)
            batch_means_info["std_arcades"].append(std_arcades)
            batch_means_info["w_ticket"].append(w_ticket)
            batch_means_info["w_arcades"].append(w_arcades)
            batch_index += 1

        node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
        time.next = minimum(node_to_process.arrival, node_to_process.completion)
        # Aggiornamento delle aree basate sul giro prima
        for i in range(0, len(node_list)):
            if node_list[i].number > 0:
                # if i == 0 or i == node_to_process.id:
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

            # Se il prossimo arrivo è su un altro centro, bisogna eliminare l'arrivo sul centro processato altrimenti
            # sarà sempre il minimo
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
                act_st = dict(actual_stats)
                act_st["wait_ticket"] = node_list[1].stat.node / node_list[1].index
                delay_arcades_avg = 0
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

            if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on ARCADE_i
                arcade_node = node_list[select_node(True)]  # on first global stats

                # Update partial stats for arcade nodes
                # if arcade_node.number > 0:
                #     arcade_node.stat.node += (time.next - current_for_update) * arcade_node.number
                #     arcade_node.stat.queue += (time.next - current_for_update) * (arcade_node.number - 1)
                #     arcade_node.stat.service += (time.next - current_for_update)

                arcade_node.number += 1  # system stats don't updated
                arcade_node.last = time.current

                if arcade_node.number == 1:
                    arcade_node.completion = time.current + get_service(arcade_node.id)

        arrival_list = [node_list[n].arrival for n in range(1, len(node_list))]
        min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]
    with open("prova.json", 'w+') as json_file:
        json.dump(batch_means_info, json_file, indent=4)
    json_file.close()
    plot_stats()
    # for i in range(0, len(node_list)):
    #    print(node_list[i].last)
    #    print("\n\nNode " + str(i))
    #    print("\nfor {0} jobs".format(node_list[i].index))
    #    print("   average interarrival time = {0:6.6f}".format(node_list[i].last / node_list[i].index))
    #    print("   average wait ............ = {0:6.6f}".format(node_list[i].stat.node / node_list[i].index))
    #    print("   average delay ........... = {0:6.6f}".format(node_list[i].stat.queue / node_list[i].index))
    #    print("   average # in the node ... = {0:6.6f}".format(node_list[i].stat.node / time.current))
    #    print("   average # in the queue .. = {0:6.6f}".format(node_list[i].stat.queue / time.current))
    #    print("   utilization ............. = {0:6.6f}".format(node_list[i].stat.service / time.current))
