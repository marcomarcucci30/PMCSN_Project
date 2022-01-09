import json
import os

from scipy.stats import pearsonr
import statistics
from math import sqrt

from matplotlib import pyplot as plt
from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from utils.rvms import idfStudent
from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream
from advanced_model.skeleton import select_queue_premium

nodes = 4  # n nodi
arrival_time = 35.0
arrival_time_morning = 14.0  # nodes = 3 min
arrival_time_afternoon = 5.0  # nodes = 4 min
arrival_time_evening = 14.0
arrival_time_night = 35.0  # nodes = 2 min

b = 256
k = 64
seed = 1234567891
START = 8.0 * 60
STOP = 1000 * 12 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
p_size = 0.6
p_premium = 0.36
p_positive = 0.05
ticket_price = 10.0
ticket_price_premium = 20.0
energy_cost = 0.4
nodes_min = 3
nodes_max = 3
delay_max = 20.0
delay_min = 8.0
seeds = [987654321, 539458255, 482548808,
         1757116804, 238927874, 841744376,
         1865511657, 482548808,430131813, 725267564]
income_list = []



def ticket_refund(avg_delay_arcades):
    perc = (avg_delay_arcades - delay_min) / (delay_max - delay_min)
    if perc > 0.8:
        return 0.8
    else:
        return perc


select_node_positive = 75


def is_positive():
    selectStream(select_node_positive)
    r = random()
    if r <= p_positive:
        return True
    else:
        return False


def plot_income():
    x = [str(income_list[i][1]) for i in range(0, len(income_list))]
    y1 = [income_list[i][0] for i in range(0, len(income_list))]
    plt.xticks(rotation=45)
    plt.plot(x, y1, 'o', mfc='none', color='black')
    plt.show()
    y2 = [income_list[i][2] for i in range(0, len(income_list))]
    plt.plot(x, y2, 'o', mfc='none', color='red')
    plt.show()


def plot_stats_global():
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=400)
    x = [str(dict_list[i]["seed"]) for i in range(0, len(dict_list))]
    y = [dict_list[i]["final_wait_system"] for i in range(0, len(dict_list))]
    plt.xticks(rotation=45)
    axs[0].set_ylabel(ylabel="Avg wait system (minutes)", fontsize=15)
    axs[0].tick_params(labelsize=10)

    axs[0].errorbar(x, y, yerr=[dict_list[i]["final_w_system"] for i in range(0, len(dict_list))], fmt='.',
                 color='blue',
                 ecolor='red', elinewidth=3, capsize=0)

    axs[0].set_xlabel(xlabel="Seed", fontsize=15)

    cellText = []
    for j in range(len(dict_list)):
        # Building cellText to create table
        row = []
        row.append(str(dict_list[j]["seed"]))
        row.append(str(dict_list[j]["final_wait_system"]))
        row.append(str(dict_list[j]["final_std_system"]))
        row.append("±" + str(dict_list[j]["final_w_system"]))
        row.append("95%")
        cellText.append(row)

    # Plotting Table and Graph
    cols = ("SEED", "MEAN VALUE", "STD", "CONFIDENCE INTERVAL", "CONFIDENCE LEVEL")
    axs[1].axis('tight')
    axs[1].axis('off')
    axs[1].table(cellText=cellText,
                 cellLoc='center',
                 colLabels=cols,
                 loc='center')


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    plt.savefig(fname=results_dir + "/adv_avg_ws_steady_state_night", bbox_inches='tight')

    plt.show()


def plot_stats_global_ticket():
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=400)
    x = [str(dict_list[i]["seed"]) for i in range(0, len(dict_list))]
    y = [dict_list[i]["final_delay_ticket_green_pass"] for i in range(0, len(dict_list))]
    plt.xticks(rotation=45)
    axs[0].set_ylabel(ylabel="Avg delay Covid-19 Green Pass (minutes)", fontsize=15)
    axs[0].tick_params(labelsize=10)

    axs[0].errorbar(x, y, yerr=[dict_list[i]["final_w_ticket_green_pass"] for i in range(0, len(dict_list))], fmt='.',
                 color='blue',
                 ecolor='red', elinewidth=3, capsize=0)

    axs[0].set_xlabel(xlabel="Seed", fontsize=15)

    cellText = []
    for j in range(len(dict_list)):
        # Building cellText to create table
        row = []
        row.append(str(dict_list[j]["seed"]))
        row.append(str(dict_list[j]["final_delay_ticket_green_pass"]))
        row.append(str(dict_list[j]["final_std_ticket_green_pass"]))
        row.append("±" + str(dict_list[j]["final_w_ticket_green_pass"]))
        row.append("95%")
        cellText.append(row)

    # Plotting Table and Graph
    cols = ("SEED", "MEAN VALUE", "STD", "CONFIDENCE INTERVAL", "CONFIDENCE LEVEL")
    axs[1].axis('tight')
    axs[1].axis('off')
    axs[1].table(cellText=cellText,
                 cellLoc='center',
                 colLabels=cols,
                 loc='center')


    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    plt.savefig(fname=results_dir + "/adv_avg_d_covid_steady_state_night", bbox_inches='tight')

    plt.show()


def plot_stats():
    x = [i for i in range(0, len(batch_means_info["avg_delay_arcades"]))]  # in 0 global stats
    y = (batch_means_info["avg_delay_arcades"][:])  # in 0 global stats
    plt.errorbar(x, y, fmt='.', color='black',
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


def plot_correlation():
    x = [i for i in range(0, len(dict_list[0]["correlation_delay_arcades"]))]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    plt.xticks(rotation=45)
    plt.rcParams["figure.figsize"] = (16, 9)

    for i in range(0, len(dict_list)):
        plt.plot(x, dict_list[i]["correlation_delay_arcades"], 'o',
                 linestyle='--', color=colors[i], label=dict_list[i]["seed"], mfc='none')

    plt.xlabel("Lag")
    plt.ylabel("Correlation wait system")
    plt.show()


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service
    index = 0.0  # jobs departed
    index_support = 0.0
    number = 0.0  # jobs in node
    last = 0.0

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0
        self.index = 0.0  # jobs departed
        self.index_support = 0.0
        self.number = 0.0  # jobs in node
        self.last = 0.0


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    priority_arrival = False
    priority_completion = False
    last = 0.0  # last arrival time
    more_p_stat = None
    less_p_stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.more_p_stat = Track()  # Track_more_p stats
        self.less_p_stat = Track()  # Track_less_p stats


class SystemTrack:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class SystemNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    last = 0.0  # last arrival time
    index = 0.0  # jobs departed
    index_support = 0.0
    number = 0.0  # jobs in node
    stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.stat = SystemTrack()  # Track stats


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


def maximum(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return None
    elif a > b:
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
        if node_list[id_node].priority_completion is True:  # green pass
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(2, 1.5, 1, 3)  # green pass
            return service
        else:
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            return service
    else:
        selectStream(id_node + select_node_arcades)
        service = TruncatedNormal(15, 3, 3, 25)  # arcade game time
        return service


def select_queue(node_id):
    if node_id == TICKET_QUEUE:
        selectStream(select_node_random)
        r = random()
        if r <= p_size:  # green pass
            node_list[node_id].priority_arrival = True  # green pass
            return
        else:
            node_list[node_id].priority_arrival = False  # test
            return
    else:
        selectStream(select_queue_premium)
        r = random()
        if r <= p_premium:
            node_list[node_id].priority_arrival = True  # premium ticket
            return
        else:
            node_list[node_id].priority_arrival = False  # standard ticket
            return


dict_list = []


def online_variance(n, mean, variance, x):
    delta = x - mean
    variance = variance + delta * delta * (n - 1) / n
    mean = mean + delta / n
    return mean, variance


if __name__ == '__main__':
    for n_nodes in range(nodes_min, nodes_max + 1):
        nodes = n_nodes
        dict_list = []
        for seed in seeds:
            batch_index = 0
            job_list = []

            # settings
            batch_means_info_struct = {
                "seed": 0,
                "n_nodes": 0,
                "lambda": 0.0,
                "b": 0,
                "k": 0,
                "income": [],
                "job_list": [],
                "avg_delay_ticket_green_pass": [],  # [elem 0-50, elem 50-100, ..]
                "avg_delay_arcades": [],
                "avg_delay_arcades_priority": [],
                "avg_wait_system": [],

                "final_delay_ticket_green_pass": 0.0,
                "final_std_ticket_green_pass": 0.0,
                "final_w_ticket_green_pass": 0.0,

                "final_delay_arcades": 0.0,
                "final_std_arcades": 0.0,
                "final_w_arcades": 0.0,

                "final_delay_arcades_priority": 0.0,
                "final_std_arcades_priority": 0.0,
                "final_w_arcades_priority": 0.0,

                "final_wait_system": 0.0,
                "final_std_system": 0.0,
                "final_w_system": 0.0,

                "correlation_delay_arcades": []
            }

            batch_means_info = batch_means_info_struct
            batch_means_info["seed"] = seed
            batch_means_info["b"] = b
            batch_means_info["k"] = k
            batch_means_info["n_nodes"] = nodes - 1
            batch_means_info["lambda"] = 1.0 / arrival_time
            # settings
            node_list = []
            for i in range(nodes + 1):
                if i == 0:
                    node_list.append(SystemNode(i))
                else:
                    node_list.append(StatusNode(i))
            plantSeeds(seed)

            time.current = START
            arrival = START  # global temp var for getArrival function     [minutes]

            # initialization of the first arrival event
            arrival += get_arrival(arrival_time)
            node = node_list[select_node(False)]  # node in cui schedulare l'arrivo
            select_queue(node.id)  # discriminazione della coda relativa all'arrivo
            node.arrival = arrival
            min_arrival = arrival
            old_index = 0
            old_batch_current = START

            while node_list[0].index_support <= b * (k - 1):
                if node_list[0].index % b == 0 and node_list[0].index != 0:  # and old_index != node_list[0].index:
                    index_arcades = 0  # all completed jobs from standard
                    index_arcades_total = 0
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            index_arcades += center.more_p_stat.index + center.less_p_stat.index
                            index_arcades_total += center.more_p_stat.index_support + center.less_p_stat.index_support
                    index_arcades = int(index_arcades)
                    index_arcades_total = int(index_arcades_total)
                    old_index = node_list[0].index
                    avg_delay_ticket_green_pass = job_list[index_arcades_total + index_arcades - 1][
                        "delay_ticket_green_pass"]  # prendo l'ultimo elemento
                    avg_delay_arcades = job_list[index_arcades_total + index_arcades - 1][
                        "delay_arcades"]  # che rappresenta la media sul
                    avg_delay_arcades_priority = job_list[index_arcades_total + index_arcades - 1][
                        "delay_arcades_priority"]  # che rappresenta la media sul
                    avg_wait_system = job_list[index_arcades_total + index_arcades - 1]["wait_system"]
                    index_more_p_arcades = 0  # all completed jobs from premium
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            index_more_p_arcades += center.more_p_stat.index

                    index_less_p_arcades = 0  # all completed jobs from standard
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            index_less_p_arcades += center.less_p_stat.index

                    income = index_less_p_arcades * ticket_price + index_more_p_arcades * ticket_price_premium - \
                             (nodes - 1) * (((time.current-old_batch_current) / 10) * energy_cost) - \
                             (index_less_p_arcades * ticket_refund(avg_delay_arcades) * ticket_price +
                              index_more_p_arcades * ticket_refund(avg_delay_arcades_priority) * ticket_price_premium)

                    #  batch
                    #  azzeriamo le statistiche (index e track)?

                    for center in node_list:
                        if center.id == 0:  # System node
                            center.index_support += center.index
                            center.index = 0.0
                            center.stat.node = 0.0
                            center.stat.queue = 0.0
                            center.stat.service = 0.0
                        else:
                            #  reset stats related to first priority queue of the center
                            center.more_p_stat.index_support += center.more_p_stat.index
                            center.more_p_stat.index = 0.0
                            center.more_p_stat.node = 0.0
                            center.more_p_stat.queue = 0.0
                            center.more_p_stat.service = 0.0

                            #  reset stats related to second priority queue of the center
                            center.less_p_stat.index_support += center.less_p_stat.index
                            center.less_p_stat.index = 0.0
                            center.less_p_stat.node = 0.0
                            center.less_p_stat.queue = 0.0
                            center.less_p_stat.service = 0.0

                    batch_means_info["avg_delay_ticket_green_pass"].append(avg_delay_ticket_green_pass)
                    batch_means_info["avg_delay_arcades"].append(avg_delay_arcades)
                    batch_means_info["avg_delay_arcades_priority"].append(avg_delay_arcades_priority)
                    batch_means_info["avg_wait_system"].append(avg_wait_system)
                    batch_means_info["income"].append(income)
                    batch_index += 1
                    old_batch_current = time.current

                node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
                time.next = minimum(node_to_process.arrival, node_to_process.completion)
                # Aggiornamento delle aree basate sul giro prima
                for i in range(0, len(node_list)):
                    if i != 0:
                        if node_list[i].priority_completion:
                            if node_list[i].more_p_stat.number > 0:
                                node_list[i].more_p_stat.node += (time.next - time.current) * node_list[
                                    i].more_p_stat.number
                                node_list[i].more_p_stat.queue += (time.next - time.current) * (
                                        node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.service += (time.next - time.current)
                            if node_list[i].less_p_stat.number > 0:
                                node_list[i].less_p_stat.node += (time.next - time.current) * (
                                        node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.queue += (time.next - time.current) * (
                                        node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.service += (time.next - time.current)
                        else:
                            if node_list[i].more_p_stat.number > 0:
                                node_list[i].more_p_stat.node += (time.next - time.current) * (
                                        node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.queue += (time.next - time.current) * (
                                        node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.service += (time.next - time.current)
                            if node_list[i].less_p_stat.number > 0:
                                node_list[i].less_p_stat.node += (time.next - time.current) * (
                                    node_list[i].less_p_stat.number)
                                node_list[i].less_p_stat.queue += (time.next - time.current) * (
                                        node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.service += (time.next - time.current)
                    else:
                        if node_list[i].number > 0:
                            node_list[i].stat.node += (time.next - time.current) * node_list[i].number
                            node_list[i].stat.queue += (time.next - time.current) * (node_list[i].number - 1)
                            node_list[i].stat.service += (time.next - time.current)

                current_for_update = time.current
                time.current = time.next  # advance the clock

                if time.current == node_to_process.arrival:

                    if node_to_process.priority_arrival is True:  # vediamo su quale coda è stato schedulato l'arrivo
                        # che stiamo processando
                        node_to_process.more_p_stat.number += 1
                    else:
                        node_to_process.less_p_stat.number += 1

                    node_list[0].number += 1  # update system stat
                    arrival += get_arrival(arrival_time)
                    node_selected_pos = select_node(False)
                    select_queue(node_selected_pos)  # discriminazione della coda relativa all'arrivo
                    # Se il prossimo arrivo è su un altro centro, bisogna eliminare l'arrivo sul centro processato
                    # altrimenti sarà sempre il minimo
                    if node_selected_pos != node_to_process.id:
                        node_to_process.arrival = INFINITY
                    node = node_list[node_selected_pos]

                    if node.arrival != INFINITY:
                        if node.priority_arrival is True:
                            node.more_p_stat.last = node.arrival
                        else:
                            node.less_p_stat.last = node.arrival

                        max_last = maximum(node.more_p_stat.last, node.less_p_stat.last)
                        if node.more_p_stat.last is not None and node_list[0].last is not None and node_list[
                            0].last < max_last:
                            node_list[0].last = max_last
                    # update node and system last arrival time

                    # Controllo che l'arrivo sul nodo i-esimo sia valido. In caso negativo
                    # imposto come ultimo arrivo del nodo i-esimo l'arrivo precedentemente
                    # considerato
                    if arrival > STOP:
                        if node.arrival != INFINITY:
                            if node.priority_arrival is True:
                                node.more_p_stat.last = node.arrival
                            else:
                                node.less_p_stat.last = node.arrival
                        # update node and system last arrival time
                        max_last = maximum(node.more_p_stat.last, node.less_p_stat.last)
                        if node_list[0].last < max_last:
                            node_list[0].last = max_last
                        node.arrival = INFINITY
                    else:
                        node.arrival = arrival

                    # caso sistema completamente vuoto prima dell'arrivo del job su node_to_process
                    if node_to_process.more_p_stat.number == 1 and node_to_process.less_p_stat.number == 0:
                        node_to_process.priority_completion = True
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                    elif node_to_process.more_p_stat.number == 0 and node_to_process.less_p_stat.number == 1:
                        node_to_process.priority_completion = False
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                else:
                    if node_to_process.priority_completion is True:
                        node_to_process.more_p_stat.index += 1  # node stats update
                        node_to_process.more_p_stat.number -= 1
                    else:
                        node_to_process.less_p_stat.index += 1  # node stats update
                        node_to_process.less_p_stat.number -= 1

                    if node_to_process.id != TICKET_QUEUE:  # system stats update
                        node_list[0].index += 1
                        node_list[0].number -= 1

                        #  Inserimento statistiche puntuali ad ogni completamento
                        actual_stats = {
                            "delay_ticket_green_pass": 0.0,
                            "delay_arcades": 0.0,
                            "delay_arcades_priority": 0.0,
                            "wait_system": 0.0
                        }
                        act_st = actual_stats
                        if node_list[0].index != 0:
                            act_st["wait_system"] = node_list[0].stat.node / node_list[0].index
                        if node_list[1].more_p_stat.index != 0:  # prendiamo il tempo di risposte del green pass
                            act_st["delay_ticket_green_pass"] = node_list[1].more_p_stat.queue / node_list[
                                1].more_p_stat.index
                        delay_arcades_avg = 0
                        delay_arcades_avg_priority = 0
                        for i in range(2, nodes + 1):
                            if node_list[i].more_p_stat.index != 0:
                                delay_arcades_avg_priority += (
                                        node_list[i].more_p_stat.queue / node_list[i].more_p_stat.index)
                            if node_list[i].less_p_stat.index != 0:
                                delay_arcades_avg += (node_list[i].less_p_stat.queue / node_list[i].less_p_stat.index)
                        delay_arcades_avg = delay_arcades_avg / (nodes - 1.0)
                        delay_arcades_avg_priority = delay_arcades_avg_priority / (nodes - 1.0)
                        act_st["delay_arcades"] = delay_arcades_avg
                        act_st["delay_arcades_priority"] = delay_arcades_avg_priority
                        job_list.append(act_st)

                    if node_to_process.more_p_stat.number > 0:
                        node_to_process.priority_completion = True
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                    elif node_to_process.less_p_stat.number > 0:
                        node_to_process.priority_completion = False
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                    else:
                        node_to_process.completion = INFINITY

                    if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on ARCADE_i
                        if not is_positive():
                            arcade_node = node_list[select_node(True)]  # on first global stats
                            select_queue(arcade_node.id)

                            # Update partial stats for arcade nodes
                            if arcade_node.priority_arrival is True:
                                arcade_node.more_p_stat.number += 1  # system stats don't updated
                                arcade_node.more_p_stat.last = time.current
                            else:
                                arcade_node.less_p_stat.number += 1  # system stats don't updated
                                arcade_node.less_p_stat.last = time.current
                            if arcade_node.more_p_stat.number == 1 and arcade_node.less_p_stat.number == 0:
                                arcade_node.priority_completion = True
                                arcade_node.completion = time.current + get_service(arcade_node.id)
                            elif arcade_node.more_p_stat.number == 0 and arcade_node.less_p_stat.number == 1:
                                arcade_node.priority_completion = False
                                arcade_node.completion = time.current + get_service(arcade_node.id)
                        else:
                            node_list[0].index += 1
                            node_list[0].number -= 1

                arrival_list = [node_list[n].arrival for n in range(1, len(node_list))]
                min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]
            #  Global batch means
            final_avg_delay_ticket_green_pass = 0.0
            final_avg_delay_arcades = 0.0
            final_avg_delay_arcades_priority = 0.0
            final_std_ticket_green_pass = 0.0
            final_std_arcades = 0.0
            final_std_arcades_priority = 0.0
            final_income = 0.0
            final_std_income = 0.0
            final_avg_wait_system = 0.0
            final_std_system = 0.0
            n = 0
            for i in range(4, len(batch_means_info["avg_delay_ticket_green_pass"])):
                n += 1
                #  avg calculation,  std calculation

                final_avg_delay_ticket_green_pass, final_std_ticket_green_pass = online_variance(n,
                                                                                                final_avg_delay_ticket_green_pass,
                                                                                                final_std_ticket_green_pass,
                                                                                                batch_means_info[
                                                                                                    "avg_delay_ticket_green_pass"][
                                                                                                    i])
                final_avg_delay_arcades, final_std_arcades = online_variance(n, final_avg_delay_arcades,
                                                                             final_std_arcades,
                                                                             batch_means_info["avg_delay_arcades"][
                                                                                 i])
                final_avg_delay_arcades_priority, final_std_arcades_priority = online_variance(n,
                                                                                               final_avg_delay_arcades_priority,
                                                                                               final_std_arcades_priority,
                                                                                               batch_means_info[
                                                                                                   "avg_delay_arcades_priority"][
                                                                                                   i])
                final_income, final_std_income = online_variance(n, final_income, final_std_income,
                                                                 batch_means_info["income"][i])
                final_avg_wait_system, final_std_system = online_variance(n, final_avg_wait_system,
                                                                          final_std_system,
                                                                          batch_means_info["avg_wait_system"][i])

            final_std_ticket_green_pass = statistics.variance(batch_means_info["avg_delay_ticket_green_pass"][4:])
            final_std_arcades = statistics.variance(batch_means_info["avg_delay_arcades"][4:])
            final_std_arcades_priority = statistics.variance(batch_means_info["avg_delay_arcades_priority"][4:])
            final_std_income = statistics.variance(batch_means_info["income"][4:])
            final_std_system = statistics.variance(batch_means_info["avg_wait_system"][4:])

            final_std_ticket_green_pass = sqrt(final_std_ticket_green_pass)
            final_std_arcades = sqrt(final_std_arcades)
            final_std_arcades_priority = sqrt(final_std_arcades_priority)
            final_std_income = sqrt(final_std_income)
            final_std_system = sqrt(final_std_system)
            #  calculate interval width
            LOC = 0.95
            u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
            t = idfStudent(n - 1, u)  # critical value of t
            final_w_ticket_green_pass = t * final_std_ticket_green_pass / sqrt(n - 1)  # interval half width
            final_w_arcades = t * final_std_arcades / sqrt(n - 1)  # interval half width
            final_w_arcades_priority = t * final_std_arcades_priority / sqrt(n - 1)  # interval half width
            final_w_income = t * final_std_income / sqrt(n - 1)  # interval half width
            final_w_system = t * final_std_system / sqrt(n - 1)  # interval half width
            batch_means_info["final_delay_ticket_green_pass"] = final_avg_delay_ticket_green_pass
            batch_means_info["final_delay_arcades"] = final_avg_delay_arcades
            batch_means_info["final_delay_arcades_priority"] = final_avg_delay_arcades_priority
            batch_means_info["final_std_ticket_green_pass"] = final_std_ticket_green_pass
            batch_means_info["final_std_arcades"] = final_std_arcades
            batch_means_info["final_std_arcades_priority"] = final_std_arcades_priority
            batch_means_info["final_w_ticket_green_pass"] = final_w_ticket_green_pass
            batch_means_info["final_w_arcades"] = final_w_arcades
            batch_means_info["final_w_arcades_priority"] = final_w_arcades_priority

            batch_means_info["final_income"] = final_income
            batch_means_info["final_std_income"] = final_std_income
            batch_means_info["final_w_income"] = final_w_income

            batch_means_info["final_wait_system"] = final_avg_wait_system
            batch_means_info["final_std_system"] = final_std_system
            batch_means_info["final_w_system"] = final_w_system
            for i in range(0, 10):
                batch_means_info["correlation_delay_arcades"].append(
                    pearsonr(batch_means_info["avg_wait_system"][:k - i],
                             batch_means_info["avg_wait_system"][i:])[0])

            dict_list.append(batch_means_info)
            path = "stats_" + str(seed) + ".json"
            with open(path, 'w+') as json_file:
                json.dump(batch_means_info, json_file, indent=4)
            json_file.close()

        plot_stats_global()
        plot_stats_global_ticket()

        # plot_stats_global()
        plot_stats_global_ticket()
        # plot_correlation()
        avg_seed_income = 0.0
        avg_wait_system = 0.0
        for i in range(0, len(dict_list)):
            avg_wait_system += dict_list[i]["final_wait_system"]
            avg_seed_income += dict_list[i]["final_income"]
        avg_seed_income = avg_seed_income / len(dict_list)
        avg_wait_system = avg_wait_system / len(dict_list)
        income_list.append((avg_seed_income, nodes - 1, avg_wait_system))
    # plot_income()