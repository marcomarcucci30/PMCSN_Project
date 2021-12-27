import json
from math import sqrt

from matplotlib import pyplot as plt


from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from utils.rvms import idfStudent
from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream
from advanced_model.skeleton import select_queue_premium

nodes = 3 # n nodi
arrival_time = 15.0
arrival_time_morning = 15.0
arrival_time_afternoon = 15.0
arrival_time_evening = 15.0
arrival_time_night = 15.0

b = 128
k = 128

# seed = 1234567891
START = 8.0 * 1440
STOP = 1000 * 12 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
p_size = 0.6
p_premium = 0.36
p_positive = 0.05


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service
    index = 0.0  # jobs departed
    number = 0.0  # jobs in node
    last = 0.0

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0
        self.index = 0.0  # jobs departed
        self.number = 0.0  # jobs in node
        self.last = 0.0


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    priority_arrival = False
    priority_completion = False
    last = 0.0  # last arrival time # TODO: DA TOGLIERE?
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
    number = 0.0  # jobs in node
    stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.stat = SystemTrack()  # Track stats


select_node_positive = 75


def is_positive():
    selectStream(select_node_positive)
    r = random()
    if r <= p_positive:
        return True
    else:
        return False


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


seeds = [987654321, 539458255, 482548808]
replicas = 10
sampling_frequency = 50
dict_list = []


def online_variance(n, mean, variance, x):
    delta = x - mean
    variance = variance + delta * delta * (n - 1) / n
    mean = mean + delta / n
    return mean, variance


def plot_stats_global():
    x = [i * sampling_frequency for i in range(0, len(dict_list[0]["avg_wait_system"]))]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    plt.xticks(rotation=45)
    plt.rcParams["figure.figsize"] = (16, 9)

    for i in range(0, len(dict_list)):
        # prova = [dict_list[i]["job_list"][j]["delay_arcades"] for j in range(0, len(dict_list[i]["job_list"]), 10)]
        # print(dict_list[i])
        plt.plot(x, [dict_list[i]["avg_wait_system"][j] for j in range(0, len(dict_list[i]["avg_wait_system"]))],
                 'o',
                 color=colors[i], label=dict_list[i]["seed"], mfc='none')

    plt.xlabel("Number of jobs")
    plt.ylabel("Avg wait system")
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
            "avg_wait_ticket_green_pass": [],  # [elem 0-50, elem 50-100, ..]
            "std_ticket_green_pass": [],
            "w_ticket_green_pass": [],
            "avg_delay_arcades": [],
            "std_arcades": [],
            "w_arcades": [],
            "avg_delay_arcades_priority": [],
            "std_arcades_priority": [],
            "w_arcades_priority": [],
            "avg_wait_system": [],
            "std_system": [],
            "w_system": [],

            "final_wait_ticket_green_pass": 0.0,
            "final_std_ticket_green_pass": 0.0,
            "final_w_ticket_green_pass": 0.0,
            "final_delay_arcades": 0.0,
            "final_std_arcades": 0.0,
            "final_w_arcades": 0.0,
            "final_delay_arcades_priority": 0.0,
            "final_std_arcades_priority": 0.0,
            "final_w_arcades_priority": 0.0,
            "correlation_delay_arcades": 0.0
        }

        batch_means_info = batch_means_info_struct
        batch_means_info["seed"] = seed
        batch_means_info["b"] = b
        batch_means_info["k"] = k
        batch_means_info["n_nodes"] = nodes - 1
        batch_means_info["lambda"] = 1.0 / arrival_time
        node_list = []
        for i in range(nodes + 1):

            if i == 0:
                node_list.append(SystemNode(i))
            else:
                node_list.append(StatusNode(i))

        plantSeeds(seed)

        for replica in range(0, replicas):
            job_list = []
            batch_means_info["job_list"] = job_list
            for center in node_list:
                if center.id == 0:
                    center.number = 0.0
                    center.last = 0.0
                    center.arrival = None
                    center.completion = None
                    center.index = 0.0
                    center.stat.node = 0.0
                    center.stat.queue = 0.0
                    center.stat.service = 0.0
                else:
                    center.arrival = None
                    center.completion = None
                    priority_arrival = False
                    priority_completion = False
                    center.more_p_stat.node = 0.0
                    center.more_p_stat.queue = 0.0
                    center.more_p_stat.service = 0.0
                    center.more_p_stat.index = 0.0
                    center.more_p_stat.number = 0.0
                    center.more_p_stat.last = 0.0
                    center.less_p_stat.node = 0.0
                    center.less_p_stat.queue = 0.0
                    center.less_p_stat.service = 0.0
                    center.less_p_stat.index = 0.0
                    center.less_p_stat.number = 0.0
                    center.less_p_stat.last = 0.0

            batch_index = 0
            time.current = START
            arrival = START  # global temp var for getArrival function     [minutes]

            # initialization of the first arrival event
            set_arrival_time(arrival_time_night)
            arrival += get_arrival(arrival_time)
            node = node_list[select_node(False)]  # node in cui schedulare l'arrivo
            select_queue(node.id)  # discriminazione della coda relativa all'arrivo
            node.arrival = arrival
            min_arrival = arrival
            old_index = 0

            while node_list[0].index <= b * k:

                if node_list[0].index % sampling_frequency == 0 and node_list[0].index != 0 and old_index != node_list[0].index:
                    old_index = node_list[0].index
                    old_index_arcades = 0
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            old_index_arcades += center.more_p_stat.index + center.less_p_stat.index
                    old_index_arcades = int(old_index_arcades)
                    if replica == 0:
                        batch_means_info["avg_wait_ticket_green_pass"].append(
                            job_list[old_index_arcades-1]["wait_ticket_green_pass"])
                        batch_means_info["std_ticket_green_pass"].append(0.0)

                        batch_means_info["avg_delay_arcades"].append(
                            job_list[old_index_arcades-1]["delay_arcades"])
                        batch_means_info["std_arcades"].append(0.0)

                        batch_means_info["avg_delay_arcades_priority"].append(
                            job_list[old_index_arcades-1]["delay_arcades_priority"])
                        batch_means_info["std_arcades_priority"].append(0.0)

                        batch_means_info["avg_wait_system"].append(
                            job_list[old_index_arcades-1]["wait_system"])
                        batch_means_info["std_system"].append(0.0)
                    else:
                        batch_means_info["avg_wait_ticket_green_pass"][batch_index], batch_means_info["std_ticket_green_pass"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_wait_ticket_green_pass"][batch_index],
                                                           batch_means_info["std_ticket_green_pass"][batch_index],
                                                           job_list[old_index_arcades-1]["wait_ticket_green_pass"])

                        batch_means_info["avg_delay_arcades"][batch_index], batch_means_info["std_arcades"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_delay_arcades"][batch_index],
                                                           batch_means_info["std_arcades"][batch_index],
                                                           job_list[old_index_arcades-1]["delay_arcades"])

                        batch_means_info["avg_delay_arcades_priority"][batch_index], batch_means_info["std_arcades_priority"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_delay_arcades_priority"][batch_index],
                                                           batch_means_info["std_arcades_priority"][batch_index],
                                                           job_list[old_index_arcades-1]["delay_arcades_priority"])



                        batch_means_info["avg_wait_system"][batch_index], batch_means_info["std_system"][
                            batch_index] = online_variance(replica + 1,
                                                           batch_means_info["avg_wait_system"][batch_index],
                                                           batch_means_info["std_system"][batch_index],
                                                           job_list[old_index_arcades-1]["wait_system"])
                    batch_index += 1


                node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
                time.next = minimum(node_to_process.arrival, node_to_process.completion)
                # Aggiornamento delle aree basate sul giro prima
                for i in range(0, len(node_list)):
                    if i != 0:
                        if node_list[i].priority_completion == True:
                            if node_list[i].more_p_stat.number > 0:
                                node_list[i].more_p_stat.node += (time.next - time.current) * node_list[i].more_p_stat.number
                                node_list[i].more_p_stat.queue += (time.next - time.current) * (node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.service += (time.next - time.current)
                            if node_list[i].less_p_stat.number > 0:
                                node_list[i].less_p_stat.node += (time.next - time.current) * (node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.queue += (time.next - time.current) * (node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.service += (time.next - time.current)
                        else:
                            if node_list[i].more_p_stat.number > 0:
                                node_list[i].more_p_stat.node += (time.next - time.current) * (node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.queue += (time.next - time.current) * (node_list[i].more_p_stat.number - 1)
                                node_list[i].more_p_stat.service += (time.next - time.current)
                            if node_list[i].less_p_stat.number > 0:
                                node_list[i].less_p_stat.node += (time.next - time.current) * (node_list[i].less_p_stat.number)
                                node_list[i].less_p_stat.queue += (time.next - time.current) * (node_list[i].less_p_stat.number - 1)
                                node_list[i].less_p_stat.service += (time.next - time.current)
                    else:
                        if node_list[i].number > 0:
                            node_list[i].stat.node += (time.next - time.current) * node_list[i].number
                            node_list[i].stat.queue += (time.next - time.current) * (node_list[i].number - 1)
                            node_list[i].stat.service += (time.next - time.current)

                current_for_update = time.current
                time.current = time.next  # advance the clock

                if time.current == node_to_process.arrival:

                    if node_to_process.priority_arrival is True:  # vediamo su quale coda è stato schedulato l'arrivo che stiamo processando
                        node_to_process.more_p_stat.number += 1
                    else:
                        node_to_process.less_p_stat.number += 1

                    node_list[0].number += 1  # update system stat
                    arrival += get_arrival(arrival_time)
                    node_selected_pos = select_node(False)
                    select_queue(node_selected_pos)  # discriminazione della coda relativa all'arrivo
                    # Se il prossimo arrivo è su un altro centro, bisogna eliminare l'arrivo sul centro processato altrimenti
                    # sarà sempre il minimo
                    if node_selected_pos != node_to_process.id:
                        node_to_process.arrival = INFINITY
                    node = node_list[node_selected_pos]

                    if node.arrival != INFINITY:
                        if node.priority_arrival is True:
                            node.more_p_stat.last = node.arrival
                        else:
                            node.less_p_stat.last = node.arrival

                        max_last = maximum(node.more_p_stat.last, node.less_p_stat.last)
                        if node.more_p_stat.last is not None and node_list[0].last is not None and node_list[0].last < max_last:
                            node_list[0].last = max_last

                        # node.last = node.arrival
                        '''if node.last is not None and node_list[0].last is not None and node_list[0].last < node.last:
                            node_list[0].last = node.last'''
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
                            # node.last = node.arrival
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
                            "wait_ticket_green_pass": 0.0,
                            "delay_arcades": 0.0,
                            "delay_arcades_priority": 0.0,
                            "wait_system": 0.0
                        }
                        act_st = actual_stats
                        if node_list[0].index != 0:
                            act_st["wait_system"] = node_list[0].stat.node / node_list[0].index
                        if node_list[1].more_p_stat.index != 0:
                            act_st["wait_ticket_green_pass"] = node_list[1].more_p_stat.node / node_list[1].more_p_stat.index
                        delay_arcades_avg = 0.0
                        delay_arcades_avg_priority = 0.0
                        for i in range(2, nodes + 1):
                            if node_list[i].more_p_stat.index != 0:
                                delay_arcades_avg_priority += (node_list[i].more_p_stat.queue / node_list[i].more_p_stat.index)
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
                            # if arcade_node.number > 0:
                            #     arcade_node.stat.node += (time.next - current_for_update) * arcade_node.number
                            #     arcade_node.stat.queue += (time.next - current_for_update) * (arcade_node.number - 1)
                            #     arcade_node.stat.service += (time.next - current_for_update)
                            if arcade_node.priority_arrival is True:
                                arcade_node.more_p_stat.number += 1  # system stats don't updated
                                arcade_node.more_p_stat.last = time.current
                            else:
                                arcade_node.less_p_stat.number += 1  # system stats don't updated
                                arcade_node.less_p_stat.last = time.current

                            # arcade_node.last = time.current

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

        dict_list.append(batch_means_info)
        for i in range(0, len(batch_means_info["std_arcades"])):
            batch_means_info["std_arcades"][i] = sqrt(batch_means_info["std_arcades"][i] / replicas)
            batch_means_info["std_arcades_priority"][i] = sqrt(batch_means_info["std_arcades_priority"][i] / replicas)
            batch_means_info["std_ticket_green_pass"][i] = sqrt(batch_means_info["std_ticket_green_pass"][i] / replicas)
            batch_means_info["std_system"][i] = sqrt(batch_means_info["std_system"][i] / replicas)
            if replicas > 1:
                LOC = 0.95
                u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
                t = idfStudent(replicas - 1, u)  # critical value of t
                w_ticket_green_pass = t * batch_means_info["std_ticket_green_pass"][i] / sqrt(replicas - 1)
                w_arcades = t * batch_means_info["std_arcades"][i] / sqrt(replicas - 1)  # interval half width
                w_arcades_priority = t * batch_means_info["std_arcades_priority"][i] / sqrt(replicas - 1)  # interval half width
                w_system = t * batch_means_info["std_system"][i] / sqrt(replicas - 1)  # interval half width
                batch_means_info["w_ticket_green_pass"].append(w_ticket_green_pass)
                batch_means_info["w_arcades"].append(w_arcades)
                batch_means_info["w_arcades_priority"].append(w_arcades_priority)
                batch_means_info["w_system"].append(w_system)

        path = "stats_" + str(seed) + ".json"
        with open(path, 'w+') as json_file:
            json.dump(batch_means_info, json_file, indent=4)
        json_file.close()

    plot_stats_global()