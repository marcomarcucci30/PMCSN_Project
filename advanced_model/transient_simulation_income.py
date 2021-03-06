import json
import os
from matplotlib import pyplot as plt

from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream
from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal

stationary = True
nodes = 2  # n nodi
arrival_time = 14.0
arrival_time_morning = 14.0  # nodes = 3 min
arrival_time_afternoon = 5.0  # nodes = 4 min
arrival_time_evening = 14.0
arrival_time_night = 35.0  # nodes = 2 min

seeds = [987654321, 539458255, 482548808]
replicas = 64
sampling_frequency = 10

n_mor_sampl = 4 * 60 / sampling_frequency
n_aft_sampl = n_mor_sampl + 5 * 60 / sampling_frequency
n_eve_sampl = n_aft_sampl + 5 * 60 / sampling_frequency
n_night_sampl = n_eve_sampl + 10 * 60 / sampling_frequency

b = 256
k = 1
START = 8.0 * 60
STOP = 1 * 1 * 1 * 1440.0 + 8.0 * 60  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
p_size = 0.6
p_positive = 0.05
p_premium = 0.36

ticket_price = 10.0
ticket_price_premium = 20.0

delay_max = 20.0
delay_min = 8.0

best_conf = 2
if best_conf == 0:
    n1 = n3 = 6
    n2 = 13
    n4 = 3

elif best_conf == 1:
    n1 = n3 = 3
    n2 = 4
    n4 = 2
else:
    n1 = n3 = 20
    n2 = 20
    n4 = 20

energy_cost = 0.4
ec_mor = (n1-1) * energy_cost
ec_aft = (n2-1) * energy_cost
ec_night = (n4-1) * energy_cost

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
    last_completion = False


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


def set_arrival_time(x):
    global arrival_time
    arrival_time = x


def get_arrival_time():
    return arrival_time


arr_est = 0


def ticket_refund(avg_delay_arcades):
    perc = (avg_delay_arcades - delay_min) / (delay_max - delay_min)
    if perc > 0.8:
        return 0.8
    else:
        return perc


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


select_queue_premium = 35
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
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=400)  # permette di generare sottografici in un grafico
    plt.setp(axs[1].xaxis.get_majorticklabels())
    x = dict_list[0]["time_current"]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    axs[1].set_ylabel(ylabel="Income ???", fontsize=15)
    axs[0].tick_params(labelsize=10)
    axs[1].tick_params(labelsize=10)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    axs[1].set_ylim([0, 2500])
    axs[0].set_ylim([0, 55])
    axs[0].set_ylabel(ylabel="Average wait system (minutes)", fontsize=15)
    axs[1].set_xlabel(xlabel="Minutes", fontsize=15)
    axs[0].vlines(480, 0, 55, color='lawngreen', label="")
    axs[1].vlines(480, 0, 2500, color='lawngreen', label="")
    axs[0].vlines(720, 0, 55, color='blue', label="")
    axs[1].vlines(720, 0, 2500, color='blue', label="")
    axs[0].vlines(1020, 0, 55, color='red', label="")
    axs[1].vlines(1020, 0, 2500, color='red', label="")
    axs[0].vlines(1320, 0, 55, color='orange', label="")
    axs[1].vlines(1320, 0, 2500, color='orange', label="")
    axs[0].legend(["08:00", "12:00", "17:00", "22:00"])
    for i in range(0, len(dict_list)):
        axs[0].plot(x, [dict_list[i]["avg_wait_system"][j] for j in range(0, len(dict_list[i]["avg_wait_system"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none')
    for i in range(0, len(dict_list)):
        axs[1].plot(x, [dict_list[i]["income"][j] for j in range(0, len(dict_list[i]["income"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none')
    axs[1].legend(["seed = " + str(dict_list[0]["seed"]), "seed = " + str(dict_list[1]["seed"]),
                "seed = " + str(dict_list[2]["seed"])])

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


def redirect_jobs(prev_nodes):
    # TUTTO SHIFTATO DI 2 PERCH?? LA POSIZIONE 1 E 2 DELLA LISTA CI SONO
    # IL SISTEMA E LA CODA DEI TAMPONI

    if prev_nodes == nodes:
        return
    if prev_nodes < nodes:  # 2-->3
        iteration = nodes - prev_nodes
        for i in range(2, prev_nodes + 1):
            if node_list[i].more_p_stat.number > 1:
                n_jobs_priority = node_list[i].more_p_stat.number
                for jobs in range(0, int(n_jobs_priority) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    node_list[pos].more_p_stat.number += 1
                    if node_list[pos].more_p_stat.number == 1 and node_list[pos].less_p_stat.number == 0:
                        node_list[pos].priority_completion = True
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    # aggiorno le stats della coda da spegnere
                    node_list[i].more_p_stat.number -= 1
            if node_list[i].less_p_stat.number > 1:
                n_jobs = node_list[i].less_p_stat.number
                for jobs in range(0, int(n_jobs) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    node_list[pos].less_p_stat.number += 1
                    if node_list[pos].more_p_stat.number == 0 and node_list[pos].less_p_stat.number == 1:
                        node_list[pos].priority_completion = False
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    node_list[i].less_p_stat.number -= 1
        return

    if prev_nodes > nodes:
        iteration = prev_nodes - nodes  # 3-->2
        for i in range(nodes + 1, nodes + 1 + iteration):
            if node_list[i].more_p_stat.number > 1:
                n_jobs_priority = node_list[i].more_p_stat.number
                for jobs in range(0, int(n_jobs_priority) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    node_list[pos].more_p_stat.number += 1
                    if node_list[pos].more_p_stat.number == 1 and node_list[pos].less_p_stat.number == 0:
                        node_list[pos].priority_completion = True
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    # aggiorno le stats della coda da spegnere
                    node_list[i].more_p_stat.number -= 1
            if node_list[i].less_p_stat.number > 1:
                n_jobs = node_list[i].less_p_stat.number
                for jobs in range(0, int(n_jobs) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    node_list[pos].less_p_stat.number += 1
                    if node_list[pos].more_p_stat.number == 0 and node_list[pos].less_p_stat.number == 1:
                        node_list[pos].priority_completion = False
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    node_list[i].less_p_stat.number -= 1

            if node_list[i].arrival != INFINITY and node_list[i].arrival is not None:  # per non perdere l'arrivo su
                # un nodo spento
                pos = select_node(True)
                node_list[pos].arrival = node_list[i].arrival
                if node_list[i].priority_arrival is True:  # vediamo su quale coda ?? stato schedulato l'arrivo che
                    # stiamo processando
                    node_list[pos].priority_arrival = True
                else:
                    node_list[pos].priority_arrival = False

            node_list[i].arrival = None

            if (node_list[i].more_p_stat.number == 1 and node_list[i].less_p_stat.number == 0) \
                    or (node_list[i].more_p_stat.number == 0 and node_list[i].less_p_stat.number == 1):
                node_list[i].last_completion = True
        return


def clear_statistics(i):
    node_list[i].arrival = None
    node_list[i].completion = None
    node_list[i].priority_arrival = False
    node_list[i].priority_completion = False
    node_list[i].last_completion = False
    node_list[i].more_p_stat.index_support += node_list[i].more_p_stat.index
    node_list[i].more_p_stat.index_support += node_list[i].more_p_stat.index
    node_list[i].more_p_stat.node = 0.0
    node_list[i].more_p_stat.queue = 0.0
    node_list[i].more_p_stat.service = 0.0
    node_list[i].more_p_stat.index = 0.0
    node_list[i].more_p_stat.number = 0.0
    node_list[i].more_p_stat.last = 0.0
    node_list[i].less_p_stat.node = 0.0
    node_list[i].less_p_stat.queue = 0.0
    node_list[i].less_p_stat.service = 0.0
    node_list[i].less_p_stat.index = 0.0
    node_list[i].less_p_stat.number = 0.0
    node_list[i].less_p_stat.last = 0.0


if __name__ == '__main__':
    for seed in seeds:

        # settings

        batch_means_info_struct = {
            "seed": 0,
            "n_nodes": 0,
            "lambda": 0.0,
            "b": 0,
            "k": 0,
            "income": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "time_current": [i * sampling_frequency + START for i in
                             range(0, int((STOP - START) / sampling_frequency) + 1)],
            "job_list": [None] * (int((STOP - START) / sampling_frequency) + 1),

            "avg_wait_ticket_green_pass": [None] * (int((STOP - START) / sampling_frequency) + 1),  # [elem 0-50, elem 50-100, ..]
            "std_ticket_green_pass": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_ticket_green_pass": [None] * (int((STOP - START) / sampling_frequency) + 1),

            "avg_delay_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "std_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),

            "avg_delay_arcades_priority": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "std_arcades_priority": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_arcades_priority": [None] * (int((STOP - START) / sampling_frequency) + 1),

            "avg_wait_system": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "std_system": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_system": [None] * (int((STOP - START) / sampling_frequency) + 1),

            "final_wait_ticket": 0.0,
            "final_std_ticket": 0.0,
            "final_w_ticket": 0.0,

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
        # (batch_means_info)
        node_list = []
        for i in range(max(n1, n2, n3, n4) + 1):
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
                    center.priority_arrival = False
                    center.priority_completion = False
                    center.more_p_stat.index_support = 0.0
                    center.less_p_stat.index_support = 0.0
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
            sampling = START  # init sampling
            sampling_count = 0
            # initialization of the first arrival event
            set_arrival_time(arrival_time_morning)
            nodes = n1
            arrival += get_arrival(arrival_time)
            node = node_list[select_node(False)]
            select_queue(node.id)
            node.arrival = arrival
            min_arrival = arrival
            old_index = 0
            count_index = 0
            count_arrival = 0

            while min_arrival < STOP:

                if time.current - sampling > sampling_frequency:
                    step = int((time.current - sampling) / sampling_frequency)

                    ''' L'energy cost viene sottratto alla fine dato che conosciamo perfettamente quanto spenderemo
                    di energia elettrica per ogni intervallo temporale'''

                    index_more_p_arcades = 0  # all completed jobs from premium
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            index_more_p_arcades += center.more_p_stat.index + center.more_p_stat.index_support

                    index_less_p_arcades = 0  # all completed jobs from standard
                    for center in node_list:
                        if center.id > TICKET_QUEUE:
                            index_less_p_arcades += center.less_p_stat.index + center.less_p_stat.index_support

                    # Update income
                    if step > 1:
                        for y in range(sampling_count, sampling_count + step - 1):

                            if batch_means_info["income"][sampling_count - 1] is not None:
                                if batch_means_info["income"][y] is None:
                                    batch_means_info["income"][y] = batch_means_info["income"][sampling_count - 1]
                                else:
                                    batch_means_info["income"][y], ingore_var = \
                                        online_variance(replica+1, batch_means_info["income"][y], 0.0, batch_means_info["income"][sampling_count - 1])



                    if batch_means_info["avg_wait_ticket_green_pass"][sampling_count + step - 1] is None:
                        if len(job_list) == 0:
                            batch_means_info["avg_wait_ticket_green_pass"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_ticket_green_pass"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades_priority"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_arcades_priority"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_wait_system"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_system"][sampling_count + step - 1] = 0.0

                            income = len(job_list) * ticket_price - 0.0

                            batch_means_info["income"][sampling_count + step - 1] = income
                        else:
                            batch_means_info["avg_wait_ticket_green_pass"][sampling_count + step - 1] = job_list[-1]["wait_ticket_green_pass"]

                            batch_means_info["std_ticket_green_pass"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades"][sampling_count + step - 1] = job_list[-1][
                                "delay_arcades"]

                            batch_means_info["std_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades_priority"][sampling_count + step - 1] = job_list[-1][
                                "delay_arcades_priority"]

                            batch_means_info["std_arcades_priority"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_wait_system"][sampling_count + step - 1] = job_list[-1]["wait_system"]

                            batch_means_info["std_system"][sampling_count + step - 1] = 0.0

                            income = index_less_p_arcades * ticket_price + index_more_p_arcades * ticket_price_premium  - \
                                     (index_less_p_arcades * ticket_refund(job_list[-1]["delay_arcades"]) * ticket_price +
                                      index_more_p_arcades * ticket_refund(job_list[-1]["delay_arcades_priority"]) * ticket_price_premium)

                            batch_means_info["income"][sampling_count + step - 1] = income

                    else:
                        if len(job_list) != 0:

                            batch_means_info["avg_wait_ticket_green_pass"][sampling_count + step - 1], \
                            batch_means_info["std_ticket_green_pass"][sampling_count + step - 1] = online_variance(replica + 1,
                                                                                                        batch_means_info[
                                                                                                            "avg_wait_ticket_green_pass"][
                                                                                                            sampling_count + step - 1],
                                                                                                        batch_means_info[
                                                                                                            "std_ticket_green_pass"][
                                                                                                            sampling_count + step - 1],
                                                                                                        job_list[- 1][
                                                                                                            "wait_ticket_green_pass"])
                            batch_means_info["avg_delay_arcades"][sampling_count + step - 1], \
                            batch_means_info["std_arcades"][sampling_count + step - 1] = online_variance(replica + 1,
                                                                                                         batch_means_info[
                                                                                                             "avg_delay_arcades"][
                                                                                                             sampling_count + step - 1],
                                                                                                         batch_means_info[
                                                                                                             "std_arcades"][
                                                                                                             sampling_count + step - 1],
                                                                                                         job_list[- 1][
                                                                                                             "delay_arcades"])

                            batch_means_info["avg_delay_arcades_priority"][sampling_count + step - 1], \
                            batch_means_info["std_arcades_priority"][sampling_count + step - 1] = online_variance(replica + 1,
                                                                                                         batch_means_info[
                                                                                                             "avg_delay_arcades_priority"][
                                                                                                             sampling_count + step - 1],
                                                                                                         batch_means_info[
                                                                                                             "std_arcades_priority"][
                                                                                                             sampling_count + step - 1],
                                                                                                         job_list[- 1][
                                                                                                             "delay_arcades_priority"])


                            batch_means_info["avg_wait_system"][sampling_count + step - 1], \
                            batch_means_info["std_system"][sampling_count + step - 1] = online_variance(replica + 1,
                                                                                                        batch_means_info[
                                                                                                            "avg_wait_system"][
                                                                                                            sampling_count + step - 1],
                                                                                                        batch_means_info[
                                                                                                            "std_system"][
                                                                                                            sampling_count + step - 1],
                                                                                                        job_list[- 1][
                                                                                                            "wait_system"])

                            income = index_less_p_arcades * ticket_price + index_more_p_arcades * ticket_price_premium - \
                                     (index_less_p_arcades * ticket_refund(
                                         job_list[-1]["delay_arcades"]) * ticket_price +
                                      index_more_p_arcades * ticket_refund(
                                                 job_list[-1]["delay_arcades_priority"]) * ticket_price_premium)

                            batch_means_info["income"][sampling_count + step - 1], ignored = online_variance(
                                replica + 1,
                                batch_means_info[
                                    "income"][sampling_count + step - 1],
                                0.0, income)

                    sampling_count += step
                    sampling = START + (sampling_frequency * (sampling_count + 1))

                node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
                time.next = minimum(node_to_process.arrival, node_to_process.completion)
                # Aggiornamento delle aree basate sul giro prima
                for i in range(0, len(node_list)):
                    if i != 0:
                        if node_list[i].priority_completion == True:
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

                day = (time.current / 1440.0) // 1
                current_lambda = time.current - day * 1440.0

                if 480.0 <= current_lambda < 720.0:  # 8-12
                    set_arrival_time(arrival_time_morning)
                    prev_nodes = nodes
                    nodes = n1
                    redirect_jobs(prev_nodes)
                elif 720.0 <= current_lambda < 1020.0:  # 12-17
                    set_arrival_time(arrival_time_afternoon)
                    prev_nodes = nodes
                    nodes = n2
                    redirect_jobs(prev_nodes)
                elif 1020.0 <= current_lambda < 1320.0:  # 17-22
                    set_arrival_time(arrival_time_evening)
                    prev_nodes = nodes
                    nodes = n3
                    redirect_jobs(prev_nodes)
                else:  # 22-8
                    set_arrival_time(arrival_time_night)
                    prev_nodes = nodes
                    nodes = n4
                    redirect_jobs(prev_nodes)

                if time.current == node_to_process.arrival:
                    if node_to_process.priority_arrival is True:  # vediamo su quale coda ?? stato schedulato l'arrivo
                        # che stiamo processando
                        node_to_process.more_p_stat.number += 1
                    else:
                        node_to_process.less_p_stat.number += 1

                    node_list[0].number += 1  # update system stat
                    arrival += get_arrival(arrival_time)
                    node_selected_pos = select_node(False)
                    select_queue(node_selected_pos)  # discriminazione della coda relativa all'arrivo
                    # Se il prossimo arrivo ?? su un altro centro, bisogna eliminare l'arrivo sul centro processato
                    # altrimenti sar?? sempre il minimo
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
                            "wait_ticket_green_pass": 0.0,
                            "delay_arcades": 0.0,
                            "delay_arcades_priority": 0.0,
                            "wait_system": 0.0
                        }
                        act_st = actual_stats
                        if node_list[0].index != 0:
                            act_st["wait_system"] = node_list[0].stat.node / node_list[0].index
                        if node_list[1].more_p_stat.index != 0:
                            act_st["wait_ticket_green_pass"] = node_list[1].more_p_stat.node / node_list[
                                1].more_p_stat.index
                        delay_arcades_avg = 0.0
                        delay_arcades_avg_priority = 0.0
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

                        if node_to_process.last_completion is True:
                            clear_statistics(node_to_process.id)

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


                arrival_list = [node_list[n].arrival for n in range(1, max(n1, n2, n3, n4) + 1)]
                min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]

        for y in range(0, len(batch_means_info["income"])):
            if batch_means_info["income"][y] is None:
                continue
            if 0 <= y <= n_mor_sampl-1:
                batch_means_info["income"][y] -= ec_mor * (y+1)
            if n_mor_sampl <= y <= n_aft_sampl-1:
                batch_means_info["income"][y] -= ec_aft * (y-n_mor_sampl+1) + (ec_mor * n_mor_sampl)
            if n_aft_sampl <= y <= n_eve_sampl-1:
                batch_means_info["income"][y] -= ec_mor * (y-n_aft_sampl+1) + (ec_mor * n_mor_sampl) + (ec_aft * (n_aft_sampl-n_mor_sampl))
            if n_eve_sampl <= y <= n_night_sampl - 1:
                batch_means_info["income"][y] -= ec_night * (y-n_eve_sampl+1) + (ec_mor * (n_eve_sampl-n_aft_sampl)) + (ec_mor * n_mor_sampl) + (ec_aft * (n_aft_sampl-n_mor_sampl))
        dict_list.append(batch_means_info)
        path = "stats_" + str(seed) + ".json"
        with open(path, 'w+') as json_file:
            json.dump(batch_means_info, json_file, indent=4)
        json_file.close()
        print(node_list[0].index)

    plot_stats_global()
