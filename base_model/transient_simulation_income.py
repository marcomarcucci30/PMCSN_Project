import json
import os
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import text

from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream
from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from utils.rvms import idfStudent

stationary = True
nodes = 2  # n nodi
arrival_time = 35.0
arrival_time_morning = 14.0  # 2 arcades for stationary
arrival_time_afternoon = 5.0  # 4 arcades for stationary
arrival_time_evening = 14.0  # 2 arcades for stationary
arrival_time_night = 35.0  # 1 arcades for stationary

seeds = [987654321, 539458255, 482548808]  # , 1865511657, 841744376,
# 430131813, 725267564]# 1757116804, 238927874, 377966758, 306186735,
# 640977820, 893367702, 468482873, 60146203, 258621233, 298382896, 443460125, 250910117, 163127968]
replicas = 64
sampling_frequency = 10

n_mor_sampl = 4 * 60 / sampling_frequency
n_aft_sampl = n_mor_sampl + 5 * 60 / sampling_frequency
n_eve_sampl = n_aft_sampl + 5 * 60 / sampling_frequency
n_night_sampl = n_eve_sampl + 10 * 60 / sampling_frequency

b = 256
k = 1
# seed = 123456789
START = 8.0 * 60
STOP = 1 * 1 * 1 * 1440.0 + 8.0 * 60  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
# ARCADE1 = 2
# ARCADE2 = 3
# ARCADE3 = 4
p_size = 0.6
p_positive = 0.05

ticket_price = 10.0

delay_max = 20.0
delay_min = 8.0

best_conf = 2
if best_conf == 0:
    n1 = n3 = 6
    n2 = 11
    n4 = 4

elif best_conf == 1:
    n1 = n3 = 3
    n2 = 4
    n4 = 2
else:
    n1 = n3 = 20
    n2 = 20
    n4 = 20

energy_cost = 0.4 #30000*110/1024/28/24/6
ec_mor = (n1-1) * energy_cost  # TODO: farlo giornaliero invece che mensile
ec_aft = (n2-1) * energy_cost
ec_night = (n4-1) * energy_cost
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
    last_completion = False

    def __init__(self, id_node):
        self.id = id_node
        self.stat = Track()  # Track stats


def set_arrival_time(x):
    global arrival_time
    arrival_time = x


def get_arrival_time():
    return arrival_time


arr_est = 0


def ticket_refund(avg_delay_arcades):
    '''if avg_delay_arcades < 10.0:
        return 0.0
    elif 20.0 > avg_delay_arcades >= 10.0:
        return 0.25
    else:
        return 0.50'''
    perc = (avg_delay_arcades - delay_min) / (delay_max - delay_min)  # TODO: non rimborsare il biglietto al 100% ?
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
            # print("Green pass: ", service)
            return service
        else:
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            # print("covid test: ", service)
            return service
    else:
        selectStream(id_node + select_node_arcades)
        service = TruncatedNormal(15, 3, 10, 20)  # arcade game time
        # service = BoundedPareto()
        # print("arcade game time: ", service)
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
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), dpi=400)  # permette di generare sottografici in un grafico
    plt.setp(axs[1].xaxis.get_majorticklabels())
    x = dict_list[0]["time_current"]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    #plt.xticks(rotation=45)
    #fig1 = plt.figure(figsize=(16,9), dpi=400)
    axs[1].set_ylabel(ylabel="Income €", fontsize=15)
    """plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels"""
    #axs[0].set(ylabel="Income (€)")
    #axs[0].ylabel("Income (€)")
    axs[0].tick_params(labelsize=10)
    axs[1].tick_params(labelsize=10)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    # plt.savefig(fname=results_dir+"/income_mor", bbox_inches='tight')
    #plt.show()
    #fig2 = plt.figure(figsize=(16, 9), dpi=400)
    axs[1].set_ylim([0, 1600])
    axs[0].set_ylim([0, 60])
    axs[0].set_ylabel(ylabel="Average wait system (minutes)", fontsize=15)
    axs[1].set_xlabel(xlabel="Minutes", fontsize=15)
    axs[0].vlines(480, 0, 60, color='lawngreen', label="")
    axs[1].vlines(480, 0, 1600, color='lawngreen', label="")
    axs[0].vlines(720, 0, 60, color='blue', label="")
    axs[1].vlines(720, 0, 1600, color='blue', label="")
    axs[0].vlines(1020, 0, 60, color='red', label="")
    axs[1].vlines(1020, 0, 1600, color='red', label="")
    axs[0].vlines(1320, 0, 60, color='orange', label="")
    axs[1].vlines(1320, 0, 1600, color='orange', label="")
    axs[0].legend(["08:00", "12:00", "17:00", "22:00"])
    text(720, 5, "entry" , rotation=90, verticalalignment='center')
    for i in range(0, len(dict_list)):
        # prova = [dict_list[i]["job_list"][j]["delay_arcades"] for j in range(0, len(dict_list[i]["job_list"]), 10)]
        # print(dict_list[i])
        axs[0].plot(x, [dict_list[i]["avg_wait_system"][j] for j in range(0, len(dict_list[i]["avg_wait_system"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none')
    for i in range(0, len(dict_list)):
        # prova = [dict_list[i]["job_list"][j]["delay_arcades"] for j in range(0, len(dict_list[i]["job_list"]), 10)]
        # print(dict_list[i])
        axs[1].plot(x, [dict_list[i]["income"][j] for j in range(0, len(dict_list[i]["income"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none')
    # plt.savefig(fname=results_dir + "/avg_wait_sys_night", bbox_inches='tight')
    axs[1].legend(["seed = " + str(dict_list[0]["seed"]), "seed = " + str(dict_list[1]["seed"]),
                "seed = " + str(dict_list[2]["seed"])])

    plt.show()

'''def plot_stats_global():
    x = dict_list[0]["time_current"]
    colors = ['red', 'royalblue', 'green', 'lawngreen', 'lightseagreen', 'orange',
              'blueviolet']
    fig1 = plt.figure(figsize=(16, 9), dpi=400)
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels

    for i in range(0, len(dict_list)):
        # prova = [dict_list[i]["job_list"][j]["delay_arcades"] for j in range(0, len(dict_list[i]["job_list"]), 10)]
        # print(dict_list[i])
        plt.plot(x, [dict_list[i]["avg_wait_system"][j] for j in range(0, len(dict_list[i]["avg_wait_system"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none', figure=fig1)

    plt.legend(["seed = " + str(dict_list[0]["seed"]), "seed = " + str(dict_list[1]["seed"]),
                "seed = " + str(dict_list[2]["seed"])])
    # plt.title("Average Wait System\n08:00-12:00")

    plt.xlabel("Number of jobs")
    plt.ylabel("Avg wait system (minutes)")
    plt.ylim([15, 100])
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '../report/images')
    """if stationary:
        plt.savefig(fname=results_dir + "/transient_night_s", bbox_inches='tight')
    else:
        plt.savefig(fname=results_dir + "/transient_night_ns", bbox_inches='tight')"""
    plt.show()
    fig2 = plt.figure(figsize=(16, 9), dpi=400)
    for i in range(0, len(dict_list)):
        # prova = [dict_list[i]["job_list"][j]["delay_arcades"] for j in range(0, len(dict_list[i]["job_list"]), 10)]
        # print(dict_list[i])
        plt.plot(x, [dict_list[i]["income"][j] for j in range(0, len(dict_list[i]["income"]))],
                 'o', color=colors[i], label=dict_list[i]["seed"], mfc='none', figure=fig2)
    plt.show()'''


'''def plot_stats_global():
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

    plt.show()'''


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


def redirect_jobs(prev_nodes):
    # TUTTO SHIFTATO DI 2 PERCHè LA POSIZIONE 1 E 2 DELLA LISTA CI SONO
    # IL SISTEMA E LA CODA DEI TAMPONI

    if prev_nodes == nodes:
        return
    if prev_nodes < nodes:  # 2-->3
        '''print(node_list[0].number, node_list[1].number, node_list[2].number, node_list[3].number,
              node_list[4].number, ", pre->")'''

        iteration = nodes - prev_nodes
        for i in range(2, prev_nodes + 1):
            if node_list[i].number > 1:
                n_jobs = node_list[i].number
                for jobs in range(0, int(n_jobs) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    # print(pos, nodes, prev_nodes)
                    node_list[pos].number += 1
                    if node_list[pos].number == 1:
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    # aggiorno le stats della coda da spegnere
                    node_list[i].number -= 1
        '''print(node_list[0].number, node_list[1].number, node_list[2].number, node_list[3].number,
              node_list[4].number, ", post->")
        print("ciao")'''

        return

    if prev_nodes > nodes:
        iteration = prev_nodes - nodes  # 3-->2
        '''print(node_list[0].number, node_list[1].number, node_list[2].number, node_list[3].number,
              node_list[4].number, ", pre<-")'''
        for i in range(nodes + 1, nodes + 1 + iteration):
            if node_list[i].number > 1:
                n_jobs = node_list[i].number
                for jobs in range(0, int(n_jobs) - 1):
                    pos = select_node(True)
                    # aggiorno le stats della nuova coda
                    node_list[pos].number += 1
                    if node_list[pos].number == 1:
                        node_list[pos].completion = time.current + get_service(node_list[pos].id)
                    # aggiorno le stats della coda da spegnere
                    node_list[i].number -= 1
            if node_list[i].arrival != INFINITY and node_list[
                i].arrival is not None:  # per non perdere l'arrivo su un nodo spento
                pos = select_node(True)
                node_list[pos].arrival = node_list[i].arrival

            node_list[i].arrival = None

            if node_list[i].number == 1:
                node_list[i].last_completion = True


        '''print(node_list[0].number, node_list[1].number, node_list[2].number, node_list[3].number,
              node_list[4].number, ", post<-")
        print("ciao")'''

        return


def clear_statistics(i):
    node_list[i].number = 0.0  # azzeramento stats nodo spento
    node_list[i].last = 0.0
    node_list[i].completion = None
    node_list[i].index = 0.0
    node_list[i].stat.node = 0.0
    node_list[i].stat.queue = 0.0
    node_list[i].stat.service = 0.0
    node_list[i].last_completion = False


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
            "avg_wait_ticket": [None] * (int((STOP - START) / sampling_frequency) + 1),  # [elem 0-50, elem 50-100, ..]
            "std_ticket": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_ticket": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "avg_delay_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "std_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_arcades": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "avg_wait_system": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "std_system": [None] * (int((STOP - START) / sampling_frequency) + 1),
            "w_system": [None] * (int((STOP - START) / sampling_frequency) + 1),
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
        # (batch_means_info)
        node_list = [StatusNode(i) for i in range(max(n1, n2, n3, n4) + 1)]  # in 0 global stats TODO: +1?

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
            sampling = START  # init sampling
            sampling_count = 0
            # initialization of the first arrival event
            set_arrival_time(arrival_time_morning)
            nodes = n1
            arrival += get_arrival(arrival_time)
            node = node_list[select_node(False)]
            node.arrival = arrival
            min_arrival = arrival
            old_index = 0

            while min_arrival < STOP:  # (node_list[0].number > 0)

                # print(time.current)
                if time.current - sampling > sampling_frequency:
                    step = int((time.current - sampling) / sampling_frequency)

                    ''' L'energy cost viene sottratto alla fine dato che conosciamo perfettamente quanto spenderemo
                    di energia elettrica per ogni intervallo temporale'''

                    # Update income
                    if step > 1:
                        for y in range(sampling_count, sampling_count + step - 1):

                            if batch_means_info["income"][sampling_count - 1] is not None:
                                if batch_means_info["income"][y] is None:
                                    batch_means_info["income"][y] = batch_means_info["income"][sampling_count - 1]
                                else:
                                    batch_means_info["income"][y], ingore_var = online_variance(replica+1, batch_means_info["income"][y], 0.0, batch_means_info["income"][sampling_count - 1])



                    if batch_means_info["avg_wait_ticket"][sampling_count + step - 1] is None:
                        if len(job_list) == 0:
                            batch_means_info["avg_wait_ticket"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_ticket"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_wait_system"][sampling_count + step - 1] = 0.0

                            batch_means_info["std_system"][sampling_count + step - 1] = 0.0

                            income = len(job_list) * ticket_price - 0.0

                            batch_means_info["income"][sampling_count + step - 1] = income
                        else:
                            batch_means_info["avg_wait_ticket"][sampling_count + step - 1] = job_list[-1]["wait_ticket"]

                            batch_means_info["std_ticket"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_delay_arcades"][sampling_count + step - 1] = job_list[-1][
                                "delay_arcades"]

                            batch_means_info["std_arcades"][sampling_count + step - 1] = 0.0

                            batch_means_info["avg_wait_system"][sampling_count + step - 1] = job_list[-1]["wait_system"]

                            batch_means_info["std_system"][sampling_count + step - 1] = 0.0

                            income = len(job_list) * ticket_price - len(job_list) * ticket_refund(
                                job_list[-1]["delay_arcades"]) * ticket_price

                            batch_means_info["income"][sampling_count + step - 1] = income

                        # batch_means_info["time_current"][sampling_count + step - 1] = time.current
                    else:
                        if len(job_list) != 0:  # TODO: le statistiche delle recpliche successive non fanno media

                            batch_means_info["avg_wait_ticket"][sampling_count + step - 1], \
                            batch_means_info["std_ticket"][sampling_count + step - 1] = online_variance(replica + 1,
                                                                                                        batch_means_info[
                                                                                                            "avg_wait_ticket"][
                                                                                                            sampling_count + step - 1],
                                                                                                        batch_means_info[
                                                                                                            "std_ticket"][
                                                                                                            sampling_count + step - 1],
                                                                                                        job_list[- 1][
                                                                                                            "wait_ticket"])
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

                            income = len(job_list) * ticket_price - len(job_list) * ticket_refund(
                                job_list[- 1]["delay_arcades"]) * ticket_price

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
                '''print(node_list[0].number, node_list[1].number, node_list[2].number, node_list[3].number,
                      node_list[4].number)'''
                for i in range(0, len(node_list)):
                    if node_list[i].number > 0:
                        # if i == 0 or i == node_to_process.id:
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

                    # Simuliamo il sistema per ogni fascia oraria, in modo da capire se la fascia in questione
                    # raggiunge la stazionarietà o meno, ottendendo quindi una configurazione minima di server
                    # necessari.

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
                            if node.last is not None and node_list[0].last is not None and node_list[
                                0].last < node.last:
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
                        if node_to_process.last_completion is True:
                            clear_statistics(node_to_process.id)

                    if node_to_process.number > 0:
                        node_to_process.completion = time.current + get_service(node_to_process.id)
                    else:
                        node_to_process.completion = INFINITY

                    if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on
                        # ARCADE_i
                        if not is_positive():
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
                        else:
                            node_list[0].index += 1
                            node_list[0].number -= 1


                arrival_list = [node_list[n].arrival for n in range(1, max(n1, n2, n3, n4) + 1)]
                min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]

            #  Global batch means
            '''final_avg_wait_ticket = 0.0
            final_avg_delay_arcades = 0.0
            final_std_ticket = 0.0
            final_std_arcades = 0.0
            n = 0
            for i in range(4, len(batch_means_info["avg_wait_ticket"])):
                # print("len job list: ", len(job_list), ", index: ",node_list[0].index, ", batch_index: ",batch_index,"
                , begin for: ", b * batch_index, ", end for: ", b * batch_index + b, ", elem_index: ", i)
                n += 1
                #  avg calculation,  std calculation

                final_avg_wait_ticket, final_std_ticket = online_variance(n, final_avg_wait_ticket, final_std_ticket,
                                                                          batch_means_info["avg_wait_ticket"][i])
                final_avg_delay_arcades, final_std_arcades = online_variance(n, final_avg_delay_arcades, final_std_arcades,
                                                                             batch_means_info["avg_delay_arcades"][i])

            final_std_ticket = statistics.variance(batch_means_info["avg_wait_ticket"][4:])
            final_std_arcades = statistics.variance(batch_means_info["avg_delay_arcades"][4:])
            final_std_ticket = sqrt(final_std_ticket)
            final_std_arcades = sqrt(final_std_arcades)
            #  calculate interval width
            LOC = 0.95
            u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
            t = idfStudent(n - 1, u)  # critical value of t
            final_w_ticket = t * final_std_ticket / sqrt(n - 1)  # interval half width
            final_w_arcades = t * final_std_arcades / sqrt(n - 1)  # interval half width
            batch_means_info["final_wait_ticket"] = final_avg_wait_ticket
            batch_means_info["final_delay_arcades"] = final_avg_delay_arcades
            batch_means_info["final_std_ticket"] = final_std_ticket
            batch_means_info["final_std_arcades"] = final_std_arcades
            batch_means_info["final_w_ticket"] = final_w_ticket
            batch_means_info["final_w_arcades"] = final_w_arcades
            batch_means_info["correlation_delay_arcades"] = pearsonr(batch_means_info["avg_delay_arcades"][:k - 1],
             batch_means_info["avg_delay_arcades"][1:])
            print(pearsonr(batch_means_info["avg_delay_arcades"][:k-1], batch_means_info["avg_delay_arcades"][1:]))'''

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
        '''for i in range(0, len(batch_means_info["std_arcades"])):
            batch_means_info["std_arcades"][i] = sqrt(batch_means_info["std_arcades"][i] / replicas)
            batch_means_info["std_ticket"][i] = sqrt(batch_means_info["std_ticket"][i] / replicas)
            batch_means_info["std_system"][i] = sqrt(batch_means_info["std_system"][i] / replicas)
            # print(batch_means_info["std_arcades"][i])
            if replicas > 1:
                LOC = 0.95
                u = 1.0 - 0.5 * (1.0 - LOC)  # interval parameter
                t = idfStudent(replicas - 1, u)  # critical value of t
                w_ticket = t * batch_means_info["std_ticket"][i] / sqrt(replicas - 1)  # interval half width
                w_arcades = t * batch_means_info["std_arcades"][i] / sqrt(replicas - 1)  # interval half width
                w_system = t * batch_means_info["std_system"][i] / sqrt(replicas - 1)  # interval half width
                batch_means_info["w_ticket"].append(w_ticket)
                batch_means_info["w_arcades"].append(w_arcades)
                batch_means_info["w_system"].append(w_system)'''
        path = "stats_" + str(seed) + ".json"
        with open(path, 'w+') as json_file:
            json.dump(batch_means_info, json_file, indent=4)
        json_file.close()
        # plot_stats()
        print(node_list[0].index)

    plot_stats_global()
