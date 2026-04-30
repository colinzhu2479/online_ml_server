"""
Online Machine Learning Prediction Server Module.

This module implements an independent client-server style prediction architecture 
in which an online ML server provides asynchronous force predictions and model 
versioning. 

The server operates independently of the molecular dynamics engine and supports 
multiple concurrent simulations. This architecture facilitates scalable deployment, 
centralized model management, and efficient reuse of learned representations 
across different molecular subsets.
"""

import numpy as np
import Pyro4
import socket
import time
from datetime import datetime
import tensorflow as tf

from scipy import spatial
from tensorflow import keras
import nn_preprocess
import os
import gc

from scipy.spatial import distance

from fragment_transform import atom_mass_mapping, direction_order
import fragment_transform
import threading
import functools
import copy
import prepare_transfer
import transfer
from pathlib import Path

import model_init
import math
import traceback
from itertools import chain
from typing import List, Dict, Any, Tuple, Optional, Union

# Server configurations
Pyro4.config.THREADPOOL_SIZE = 2000
log_path: str = ''

os.environ["PYRO_LOGFILE"] = log_path + "pyro.log"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"
running: bool = True

print = functools.partial(print, flush=True)


@Pyro4.expose
class NnServer(object):
    """
    Pyro4 RPC Server class exposing ML prediction and retraining endpoints to clients.
    Manages loaded neural network models, incoming transfer data, and concurrent requests.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NnServer with paths from the provided configuration and 
        sets up all instance variables for state management.
        """
        self.model_path = config['models']['model_path']
        self.train_set_path = config['models']['train_set_path']
        self.temp_path = config['models']['temp_path']

        # Instance state variables
        self.sim_list: Dict[str, List[int]] = {}  # [predictable_count, non_predictable_count]
        self.transfer_info: Dict[str, List[Any]] = {} 

        self.e_model_list: Dict[str, keras.Model] = {}
        self.f_model_list: Dict[str, keras.Model] = {}

        self.e_model_para: Dict[str, List[float]] = {}
        self.f_model_para: Dict[str, List[float]] = {}

        self.e_model_temp: Dict[str, keras.Model] = {}  # Model copies for retraining
        self.f_model_temp: Dict[str, keras.Model] = {}
        
        # Stores transfer data sent from clients
        self.transfer_set: Dict[str, Dict[str, List[Any]]] = {}  
        self.transfer_temp: Dict[str, Dict[str, List[Any]]] = {}  

        self.e_train_set: Dict[str, List[Any]] = {} 
        self.train_set: Dict[str, Dict[str, List[Any]]] = {}  

        self.tree: Dict[str, spatial.KDTree] = {}
        self.raw_tree: Dict[str, spatial.KDTree] = {}
        self.train_set_min_dis: Dict[str, float] = {}  

        self.transfer_size: Dict[str, int] = {}
        self.radius: Dict[str, float] = {}
        self.inertia: Dict[str, float] = {}
        self.on_training: Dict[str, bool] = {}

        self.can_predict: Dict[str, bool] = {}
        self.predicting: Dict[str, int] = {}

        self.n_p: int = 0
        self.n_c: int = 0

        # Concurrency and logging primitives
        self.stop_event: threading.Event = threading.Event()
        self._init_locks: Dict[str, threading.Lock] = {}
        self._lock0: threading.Lock = threading.Lock()  
        self._lock: Dict[str, threading.Lock] = {}  
        self._lock2: Dict[str, threading.Lock] = {}  

        # Start background logging thread
        self.thread0: threading.Thread = threading.Thread(target=self._auto_log, daemon=False)
        self.thread0.start()

    def _auto_log(self) -> None:
        """Background thread target that periodically logs the server status."""
        f_path = log_path + 'history.txt'
        while not self.stop_event.wait(timeout=1800):
            mode = 'a' if os.path.exists(f_path) else 'w'
            with open(f_path, mode) as f:
                t = time.ctime()
                f.write(t + '\n')
                f.write(self.get_stat())

    def get_stat(self) -> str:
        """Format and return high-level server stats for periodic logging."""
        training = ', '.join([i for i, is_training in self.on_training.items() if is_training])
        status = f'Server status at {time.ctime()}:\nModels loaded: {list(self.sim_list.keys())}\nModels on training: {training}\nTotal prediction: {self.n_p}\nTotal not predicted: {self.n_c}\n\nModel predictability info (predictable, non-predictable):\n'
        for idd, counts in self.sim_list.items():
            status += f'{idd}: {counts}\n'
        return status + '\n'

    def query(self, xyz: List[List[float]], atn: List[int], clu_r: float = 1.0, classify: bool = True) -> List[Any]:
        """
        Endpoint for clients to query energy and force predictions for a specific configuration.
        """
        xyz_t, atn_t = nn_preprocess.input_process(np.array(xyz), np.array(atn, dtype=int), transfer_fragment=True)
        id = get_name(get_id(atn_t))
        num = len(atn_t)

        if id not in self.sim_list:
            gateway_lock = self._init_locks.setdefault(id, threading.Lock())
            with gateway_lock:
                if id not in self.sim_list:
                    try:
                        e_temp, para_temp = load_energy_model([id], self.model_path)
                        self.e_model_list.update(e_temp)
                        self.e_model_para.update(para_temp)
                    except:
                        return [False]
                    else:
                        self.sim_list[id] = [0, 0]
                        self.can_predict[id] = True
                        self.predicting[id] = 0
                        self._lock2[id] = threading.Lock()

                        for tar in ['x', 'y', 'z']:
                            f_temp, fpara_temp = load_force_model([id], tar, self.model_path)
                            self.f_model_list.update(f_temp)
                            self.f_model_para.update(fpara_temp)

                        if classify:
                            self.e_train_set[id] = np.loadtxt(self.train_set_path + id + '_train.txt').tolist()
                            self.tree[id] = spatial.KDTree(self.e_train_set[id])
                            self.raw_tree[id] = spatial.KDTree(self.e_train_set[id])
                            self.radius[id] = np.loadtxt(self.train_set_path + id + '_radius.txt').reshape(-1)[0] * clu_r
                            self.inertia[id] = np.loadtxt(self.train_set_path + id + '_inertia.txt').reshape(-1)[0] * clu_r

        predictable = True if not classify else bool(__classify__(xyz_t, self.tree[id], self.radius[id]))

        if predictable:
            while True:
                with self._lock2[id]:
                    if self.can_predict[id]:
                        self.predicting[id] += 1
                        break
                time.sleep(0.01)

            try:
                p_e = predict_energy(xyz_t, [np.arange(len(atn_t))], self.e_model_list, self.e_model_para, id)[0]
                p_f = predict_force(atn, xyz, [np.arange(len(atn_t))], self.f_model_list, self.f_model_para, id)[0]
            finally:
                with self._lock2[id]:
                    self.predicting[id] -= 1

            self.n_p += 1
            self.sim_list[id][0] += 1
            return [predictable, p_e[0].tolist(), p_f.tolist()]
        else:
            self.n_c += 1
            self.sim_list[id][1] += 1
            return [predictable]

    def get_status(self) -> str:
        """Returns the current operational status of the server."""
        training = ', '.join([i for i, is_training in self.on_training.items() if is_training])
        status = f'Server status at {time.ctime()}:\nModels loaded: {list(self.sim_list.keys())}\nModels on training: {training}\nTotal prediction: {self.n_p}\nTotal not predicted: {self.n_c}\n\nModel predictability info (predictable, non-predictable):\n'
        for idd, counts in self.sim_list.items():
            status += f'{idd}: {counts}\n'
        status += '\nCollected transfer size:\n'
        for idd in self.transfer_set.keys():
            status += f"{idd}: {len(self.transfer_set[idd]['zd'])}/{self.transfer_size[idd]}\n"
        return status

    @Pyro4.oneway
    def collect_e(self, xyz: List[List[float]], atn: List[int], e: float, f: List[List[float]], ratio: float = 1.0) -> None:
        """
        Endpoint for clients to submit unpredicted ground-truth configurations back to the server.
        If a threshold of data is reached, an asynchronous retraining task is triggered.
        """
        tf.get_logger().setLevel('ERROR')
        atn_t = np.sort(atn)
        id = get_name(get_id(atn_t))
        num_atom = len(atn_t)

        x_dist, x_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'x')
        y_dist, y_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'y')
        z_dist, z_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'z')

        fragment_lock = self._lock.setdefault(id, threading.Lock())
        with fragment_lock:
            if id not in self.transfer_set:
                self.transfer_set[id] = {
                    'xd': [x_dist], 'yd': [y_dist], 'zd': [z_dist],
                    'xf': [x_f], 'yf': [y_f], 'zf': [z_f], 'e': [e], 
                    'f':[list(chain.from_iterable(f))], 'xyz':[list(chain.from_iterable(xyz))], 'atn':[atn]
                }
                self.transfer_size[id] = num_atom ** 2 * 10
                self.on_training[id] = False
            else:
                self.transfer_set[id]['xd'].append(x_dist)
                self.transfer_set[id]['yd'].append(y_dist)
                self.transfer_set[id]['zd'].append(z_dist)
                self.transfer_set[id]['xf'].append(x_f)
                self.transfer_set[id]['yf'].append(y_f)
                self.transfer_set[id]['zf'].append(z_f)
                self.transfer_set[id]['e'].append(e)
                self.transfer_set[id]['f'].append(f)
                self.transfer_set[id]['xyz'].append(xyz)
                self.transfer_set[id]['atn'].append(atn)

        trigger_update = False
        with self._lock[id]:
            if len(self.transfer_set[id]['xd']) >= self.transfer_size[id] and not self.on_training[id]:
                self.on_training[id] = True
                trigger_update = True
                print(f'transfer triggered on {time.ctime()} for {id}')
                self.transfer_temp[id] = copy.deepcopy(self.transfer_set[id])
                self.transfer_set[id] = {'xd': [], 'yd': [], 'zd': [], 'xf': [], 'yf': [], 'zf': [], 'e': [], 'f':[], 'xyz':[], 'atn':[]}

        if trigger_update:
            try:
                time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                time_ = time.time()
                
                # Check if model exists; if not, initialize one
                if id in self.sim_list.keys():
                    if id not in self.train_set.keys():
                        self.train_set[id] = prepare_transfer.load_force_input(id, self.train_set_path)

                    if id + 'p' not in self.e_model_temp.keys():
                        self.e_model_temp[id + 'p'] = keras.models.load_model(self.model_path + f'{id}/' + "%s_primary.tf" % str(id), custom_objects={"Dense": tf.keras.layers.Dense})
                        self.e_model_temp[id + 's'] = keras.models.load_model(self.model_path + f'{id}/' + "%s_secondary.tf" % str(id), custom_objects={"Dense": tf.keras.layers.Dense})
                        for direction in ['x', 'y', 'z']:
                            self.f_model_temp[direction + id + 'p'] = keras.models.load_model(self.model_path + f'{id}/{direction}/' + "%s_primary.tf" % str(id))
                            self.f_model_temp[direction + id + 's'] = keras.models.load_model(self.model_path + f'{id}/{direction}/' + "%s_secondary.tf" % str(id))
                else:
                    self.initialize_model(id, num_atom, time_tag, self.temp_path, ratio=ratio)

                # Transfer learning Execution
                transfer_order, transfer_order_list = transfer.main(
                    np.array(self.train_set[id]['zd']), np.array(self.train_set[id]['e']),
                    np.array(self.transfer_temp[id]['zd']), np.array(self.transfer_temp[id]['e']),
                    self.e_model_temp, self.e_model_para, self.radius[id], self.inertia[id], 
                    id, len(self.transfer_temp[id]['xf'][0]), direction=''
                ) 
                
                for direction in ['x', 'y', 'z']:
                    transfer.main_with_order(
                        np.array(self.train_set[id][direction + 'd']), np.array(self.train_set[id][direction + 'f']),
                        np.array(self.transfer_temp[id][direction + 'd']), np.array(self.transfer_temp[id][direction + 'f']), 
                        self.f_model_temp, self.f_model_para, id, len(self.transfer_temp[id]['xf'][0]), 
                        transfer_order_list, direction=direction
                    )

                self.save_model_info(time_tag, transfer_order, round(time.time() - time_, 2), id)

                # Reload newly minted models into memory
                new_e_primary = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/%s_primary.tf' % id)
                new_e_secondary = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/%s_secondary.tf' % id)
                
                temp_f_dict = {}
                for direction in ['x', 'y', 'z']:
                    temp_f_dict[direction + id + 'p'] = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/{direction}/' +"%s_primary.tf" % str(id))
                    temp_f_dict[direction + id + 's'] = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/{direction}/' +"%s_secondary.tf" % str(id))

                self.e_train_set[id].extend([self.transfer_temp[id]['zd'][i] for i in transfer_order])
                temp_tree = spatial.KDTree(self.e_train_set[id])

                # Commit new configurations to historical training data cache
                for k in ['xd', 'yd', 'zd', 'xf', 'yf', 'zf', 'e']:
                    self.train_set[id][k].extend([self.transfer_temp[id][k][i] for i in transfer_order])

                # Temporarily lock inquiries to perform hot-swap of models
                if id in self.sim_list.keys(): 
                    while True:
                        with self._lock2[id]:
                            self.can_predict[id] = False
                            if self.predicting[id] <= 0:
                                break
                        time.sleep(0.01)

                self.tree[id] = temp_tree
                self.e_model_list[id + 'p'] = new_e_primary
                self.e_model_list[id + 's'] = new_e_secondary

                for direction in ['x', 'y', 'z']:
                    self.f_model_list[direction + id + 'p'] = temp_f_dict[direction + id + 'p']
                    self.f_model_list[direction + id + 's'] = temp_f_dict[direction + id + 's']

                if id not in self.sim_list.keys():
                    gateway_lock = self._init_locks.setdefault(id, threading.Lock())
                    with gateway_lock:
                        self.sim_list[id] = [0, len(self.transfer_set[id]["xd"]) + len(self.transfer_temp[id]['xd'])]
                        self.can_predict[id] = True
                        self.predicting[id] = 0
                        self._lock2[id] = threading.Lock()
            except Exception as e:
                print(f"CRITICAL ERROR during retraining for {id}: {e}")
                err_str = traceback.format_exc()
                print("Error_info:", err_str)
            finally: 
                # Re-enable inquiries
                if id in self.can_predict and id in self._lock2:
                    with self._lock2[id]:
                        self.can_predict[id] = True
                if id in self.transfer_temp:
                    del self.transfer_temp[id]
                gc.collect()
                with self._lock[id]:
                    self.on_training[id] = False
                print(f'transfer finished on {time.ctime()} for {id}')

    def initialize_model(self, id: str, num_atom: int, time_tag: str, temp_p: str, ratio: float = 1.0) -> None:
        """Initializes network architectures and dynamic threshold scaling for a novel fragment."""
        print(f'Model initialization :{id}')
        self.train_set[id] = {'xd': [], 'yd': [], 'zd': [], 'xf': [], 'yf': [], 'zf': [], 'e': []}

        # Initialize generic initial models based on dimension metadata
        self.e_model_temp[id + 'p'] = model_init.build_model(self.transfer_temp[id]['zd'][0], 'energy', num_atom, 4, 4)
        self.e_model_temp[id + 's'] = model_init.build_model(self.transfer_temp[id]['zd'][0], 'energy', num_atom, 4, 4)
        
        x_range, x_min, y_range, y_min, e_range, e_min = model_init.def_normalization(
            np.array(self.transfer_temp[id]['zd']), np.array(self.transfer_temp[id]['e']), 'energy')
            
        self.e_model_para[id + 'px'] = [x_range, x_min]
        self.e_model_para[id + 'py'] = [y_range, y_min]
        self.e_model_para[id + 'sy'] = [e_range, e_min]

        temp_path = temp_p + f'{id}/init/'
        save_para(id, x_range, x_min, y_range, y_min, e_range, e_min, Path(temp_path), temp_path)

        self.e_train_set[id] = []
        self.tree[id] = spatial.KDTree(self.transfer_temp[id]['zd'])
        self.raw_tree[id] = spatial.KDTree(self.transfer_temp[id]['zd'])

        # Determine thresholds based on atom counts
        if num_atom == 3:
            self.radius[id] = 0.01032071
            self.inertia[id] = 0.000003050566
        elif num_atom == 4:
            self.radius[id] = 0.04085604
            self.inertia[id] = 0.000304219896
        elif num_atom == 6:
            self.radius[id] = 0.15443308
            self.inertia[id] = 0.001101525739
        elif num_atom == 7:
            self.radius[id] = 0.18680639
            self.inertia[id] = 0.002386698899
        elif num_atom > 4:
            self.radius[id] = 0.03 * num_atom * ratio
            self.inertia[id] = 0.01 * self.radius[id] * math.sqrt(len(self.transfer_temp[id]['zd'][0])) * ratio
        elif num_atom >= 3:
            self.radius[id] = (0.03 * num_atom - 0.08) * ratio
            self.inertia[id] = (0.01 * self.radius[id] * math.sqrt(len(self.transfer_temp[id]['zd'][0])) - 0.00017) * ratio

        for direction in ['x', 'y', 'z']:
            self.f_model_temp[direction + id + 'p'] = model_init.build_model(
                self.transfer_temp[id][f'{direction}d'][0], 'force', num_atom, 6, 4)
            self.f_model_temp[direction + id + 's'] = model_init.build_model(
                self.transfer_temp[id][f'{direction}d'][0], 'force', num_atom, 6, 4)
                
            x_range, x_min, y_range, y_min, e_range, e_min = model_init.def_normalization(
                np.array(self.transfer_temp[id][f'{direction}d']),
                np.array(self.transfer_temp[id][f'{direction}f']), 'force')

            self.f_model_para[direction + id + 'px'] = [x_range, x_min]
            self.f_model_para[direction + id + 'py'] = [y_range, y_min]
            self.f_model_para[direction + id + 'sy'] = [e_range, e_min]

            save_para(id, x_range, x_min, y_range, y_min, e_range, e_min, Path(temp_path + f'{direction}/'), temp_path+ f'{direction}/')

    def save_model_info(self, time_tag: str, transfer_order: List[int], time_spent: float, id: str) -> None:
        """Saves Keras models and logs post-transfer learning execution."""
        temp_path = self.temp_path + f'{id}/{time_tag}/'
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        
        self.e_model_temp[f"{id}p"].save(temp_path + f"{id}_primary.tf", overwrite=True)
        self.e_model_temp[f"{id}s"].save(temp_path + f"{id}_secondary.tf", overwrite=True)

        for direction in ['x', 'y', 'z']:
            Path(temp_path + direction + '/').mkdir(parents=True, exist_ok=True)
            self.f_model_temp[direction + f"{id}p"].save(temp_path + direction + f"/{id}_primary.tf", overwrite=True)
            self.f_model_temp[direction + f"{id}s"].save(temp_path + direction + f"/{id}_secondary.tf", overwrite=True)

        with open(temp_path + 'transfer_info.txt', "w") as f:
            f.write(f'Time: {time.ctime()}\n')
            f.write(f'train set size: {len(self.train_set[id]["xd"])}\n')
            f.write(f'transfer set size: {len(self.transfer_temp[id]["zd"])}\n')
            f.write(f'transfer used size: {len(transfer_order)}\n')
            f.write(f'transfer wall time spend: {time_spent}s\n')
            
        np.savetxt(temp_path + 'transfer_xd.txt', self.transfer_temp[id]["zd"])
        np.savetxt(temp_path + "order_used.txt", transfer_order, fmt="%i")
        np.savetxt(temp_path + 'transfer_e.txt', self.transfer_temp[id]["e"])

    @Pyro4.oneway
    def clear_status(self) -> None:
        """Resets prediction and failure counters."""
        self.n_p = 0
        self.n_c = 0
        for i in self.sim_list.keys():
            self.sim_list[i] = [0, 0]

    @Pyro4.oneway
    def shutdown(self) -> None:
        """Gracefully shuts down the server and outputs a final report."""
        self.stop_event.set()
        self.thread0.join()
        status = self.get_stat()
        with open(log_path + 'report.txt', 'w') as f:
            f.write(f'Printed time: {time.ctime()}\n')
            f.write(f'Total number of predictions: {self.n_p}\n')
            f.write(f'Total number of calculations: {self.n_c}\n')
            f.write(status)
            print('Server shutting down.')
        global running
        running = False


# ==========================================
# Helper functions strictly decoupled from class state
# ==========================================

def save_para(
    id: str, x_range: float, x_min: float, y_range: float, y_min: float, 
    e_range: float, e_min: float, path: Path, temp_path: str
) -> None:
    """Helper function to save normalization parameters persistently."""
    path.mkdir(parents=True, exist_ok=True)
    np.savetxt(temp_path + f"{id}_p_x.txt", np.concatenate([[x_range], [x_min]], axis=0))
    np.savetxt(temp_path + f"{id}_p_y.txt", np.concatenate([[y_range], [y_min]], axis=0))
    np.savetxt(temp_path + f"{id}_s_y.txt", np.concatenate([[e_range], [e_min]], axis=0))


def __classify__(xyz: np.ndarray, tree_o: spatial.KDTree, radius: float) -> bool:
    """Evaluates if a set of coordinates falls within the known radius of the KDTree data structure."""
    dist_o, _ = tree_o.query(xyz)
    return bool(dist_o <= radius)


def predict_energy(
    xyz: np.ndarray, partition_index: List[Any], model_dict: Dict[str, keras.Model], 
    param_dict: Dict[str, List[float]], model_name: str
) -> np.ndarray:
    """Computes the primary and delta predictions for energy, recovering raw physical scale via loaded params."""
    p_dis = np.array(xyz)
    x_range, xmin = param_dict[f"{model_name}px"]
    p_dis = ((p_dis - xmin) / x_range).reshape(1, np.size(p_dis))

    y_range, y_min = param_dict[f"{model_name}py"]
    e_range, e_min = param_dict[f"{model_name}sy"]

    p_e = model_dict[f"{model_name}p"](p_dis) * y_range + y_min + \
          (model_dict[f"{model_name}s"](p_dis) * e_range + e_min) / 627.503
    return p_e.numpy()


def predict_force(
    atomic_num: List[int], xyz: List[List[float]], partition_index: List[np.ndarray], 
    model_dict: Dict[str, keras.Model], param_dict: Dict[str, List[float]], 
    fragment_name_input: Optional[str], real: Optional[Any] = None
) -> np.ndarray:
    """Predicts force arrays over defined partitions (subsystems)."""
    direction = ['x', 'y', 'z']
    predicted_energy_list = np.zeros(len(partition_index), dtype='object')
    n = 0
    for p in partition_index:  
        num_atom = len(p)
        num_dis = int(num_atom * (num_atom - 1) / 2)
        p_dis = np.zeros(num_dis)

        partition_ar = np.array(atomic_num)[p]
        xyz_t = np.array(xyz)[p]
        force = np.zeros([len(partition_ar), 3])
        ref = np.ones([len(partition_ar), 3]) * 3
        atomic_mass = atom_mass_mapping(partition_ar)
        reconstructed_force = np.zeros([3, len(partition_ar)])

        for c, ii in enumerate(fragment_transform.global_transform_fragment(xyz_t, force, partition_ar, atomic_mass, ref)):
            xyz_tt, force, partition_ar, v, order = ii
            target_direction = direction[c]
            p_dis = p_dis.reshape(-1)
            dis_matrix = distance.cdist(xyz_tt, xyz_tt)
            
            k = 0
            for i in range(len(dis_matrix) - 1):
                for j in range(i, len(dis_matrix) - 1):
                    p_dis[k] = dis_matrix[i][j + 1]
                    k += 1

            fragment_name = fragment_name_input or str(num_atom)

            x_range, xmin = param_dict[f"{target_direction}{fragment_name}px"]
            p_dis = ((p_dis - xmin) / x_range).reshape(1, num_dis)

            y_range, y_min = param_dict[f"{target_direction}{fragment_name}py"]
            e_range, e_min = param_dict[f"{target_direction}{fragment_name}sy"]

            p_e = model_dict[f"{target_direction}{fragment_name}p"](p_dis) * y_range + y_min + \
                  (model_dict[f"{target_direction}{fragment_name}s"](p_dis) * e_range + e_min) / 627.503
            reconstructed_force[c] = p_e[0].numpy()[np.argsort(order)]

        inv_v = np.linalg.inv(v)
        predicted_energy_list[n] = np.matmul(inv_v, reconstructed_force).T
        n += 1
    return predicted_energy_list


def get_id(atn: np.ndarray) -> str:
    """Converts a numpy array of atomic numbers into a hyphen-delimited signature."""
    return '-'.join(map(str, atn)) + '-'


def get_name(id: str) -> str:
    """Decodes a hyphen-delimited signature into a stoichiometric formula string."""
    atn = np.array((id + '0').split('-')).astype(int)
    name = ''
    atnum = {0: '', 1: 'H', 6: "C", 7: "N", 8: "O"}
    for i in range(1, np.max(atn) + 1):
        count = np.sum(atn == i)
        if count > 0:
            name += atnum.get(i, '') + str(count)
    return name


def load_energy_model(num_atom_list: List[str], model_path_root: str) -> Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]:
    """Loads energy models and metadata from disk paths."""
    tf.get_logger().setLevel('ERROR')
    model_dict = {}
    param_dict = {}
    for ll in num_atom_list:
        model_path = os.path.join(model_path_root, str(ll), '')
        form = '.h5' if os.path.exists(model_path + f"{ll}_primary.h5") else '.tf'
        
        model_dict[f"{ll}p"] = keras.models.load_model(model_path + f"{ll}_primary{form}")
        model_dict[f"{ll}s"] = keras.models.load_model(model_path + f"{ll}_secondary{form}")

        param_dict[f"{ll}px"] = np.loadtxt(model_path + f"{ll}_p_x.txt")
        param_dict[f"{ll}py"] = np.loadtxt(model_path + f"{ll}_p_y.txt")
        param_dict[f"{ll}sy"] = np.loadtxt(model_path + f"{ll}_s_y.txt")
        print(f'{ll} energy model loaded.')
    return model_dict, param_dict


def load_force_model(frag_list: List[str], target_direction: str, model_path_root: str) -> Tuple[Dict[str, keras.Model], Dict[str, np.ndarray]]:
    """Loads force models and metadata from disk paths."""
    tf.get_logger().setLevel('ERROR')
    model_dict = {}
    param_dict = {}
    for ll in frag_list:
        model_path = os.path.join(model_path_root, str(ll), target_direction, '')
        format_type = '.h5' if os.path.exists(model_path + f"{ll}_primary.h5") else '.tf'

        model_dict[f"{target_direction}{ll}p"] = keras.models.load_model(model_path + f"{ll}_primary{format_type}")
        model_dict[f"{target_direction}{ll}s"] = keras.models.load_model(model_path + f"{ll}_secondary{format_type}")
        
        param_dict[f"{target_direction}{ll}px"] = np.loadtxt(model_path + f"{ll}_p_x.txt")
        param_dict[f"{target_direction}{ll}py"] = np.loadtxt(model_path + f"{ll}_p_y.txt")
        param_dict[f"{target_direction}{ll}sy"] = np.loadtxt(model_path + f"{ll}_s_y.txt")
        print(f'{ll} {target_direction} force model loaded.')
    return model_dict, param_dict


def launch(
    config: Dict[str, Any], pw: bytes = b"xiao", 
    conn_f: str = '/N/project/sico/nn_server/connection.txt',
    log_f: str = '/N/project/sico/nn_server/log.txt'
) -> None:
    """Registers the server daemon and launches the Pyro4 event loop."""
    host = socket.gethostname()
    ip = Pyro4.socketutil.getIpAddress(host)

    daemon = Pyro4.Daemon(host=ip)
    daemon._pyroHmacKey = pw

    uri = daemon.register(NnServer(config))
    with open(conn_f, 'w') as f:
        f.write(str(uri) + '\n')
        f.write(ip + '\n')
        f.write(host + '\n')
        f.write(time.ctime() + '\n')
        
    global running
    print("Ready for connections.")
    daemon.requestLoop(loopCondition=lambda: running)


if __name__ == "__main__":
    import sys
    try:
        import tomllib as toml
    except ImportError:
        import toml

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    os.chdir(script_dir)
    with open("server.toml", "rb") as f:
        config = toml.load(f)
        
    launch(config, pw=config["server"]["password"].encode(), 
           conn_f=config["server"]["connection_file"], log_f=config["server"]["log_file"])
