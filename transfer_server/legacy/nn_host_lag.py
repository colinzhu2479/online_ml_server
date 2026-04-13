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
# from transfer.NN_force import model_path
import threading
import functools
import copy
import prepare_transfer
import transfer
from pathlib import Path

import model_init
import math

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

Pyro4.config.DETAILED_TRACEBACK = True
Pyro4.config.LOGWIRE = True


Pyro4.config.THREADPOOL_SIZE = 2000

log_path = ''

os.environ["PYRO_LOGFILE"] = log_path+"pyro.log"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"
running = True

print = functools.partial(print, flush=True)

def _auto_log(stop_event):
    f_path = log_path+'history.txt'
    while not stop_event.wait(timeout=1800):
        #time.sleep(10)
        if os.path.exists(f_path):
            f = open(f_path,'a')
        else:
            f = open(f_path,'w')
        t = time.ctime()
        f.write(t+'\n')
        f.write(get_stat())
        f.close()


@Pyro4.expose
class NnServer(object):

    # self.id_list = []
    sim_list = dict({}) #[predicable, non predictable]

    transfer_info = dict({}) # [id:[transfer set, used transfer set, wall time spent(s)]]

    e_model_list = dict({})
    f_model_list = dict({})

    e_model_para = dict({})
    f_model_para = dict({})

    e_model_temp = dict({}) #model copy for 
    f_model_temp = dict({})
    transfer_set = dict({}) # {id:[(xyz,F1,E1),()]} #storing transfer data from client
    # second thought {id:[((x_dist_list, y_dist_list, z_dist_list),(F_x, F_y, F_z), E1),()]}
    # third thought {id: {xd:[], yd:[], zd:[], xf:[], yf:[], zf:[], e:[]}}
    transfer_temp = dict({}) # a copy of transfer set for training

    #.f_transfer_set = dict
    e_train_set = dict({}) # {id:z_dist_list,}

    train_set = dict({}) # {id:[((x_dist_list, y_dist_list, z_dist_list),(F_x, F_y, F_z), E1),()]}
    #{id: {xd:[], yd:[], zd:[], xf:[], yf:[], zf:[], e:[]}}
    
    tree = dict({})
    raw_tree = dict({})
    train_set_min_dis = dict({}) # keep track of the closest distance to any training point
    
    #.f_train_set = dict
    transfer_size = dict({})
    radius = dict({})
    inertia = dict({})
    on_training = dict({})
    
    #model_path=''#config['models']['model_path']
    #train_set_path=''#config['models']['train_set_path']
    n_p = 0
    n_c = 0

    stop_event = threading.Event()
    thread0 = threading.Thread(target=_auto_log, args=(stop_event,), daemon=False)
    thread0.start()

    _lock = threading.Lock()

    def __init__(self, config):
        self.model_path=config['models']['model_path']
        self.train_set_path=config['models']['train_set_path']
        self.temp_path=config['models']['temp_path']


    def query(self, xyz, atn, clu_r=1, classify=True):
        #print(1)
        xyz_t, atn_t = nn_preprocess.input_process(np.array(xyz), np.array(atn, dtype=int), transfer_fragment=True)
        #print(2)
        id = get_name(get_id(atn_t))
        num = len(atn_t)
        if id not in NnServer.sim_list.keys():
            #print(id,NnServer.sim_list.keys())
            try:
                e_temp, para_temp = load_energy_model([id], self.model_path)
                NnServer.e_model_list.update(e_temp)
                NnServer.e_model_para.update(para_temp)
                #print(-0.9)
            except:
                return [False]
            else:
                
                NnServer.sim_list[id]=[0,0]
                
                for tar in ['x', 'y', 'z']:
                    f_temp, fpara_temp = load_force_model([id], tar, self.model_path)
                    NnServer.f_model_list.update(f_temp)
                    NnServer.f_model_para.update(fpara_temp)
                if classify:
                    #print(1)
                    NnServer.e_train_set[id] = np.loadtxt(self.train_set_path+id+'_train.txt').tolist()
                    # distance list, unnormalized, transformed
                    NnServer.tree[id] =  spatial.KDTree(NnServer.e_train_set[id])
                    NnServer.raw_tree[id] = spatial.KDTree(NnServer.e_train_set[id])

                    NnServer.radius[id]=np.loadtxt(self.train_set_path+id+'_radius.txt').reshape(-1)[0]*clu_r
                    NnServer.inertia[id]=np.loadtxt(self.train_set_path+id+'_inertia.txt').reshape(-1)[0]*clu_r
                    #print(2)

        if classify is False:
            predictable = True
        else:
            predictable = bool(__classify__(xyz_t, NnServer.tree[id], NnServer.radius[id]))
        #print(3,predictable,bool(predictable))
        if predictable is True:
            #print(3.1)
            p_e = predict_energy(xyz_t, [np.arange(len(atn_t))], NnServer.e_model_list, NnServer.e_model_para, id)[0]
            #print(4)
            p_f = predict_force(atn,xyz,[np.arange(len(atn_t))], NnServer.f_model_list, NnServer.f_model_para, id)[0]
            #print(p_e,p_f)
            NnServer.n_p += 1
            NnServer.sim_list[id][0] += 1
            return [predictable, p_e[0].tolist(), p_f.tolist()]
        else:
            NnServer.n_c += 1
            NnServer.sim_list[id][1] += 1
        return [predictable]

    def get_status(self):
        training = ''
        
        for i in NnServer.on_training.keys():
            if NnServer.on_training[i] == True:
                training = training + i +', '

        status = f'Server status at {time.ctime()}:\nModels loaded: {NnServer.sim_list.keys()}\nModels on training: {training}\nTotal prediction: {NnServer.n_p}\nTotal not predicted: {NnServer.n_c}\n\nModel predictability info (predictable, non-predictable):\n'
        for idd in NnServer.sim_list.keys():
            status = status + f'{idd}: ' + str(NnServer.sim_list[idd]) + '\n'
        status += '\nCollected transfer size:\n'
        for idd in NnServer.transfer_set.keys():
            status = status + f'{idd}: ' + str(len(NnServer.transfer_set[idd]['zd'])) + f'/{NnServer.transfer_size[idd]}\n'

        return status

    @Pyro4.oneway
    def collect_e(self, xyz, atn, e, f, ratio=1):
        tf.get_logger().setLevel('ERROR')
        #xyz_t, atn_t, target_f = nn_preprocess.input_process(xyz, atn, transfer_fragment=True)
        atn_t = np.sort(atn)
        id = get_name(get_id(atn_t))
        num_atom = len(atn_t)

        x_dist, x_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'x')
        y_dist, y_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'y')
        z_dist, z_f = prepare_transfer.get_force_io(np.array(xyz), atn, num_atom, f, 'z')

        if id not in NnServer.transfer_set.keys(): # transfer data collection
            with NnServer._lock:
                #NnServer.transfer_set[id] = [(xyz, (f, e))]
                #NnServer.transfer_set[id] = [((x_dist, y_dist, z_dist), (x_f, y_f, z_f), e)]
                NnServer.transfer_set[id] = dict({'xd':[x_dist], 'yd':[y_dist], 'zd':[z_dist], 'xf':[x_f], 'yf':[y_f],
                                                  'zf':[z_f], 'e':[e]})
                NnServer.transfer_size[id] = num_atom**2*10
                NnServer.on_training[id] = False
        else:
            with NnServer._lock:
                #NnServer.transfer_set[id].append(((x_dist, y_dist, z_dist), (x_f, y_f, z_f), e))
                NnServer.transfer_set[id]['xd'].append(x_dist)
                NnServer.transfer_set[id]['yd'].append(y_dist)
                NnServer.transfer_set[id]['zd'].append(z_dist)
                NnServer.transfer_set[id]['xf'].append(x_f)
                NnServer.transfer_set[id]['yf'].append(y_f)
                NnServer.transfer_set[id]['zf'].append(z_f)
                NnServer.transfer_set[id]['e'].append(e)

        # check transfer set size
        NnServer._lock.acquire()
        if len(NnServer.transfer_set[id]['xd']) >= NnServer.transfer_size[id] and not NnServer.on_training[id]:
            time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            NnServer.on_training[id] = True
            print(f'transfer triggered on {time.ctime()} for {id}')

            # light copy transfer set to transfer temp
            NnServer.transfer_temp[id] = copy.deepcopy(NnServer.transfer_set[id])
            # reset transfer set
            NnServer.transfer_set[id] = dict({'xd': [], 'yd': [], 'zd': [], 'xf': [], 'yf': [],
                                              'zf': [], 'e': []})
            NnServer._lock.release()
            time_ = time.time()
            if id in NnServer.sim_list.keys():  # this suggests fragment recognized and has a loaded model
                '''
                Transfer learning path, assume simplex recognized, exist in simplex list, train set and model exist
                '''

                # load and process train set
                if id not in NnServer.train_set.keys():
                    # heavy train set load
                    NnServer.train_set[id] = prepare_transfer.load_force_input(id, self.train_set_path)


                if id + 'p' not in NnServer.e_model_temp.keys(): #make model template for transfer learning
                    #NnServer.e_model_temp[id + 'p'] = copy_model(NnServer.e_model_list[id + 'p'])
                    #NnServer.e_model_temp[id + 's'] = copy_model(NnServer.e_model_list[id + 's'])
                    NnServer.e_model_temp[id + 'p'] = keras.models.load_model(self.model_path+f'{id}/' +
                                                                              "%s_primary.tf" % str(id),custom_objects={"Dense": tf.keras.layers.Dense})
                    NnServer.e_model_temp[id + 's'] = keras.models.load_model(self.model_path+f'{id}/' +
                                                                              "%s_secondary.tf" % str(id),custom_objects={"Dense": tf.keras.layers.Dense})
                    for direction in ['x', 'y', 'z']:
                        #NnServer.f_model_temp[direction + id + 'p'] = copy_model(
                        #    NnServer.f_model_list[direction + id + 'p'])
                        #NnServer.f_model_temp[direction + id + 's'] = copy_model(
                        #    NnServer.f_model_list[direction + id + 's'])
                        NnServer.f_model_temp[direction + id + 'p'] = keras.models.load_model(
                            self.model_path + f'{id}/{direction}/' + "%s_primary.tf" % str(id))
                        NnServer.f_model_temp[direction + id + 's'] = keras.models.load_model(
                            self.model_path + f'{id}/{direction}/' + "%s_secondary.tf" % str(id))

                '''
                if fragment not recognized, or not in the simplex list thus no model and para associated, activate initialization
                '''
            else:
                initialize_model(id, num_atom, time_tag, self.temp_path, ratio=ratio)


            # begin transfer learning
            if len(NnServer.train_set[id]["zd"]) > 0:
                print(f'{id} {len(NnServer.train_set[id]["zd"])}zd:', NnServer.train_set[id]['zd'][0])
                print(f'{id} {len(NnServer.train_set[id]["e"])}e:', NnServer.train_set[id]['e'][0])
                print(f'{id} {len(NnServer.train_set[id]["xf"])}xf:', NnServer.train_set[id]['xf'][0])
            print(f'{id} {len(NnServer.transfer_temp[id]["zd"])}transfer zd:', NnServer.transfer_temp[id]['zd'][0])
            print(f'{id} {len(NnServer.transfer_temp[id]["e"])}transfer e:', NnServer.transfer_temp[id]['e'][0])
            print(f'{id} {len(NnServer.transfer_temp[id]["xf"])}transfer xf:', NnServer.transfer_temp[id]['xf'][0])

            transfer_order, transfer_order_list = transfer.main(np.array(NnServer.train_set[id]['zd']),np.array(NnServer.train_set[id]['e']),np.array(NnServer.transfer_temp[id]['zd']),
                          np.array(NnServer.transfer_temp[id]['e']), NnServer.e_model_temp, NnServer.e_model_para,
                          NnServer.radius[id],NnServer.inertia[id],id,len(NnServer.transfer_temp[id]['xf'][0]), direction='')  # energy transfer
            '''main(x_t, y_t, x, y, model_dict, param_dict, radius, avg_inertia, iD, num_atom, clustering_mode='sequential',
     partition='equal', mode='slice', inertia_m=4
     )'''
            for direction in ['x', 'y', 'z']:  # force transfer
                #transfer.main(np.array(NnServer.train_set[id][direction+'d']), np.array(NnServer.train_set[id][direction+'f']),
                #              np.array(NnServer.transfer_temp[id][direction+'d']),
                #              np.array(NnServer.transfer_temp[id][direction+'f']), NnServer.f_model_temp, NnServer.f_model_para,
                #              NnServer.radius[id], NnServer.inertia[id], id, len(NnServer.train_set[id]['xf'][0]), direction=direction)
                transfer.main_with_order(np.array(NnServer.train_set[id][direction+'d']), np.array(NnServer.train_set[id][direction+'f']),
                              np.array(NnServer.transfer_temp[id][direction+'d']),
                              np.array(NnServer.transfer_temp[id][direction+'f']), NnServer.f_model_temp, NnServer.f_model_para,
                              id, len(NnServer.transfer_temp[id]['xf'][0]), transfer_order_list, direction=direction)

            # update models, training set span, simplex list if not exist
            with NnServer._lock:
                # for i in range(len(NnServer.transfer_temp[id])):
                    #NnServer.e_train_set[id].append(NnServer.transfer_temp[id][i][0][2])
                    # {id: [((x_dist_list, y_dist_list, z_dist_list), (F_x, F_y, F_z), E1), ()]}

                #save model and transfer info
                save_model_info(time_tag, transfer_order, round(time.time() - time_, 2),id,
                                NnServer.transfer_temp[id]['zd'], NnServer.transfer_temp[id]['e'], self.temp_path
                                , NnServer.tree, NnServer.raw_tree)


                #update e train set for classification
                #{id: {xd:[], yd:[], zd:[], xf:[], yf:[], zf:[], e:[]}}
                NnServer.e_train_set[id].extend([NnServer.transfer_temp[id]['zd'][i] for i in transfer_order])
                print(f"classifier x {id}: length {len(NnServer.e_train_set[id])}")
                sample = NnServer.e_train_set[id][0]
                print(f"first: {sample}")
                if f"z{id}p" in NnServer.f_model_list.keys():
                    print(f'fz p prediction: {NnServer.f_model_list[f"z{id}p"](np.array(sample).reshape([1,len(sample)]))}')
                print(f"last: {NnServer.e_train_set[id][-1]},{NnServer.transfer_temp[id]['zd'][transfer_order[-1]]}")
                del NnServer.tree[id]
                NnServer.tree[id] =  spatial.KDTree(NnServer.e_train_set[id])

                NnServer.train_set[id]['xd'].extend([NnServer.transfer_temp[id]['xd'][i] for i in transfer_order])
                NnServer.train_set[id]['yd'].extend([NnServer.transfer_temp[id]['yd'][i] for i in transfer_order])
                NnServer.train_set[id]['zd'].extend([NnServer.transfer_temp[id]['zd'][i] for i in transfer_order])
                NnServer.train_set[id]['xf'].extend([NnServer.transfer_temp[id]['xf'][i] for i in transfer_order])
                NnServer.train_set[id]['yf'].extend([NnServer.transfer_temp[id]['yf'][i] for i in transfer_order])
                NnServer.train_set[id]['zf'].extend([NnServer.transfer_temp[id]['zf'][i] for i in transfer_order])
                NnServer.train_set[id]['e'].extend([NnServer.transfer_temp[id]['e'][i] for i in transfer_order])
                tran_size = len(NnServer.transfer_temp[id]['xd'])
                del NnServer.transfer_temp[id]


                if id + 'p' in NnServer.e_model_list.keys():
                    del NnServer.e_model_list[id + 'p']
                    del NnServer.e_model_list[id + 's']
                #NnServer.e_model_list[id + 'p'] = copy_model(NnServer.e_model_temp[id + 'p'])
                NnServer.e_model_list[id + 'p'] = keras.models.load_model(self.temp_path+f'{id}/{time_tag}/' +
                                                                          "%s_primary.tf" % str(id))

                #NnServer.e_model_list[id + 's'] = copy_model(NnServer.e_model_temp[id + 's'])
                NnServer.e_model_list[id + 's'] = keras.models.load_model(self.temp_path+f'{id}/{time_tag}/' +
                                                                          "%s_secondary.tf" % str(id))

                for direction in ['x', 'y', 'z']:
                    if direction + id + 'p' in NnServer.f_model_list.keys():
                        del NnServer.f_model_list[direction + id + 'p']
                        del NnServer.f_model_list[direction + id + 's']
                    #NnServer.f_model_list[direction + id + 'p'] = copy_model(
                    #    NnServer.f_model_temp[direction + id + 'p'])
                    NnServer.f_model_list[direction+id + 'p'] = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/{direction}/' +
                                                                              "%s_primary.tf" % str(id))

                    NnServer.f_model_list[direction+id + 's'] = keras.models.load_model(self.temp_path + f'{id}/{time_tag}/{direction}/' +
                                                                              "%s_secondary.tf" % str(id))
                    #NnServer.f_model_list[direction + id + 's'] = copy_model(
                    #    NnServer.f_model_temp[direction + id + 's'])
                gc.collect()

                if id not in NnServer.sim_list.keys():
                    print(f'{id} transfer size, train size: ',len(NnServer.transfer_set[id]["xd"]),tran_size)
                    NnServer.n_c = NnServer.n_c + len(NnServer.transfer_set[id]["xd"])+tran_size
                    NnServer.sim_list[id] = [0, len(NnServer.transfer_set[id]["xd"])+tran_size]
                NnServer.on_training[id] = False
                print(f'transfer finished on {time.ctime()} for {id}')



        else:
            NnServer._lock.release()
       
    @Pyro4.oneway
    def clear_status(self):
        NnServer.n_p = 0
        NnServer.n_c = 0
        for i in NnServer.sim_list.keys():
            NnServer.sim_list[i] = [0,0]
 
    @Pyro4.oneway
    def shutdown(self):
        NnServer.stop_event.set()
        #NnServer.thread0.exit()
        NnServer.thread0.join()
        status = get_stat()
        with open(log_path+'report.txt','w') as f:
            f.write(f'Printed time: {time.ctime()}\n')
            f.write(f'Total number of predictions: {NnServer.n_p}\n')
            f.write(f'Total number of calculations: {NnServer.n_c}\n')
            f.write(status)
            print('Server shutting down.')
        global running
        running = False


def initialize_model(id, num_atom, time_tag, temp_p, ratio=1):
    '''
    build_model(x_0, target_type, num_atom, num_node_ratio, num_layer, activation=act_gaussian()): return e_model
    def_normalization(x, y, target_type, xshift=0, x_has_min=False): return x_range, x_min, y_range, y_min, e_range, e_min

    x_range = param_dict[target_direction+"%s"%str(fragment_name)+"px"][0]
    xmin =param_dict[target_direction+"%s"%str(fragment_name)+"px"][1]

    np.savetxt(file_save_path + "%s_p_x.txt" % str(frag_name), np.concatenate([[x_range],[xmin]],axis=0))
    np.savetxt(file_save_path + "%s_p_y.txt" % str(frag_name), np.concatenate([[y_range],[ymin]],axis=0))
    '''
    #NnServer.transfer_set
    print(f'Model initialization :{id}')
    print(f'number of atom :{num_atom}')
    print(f"number of feature :{len(NnServer.transfer_temp[id]['zd'][0])}")
    print(f"number of samples :{len(NnServer.transfer_temp[id]['zd'])}\n")

    NnServer.train_set[id] = dict({'xd': [], 'yd': [], 'zd': [], 'xf': [], 'yf': [],
                                      'zf': [], 'e': []})

    NnServer.e_model_temp[id+'p'] = model_init.build_model(NnServer.transfer_temp[id]['zd'][0], 'energy', num_atom, 4, 4)
    NnServer.e_model_temp[id+'s'] = model_init.build_model(NnServer.transfer_temp[id]['zd'][0], 'energy', num_atom, 4, 4)
    x_range, x_min, y_range, y_min, e_range, e_min = model_init.def_normalization(np.array(NnServer.transfer_temp[id]['zd']), np.array(NnServer.transfer_temp[id]['e']), 'energy')
    NnServer.e_model_para[id+'px'] = [x_range, x_min]
    NnServer.e_model_para[id + 'py'] = [y_range, y_min]
    NnServer.e_model_para[id + 'sy'] = [e_range, e_min]

    temp_path = temp_p+f'{id}/{time_tag}/init/'
    path = Path(temp_path)
    save_para(id, x_range, x_min, y_range, y_min, e_range, e_min, path, temp_path)

    NnServer.e_train_set[id] = []
    NnServer.tree[id] = spatial.KDTree(NnServer.transfer_temp[id]['zd'])
    NnServer.raw_tree[id] = spatial.KDTree(NnServer.transfer_temp[id]['zd'])

    if num_atom > 4:
        NnServer.radius[id] = 0.03*num_atom*ratio
        NnServer.inertia[id] = 0.01 * NnServer.radius[id] * math.sqrt(len(NnServer.transfer_temp[id]['zd'][0]))*ratio
    elif num_atom >= 3:
        NnServer.radius[id] = (0.03*num_atom-0.08)*ratio
        NnServer.inertia[id] = (0.01 * NnServer.radius[id] * math.sqrt(len(NnServer.transfer_temp[id]['zd'][0]))-0.00017)*ratio

    if num_atom <=2 or NnServer.radius[id] <0 or NnServer.inertia[id]<0:
        print(f'{id}, ', f'number of atom {num_atom}, ',
                                          f'radius: {NnServer.radius[id]}, ', f'inertia: {NnServer.inertia[id]}, ')
        with open(temp_path+'error.txt','w') as f:
            f.write(f'{id}, number of atom {num_atom}, radius: {NnServer.radius[id]}, inertia: {NnServer.inertia[id]}')
        exit()



    for direction in ['x', 'y', 'z']:
        NnServer.f_model_temp[direction+id + 'p'] = model_init.build_model(NnServer.transfer_temp[id][f'{direction}d'][0], 'force', num_atom,
                                                                 6, 4)
        NnServer.f_model_temp[direction+id + 's'] = model_init.build_model(NnServer.transfer_temp[id][f'{direction}d'][0], 'force', num_atom,
                                                                 6, 4)
        x_range, x_min, y_range, y_min, e_range, e_min = model_init.def_normalization(np.array(NnServer.transfer_temp[id][f'{direction}d']),
                                                                                      np.array(NnServer.transfer_temp[id][f'{direction}f']),
                                                                                      'force', )

        NnServer.f_model_para[direction+id + 'px'] = [x_range, x_min]
        NnServer.f_model_para[direction+id + 'py'] = [y_range, y_min]
        NnServer.f_model_para[direction+id + 'sy'] = [e_range, e_min]

        #save_fpara(x_range, x_min, y_range, y_min, e_range, e_min)
        path = Path(temp_path+f'{direction}/')
        save_para(id, x_range, x_min, y_range, y_min, e_range, e_min, path, temp_path)
    return


def save_para(id, x_range, x_min, y_range, y_min, e_range, e_min, path, temp_path):
    path.mkdir(parents=True, exist_ok=True)
    np.savetxt(temp_path + f"{id}_p_x.txt", np.concatenate([[x_range], [x_min]], axis=0))
    np.savetxt(temp_path + f"{id}_p_y.txt", np.concatenate([[y_range], [y_min]], axis=0))
    np.savetxt(temp_path + f"{id}_s_y.txt", np.concatenate([[e_range], [e_min]], axis=0))
    return


def save_model_info(time_tag, transfer_order, time_spent, id, transfer_temp, transfer_e, temp_p, tree, raw_tree):
    temp_path = temp_p+f'{id}/{time_tag}/'

    path = Path(temp_path)
    path.mkdir(parents=True, exist_ok=True)
    NnServer.e_model_temp["%s" % str(id) + "p"].save(temp_path + "%s_primary.tf" % str(id), overwrite=True)
    NnServer.e_model_temp["%s" % str(id) + "s"].save(temp_path + "%s_secondary.tf" % str(id), overwrite=True)

    for direction in ['x', 'y', 'z']:
        path2 = Path(temp_path+direction+'/')
        path2.mkdir(parents=True, exist_ok=True)
        NnServer.f_model_temp[direction+"%s" % str(id) + "p"].save(temp_path+direction + "/%s_primary.tf" % str(id), overwrite=True)
        NnServer.f_model_temp[direction+"%s" % str(id) + "s"].save(temp_path+direction + "/%s_secondary.tf" % str(id), overwrite=True)

    with open(temp_path + 'transfer_info.txt', "w") as f:
        f.write(f'Time: {time.ctime()}')
        f.write(f'train set size: {len(NnServer.train_set[id]["xd"])}\n')
        f.write(f'transfer set size: {len(NnServer.transfer_temp[id]["xd"])}\n')
        f.write(f'transfer used size: {len(transfer_order)}\n')
        f.write(f'transfer wall time spend: {time_spent}s\n')

        dist0 = raw_tree[id].query(transfer_temp)[0]
        dist1 = tree[id].query(transfer_temp)[0]
        f.write(f'average closest training point distance: {np.mean(dist1)}\n')
        f.write(f'average closest latest training point distance: {np.mean(dist0)}\n')
        f.write(f"if all latest distance <= raw: {(dist1>=dist0).all()}\n")
        f.write(f'transfer order: {transfer_order}\n')

    np.savetxt(temp_path+'transfer_xd.txt', transfer_temp)
    np.savetxt(temp_path+'transfer_e.txt', transfer_e)
    return


def copy_model(model):
    #new_dict = {}
    #for k, model in model_list.items():
    new = keras.models.clone_model(model)
    new.build(model.input_shape)
    new.set_weights(model.get_weights())
        #new_dict[k] = new
    return new #new_dict

def run_transfer(e_model, f_model):
    return

def __classify__(xyz, tree_o, radius):
    #tree_o = spatial.KDTree(train_set)
    dist_o, order_o = tree_o.query(xyz)
    predictable = dist_o <= radius
    return predictable


def predict_energy(xyz, partition_index,model_dict, param_dict, model_name):
    
    p_dis = np.array(xyz)
    #print(3.1)
        ####predict energy from models for 1 simplex
    x_range = param_dict["%s"%str(model_name)+"px"][0]
    xmin = param_dict["%s"%str(model_name)+"px"][1]
    p_dis = ((p_dis-xmin)/x_range).reshape(1,np.size(p_dis))
        #print(p_dis)
    y_range = param_dict["%s"%str(model_name)+"py"][0]
    y_min = param_dict["%s"%str(model_name)+"py"][1]

    e_range = param_dict["%s"%str(model_name)+"sy"][0]
    e_min = param_dict["%s"%str(model_name)+"sy"][1]
    #print(3.2)
    p_e = model_dict["%s"%str(model_name)+"p"](p_dis)*y_range+y_min+\
              (model_dict["%s"%str(model_name)+"s"](p_dis)*e_range+e_min)/627.503
    #print(3.3)
    return p_e.numpy()

def predict_force(atomic_num, xyz, partition_index,model_dict, param_dict, fragment_name_input,real=None):
    #print(4.01)
    direction = ['x','y','z']
    predicted_energy_list = np.zeros(len(partition_index),dtype='object')
    n=0
    #print(4.02)
    for p in partition_index: ### p every simplex
        #print(4.021)
        num_atom = len(p)
        num_dis = int(num_atom * (num_atom - 1) / 2)
        p_dis = np.zeros(num_dis)
        #print(4.022)
        #print(p,atomic_num, xyz)
        partition_ar = np.array(atomic_num)[p]
        #print(4.023)
        xyz_t = np.array(xyz)[p]
        #print(4.03)
        """force related transformation specifically"""
        force = np.zeros([len(partition_ar),3])
        ref = np.ones([len(partition_ar),3])*3
        atomic_mass = atom_mass_mapping(partition_ar)
        '''need a global permutation information'''
        reconstructed_force = np.zeros([3,len(partition_ar)])
        #print(4.1)
        for c,ii in enumerate(fragment_transform.global_transform_fragment(xyz_t, force, partition_ar,
                                                                                 atomic_mass, ref)):
            #print(4.2)
            xyz_tt, force, partition_ar, v,order = ii
            target_direction=direction[c]
            p_dis=p_dis.reshape(-1)
            dis_matrix = distance.cdist(xyz_tt, xyz_tt)
            k = 0
            for i in range(len(dis_matrix) - 1):
                for j in range(i, len(dis_matrix) - 1):
                    p_dis[k] = dis_matrix[i][j + 1]
                    k += 1

            if fragment_name_input is None:
                fragment_name=str(num_atom)
            else:
                fragment_name=fragment_name_input
            #print(4.3)
            ####predict energy from models for 1 simplex
            x_range = param_dict[target_direction+"%s"%str(fragment_name)+"px"][0]
            xmin =param_dict[target_direction+"%s"%str(fragment_name)+"px"][1]
            p_dis = ((p_dis-xmin)/x_range).reshape(1,num_dis)

            y_range =param_dict[target_direction+"%s"%str(fragment_name)+"py"][0]
            y_min = param_dict[target_direction+"%s"%str(fragment_name)+"py"][1]

            e_range = param_dict[target_direction+"%s"%str(fragment_name)+"sy"][0]
            e_min = param_dict[target_direction+"%s"%str(fragment_name)+"sy"][1]
            #print(4.4)
            p_e = model_dict[target_direction+"%s"%str(fragment_name)+"p"](p_dis)*y_range+y_min+\
                  (model_dict[target_direction+"%s"%str(fragment_name)+"s"](p_dis)*e_range+e_min)/627.503
            reconstructed_force[c] = p_e[0].numpy()[np.argsort(order)]

        inv_v = np.linalg.inv(v)

        dir_ord = direction_order(target_direction)
        predicted_energy_list[n]=np.matmul(inv_v,reconstructed_force).T
        n+=1
        #print(4.5)
    return predicted_energy_list


def get_stat():
    training=''        
    for i in NnServer.on_training.keys():
        if NnServer.on_training[i] == True:
            training = training + i +', '

    status = f'Server status at {time.ctime()}:\nModels loaded: {NnServer.sim_list.keys()}\nModels on training: {training}\nTotal prediction: {NnServer.n_p}\nTotal not predicted: {NnServer.n_c}\n\nModel predictability info (predictable, non-predictable):\n'
    for idd in NnServer.sim_list.keys():
        status = status + f'{idd}: ' + str(NnServer.sim_list[idd]) + '\n'
    status += '\n'

    return status



def get_id(atn):
    s = ''
    for i in atn:
        s=s+f'{i}-'
    return s


def get_name(id):
    atn = np.array((id+'0').split('-')).astype(int)
    name=''
    atnum = dict({0:'',1:'H', 6:"C", 7:"N", 8: "O"})
    for i in range(1,np.max(atn)+1):
        count = np.sum(atn == i)
        if count > 0:
            name += atnum[i]
            name += str(count)
    return name



def load_energy_model(num_atom_list,model_path_root):
    tf.get_logger().setLevel('ERROR')
    model_dict = dict({})
    param_dict = dict({})
    for ll in num_atom_list:
        l=ll#str(ll.count('-'))
        model_path = model_path_root +str(ll)+'/'
        if os.path.exists(model_path+"%s_primary.h5"%str(l)):
            form='.h5'
        elif os.path.exists(model_path+"%s_primary.tf"%str(l)):
            form ='.tf'
        else:
            #print('Error: energy model file for %s not found.'%num_atom_list)
            #return {"%s"%str(ll)+"px":'error'}
            #print(coords)
            #exit()
            pass
        model = keras.models.load_model(model_path+"%s_primary"%str(l)+form)

        model_dict["%s"%str(ll)+"p"] = model
        model = keras.models.load_model(model_path + "%s_secondary" % str(l)+form)

        model_dict["%s"%str(ll)+"s"] = model
        param_dict["%s"%str(ll)+"px"] = np.loadtxt(model_path+"%s_p_x.txt"%str(l))
        param_dict["%s" % str(ll) + "py"] = np.loadtxt(model_path + "%s_p_y.txt"%str(l))
        param_dict["%s" % str(ll) + "sy"] = np.loadtxt(model_path + "%s_s_y.txt"%str(l))
        print(f'{ll} energy model loaded.')
    return model_dict, param_dict


def load_force_model(frag_list,target_direction,model_path_root):
    tf.get_logger().setLevel('ERROR')
    model_dict = dict({})
    param_dict = dict({})
    for ll in frag_list:
        l=ll#str(ll.count('-'))
        #model_path = model_path_root + str(l) + '/' + target_direction + ' 1000/'
        model_path = model_path_root +str(ll)+'/'+ target_direction + '/'
        if os.path.exists(model_path+"%s_primary.h5"%str(l)):
            format_type='.h5'
        elif os.path.exists(model_path+"%s_primary.tf"%str(l)):
            format_type='.tf'
        else:
            print('Model not found. '+ model_path+"%s_primary"%str(l))
            #exit()
        #print('Loading models from '+ model_path)
        model = keras.models.load_model(model_path+"%s_primary"%str(l)+format_type)
        model_dict[target_direction+"%s"%str(ll)+"p"] = model
        model = keras.models.load_model(model_path + "%s_secondary" % str(l)+format_type)
        model_dict[target_direction+"%s"%str(ll)+"s"] = model
        param_dict[target_direction+"%s"%str(ll)+"px"] = np.loadtxt(model_path+"%s_p_x.txt"%str(l))
        param_dict[target_direction+"%s" % str(ll) + "py"] = np.loadtxt(model_path + "%s_p_y.txt"%str(l))
        param_dict[target_direction+"%s" % str(ll) + "sy"] = np.loadtxt(model_path + "%s_s_y.txt"%str(l))
        print(f'{ll} {target_direction} force model loaded.')
    return model_dict, param_dict


def launch(config, pw = b"xiao", conn_f = '/N/project/sico/nn_server/connection.txt', log_f = '/N/project/sico/nn_server/log.txt'):

    host=socket.gethostname()
    ip=Pyro4.socketutil.getIpAddress(host)

    daemon = Pyro4.Daemon(host=ip)
    daemon._pyroHmacKey = pw

    uri = daemon.register(NnServer(config))
    with open(conn_f,'w') as f:
        f.write(str(uri)+'\n')
        f.write(ip+'\n')
        f.write(host+'\n')
        f.write(time.ctime()+'\n')
    global running
    print("Ready for connections.")
    daemon.requestLoop(loopCondition=lambda:running)


if __name__ == "__main__":
    import sys
    try:
        import tomllib as toml
    except:
        import toml
    
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    os.chdir(script_dir)
    with open("server.toml","rb") as f:
        config = toml.load(f)
    launch(config, pw=config["server"]["password"], conn_f=config["server"]["connection_file"], log_f = config["server"]["log_file"])
