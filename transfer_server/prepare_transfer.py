import numpy as np
import fragment_transform
from scipy.spatial import distance


def load_force_input(frag_name, file_path):
    train_dict = dict({})
    for direction in ['x', 'y', 'z']:
        for ty in ['d','f']:
            train_dict[direction + ty] = []
    atom_num = np.loadtxt(file_path+f'{frag_name}atn.txt')
    num_atom = len(atom_num[0])

    xyz_a = np.loadtxt(file_path+f'{frag_name}_train_xyz.txt').reshape(-1)
    xyz_a = xyz_a.reshape([int(len(xyz_a)/3/num_atom),num_atom,3])

    force = np.loadtxt(file_path+f'{frag_name}_f.txt').reshape(-1)
    force = force.reshape([int(len(force)/3/num_atom),num_atom,3])

    train_dict['e'] = np.loadtxt(file_path+f'{frag_name}_e.txt').reshape(-1).tolist()


    for n in range(len(atom_num)):
        #dict_temp = dict({})
        for direction in ['x', 'y', 'z']:
            dis_list, force_t = get_force_io(xyz_a[n], atom_num[n], num_atom, force[n], direction)
            #dict_temp[direction+'d'], dict_temp[direction+'f'] = dis_list, force_t
            #train_dict[frag_name].append(((dict_temp['xd'], dict_temp['yd'], dict_temp['zd']),
            #                          (dict_temp['xf'], dict_temp['yf'], dict_temp['zf']), e[n]))
            train_dict[direction+'d'].append(dis_list)
            train_dict[direction+'f'].append(force_t)
    return train_dict

def get_force_io(xyz_a, atom_num, num_atom, force, target_direction):
    num_dis = num_atom * (num_atom - 1) / 2
    dis_list = np.zeros([int(num_dis)])

    '''reference geometry setup'''
    # ref  =np.zeros([num_atom,3])
    # n=0
    # atomic_numbers, xyz = atom_num[n], xyz_a[n]
    # atomic_mass = atom_mass_mapping(atomic_numbers)
    # xyz = xyz[np.argsort(atomic_numbers)]  #### sort all coordinate by increasing atomic number
    ##xyz, force_discard, atomic_numbers, v = fragment_transform.transform_fragment(xyz, force[n], atomic_numbers, atomic_mass,
    ##                                                                         target_direction, ref)
    ##ref=np.array(xyz)
    ref = np.ones([num_atom, 3]) * 3
    ''''''

    atomic_numbers, xyz = atom_num, xyz_a
    atomic_mass = fragment_transform.atom_mass_mapping(atomic_numbers)
    # xyz = xyz[np.argsort(atomic_numbers)]  #### sort all coordinate by increasing atomic number
    xyz, force_t, atomic_numbers, v, order = fragment_transform.transform_fragment(xyz, np.asarray(force), atomic_numbers,
                                                                                    atomic_mass, target_direction,
                                                                                    ref)
    #### simple sorting dis list
    dis_matrix = distance.cdist(xyz, xyz)
    # if (n ==0 or n==1or n==2 or n==12) and False:
    #   print(dis_matrix)
    k = 0
    for i in range(len(dis_matrix) - 1):
        for j in range(i, len(dis_matrix) - 1):
            dis_list[k] = dis_matrix[i][j + 1]
            k += 1
    # v[0]=math.copysign(1,v[0,0])*v[0]
    # v[1] = math.copysign(1, v[1, 1]) * v[1]
    # v[2]=math.copysign(1,v[2,2])*v[2]
    force_t = np.array([np.matmul(v, force_t[iii]) for iii in range(num_atom)])[:,fragment_transform.direction_order(target_direction)]

    # dis_list=np.delete(dis_list,np.where(energy==0),axis=0)
    # energy = np.delete(energy,np.where(energy==0))
    return dis_list.tolist(), force_t.tolist()
