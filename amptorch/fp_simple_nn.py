import sys
import copy
import time
import os
import pickle
from pickle import load
from collections import defaultdict, OrderedDict
import shutil
import numpy as np
from ase import io
from ase.db import connect
from simple_nn.features.symmetry_function._libsymf import lib, ffi
from simple_nn.features.symmetry_function import _gen_2Darray_for_ffi

def make_amp_descriptors_simple_nn(atoms, Gs, elements, forcetraining, cores, label, save):
    """
    uses simple_nn to make descriptors in the amp format.
    Only creates the same symmetry functions for each element
    for now.
    Parameters: 
        atoms (list of ASE Atoms objects)
            a list of the atoms you'd like to make descriptors for
        Gs (Gaussian finger-printing parameters)
            a dictionary of Gaussian finger-printing parameters. Behler-Parrinello style. 
        elements (list of strings for atom types, or str "all")
            a list of atom types in provided list of atoms which you want to calculate fingerprints for. 
        forcetraining (boolean)
            if True, store fingerprint primes. 
            if False, store only fingerprints. 
        cores (int for multithreading, default = 1)
        label (str)
        save (boolean, default = True)
            if set to True, return None, None, but saved the files. 
            if set to False, return fps, and fp_primes. 
    """
    calculated = False
    if save is False and calculated is False:
        fps, fp_primes = stored_fps(atoms, Gs, forcetraining=forcetraining)
        return fps, fp_primes
    traj, calculated, cffi_out = make_simple_nn_fps(atoms, Gs, elements=elements,
            label=label)
    if calculated is True and save is False:
        fps, fp_primes = convert_simple_nn_fps(traj, Gs, cffi_out, forcetraining, cores, save=save)
        return fps, fp_primes
    else: return None, None

def make_simple_nn_fps(traj, Gs, label, elements="all"):
    """
    generates descriptors using simple_nn. The files are stored in the
    ./data folder. These descriptors will be in the simple_nn form and
    not immediately useful for other programs
    Parameters:
        traj (list of ASE atoms objects):
            a list of the atoms you'd like to make descriptors for
        descriptors (tuple):
            a tuple containing (g2_etas, g2_rs_s, g4_etas, cutoff, g4_zetas, g4_gammas)
        clean_up_directory (bool):
            if set to True, the input files made by simple_nn will
            be deleted
    returns:
        None
    """
    # handle inputs
    if type(traj) != list:
        traj = [traj]

    G = copy.deepcopy(Gs)
    traj = factorize_data(traj, G)
    calculated = False
    cffi_out = None
    if len(traj) > 0:
        # order descriptors for simple_nn
        cutoff = G["cutoff"]
        G["G2_etas"] = [a / cutoff**2 for a in G["G2_etas"]]
        G["G4_etas"] = [a / cutoff**2 for a in G["G4_etas"]]
        descriptors = (
            G["G2_etas"],
            G["G2_rs_s"],
            G["G4_etas"],
            G["cutoff"],
            G["G4_zetas"],
            G["G4_gammas"],
        )
        if elements == "all":
            atom_types = []
            # TODO rewrite this
            for image in traj:
                atom_types += image.get_chemical_symbols()
                atom_types = list(set(atom_types))
        else:
            atom_types = traj[0].get_chemical_symbols()
            atom_types = list(set(atom_types))

        params_set = make_snn_params(atom_types, *descriptors)

        # build the descriptor object
        cffi_out = defaultdict()
        for image_idx, atoms in enumerate(traj):
            x_out, dx_out = wrap_symmetry_functions(atoms, params_set)
            cffi_out[image_idx] = defaultdict()
            cffi_out_i = cffi_out[image_idx]
            cffi_out_i['x'] = x_out
            cffi_out_i['dx'] = dx_out
        calculated = True
    return traj, calculated, cffi_out

def factorize_data(traj, Gs):
    new_traj = []
    if os.path.isdir("amp-data-fingerprint-primes.ampdb/"):
        for image in traj:
            hash = get_hash(image, Gs)
            if os.path.isfile(
                "amp-data-fingerprint-primes.ampdb/loose/" + hash
            ) and os.path.isfile("amp-data-fingerprints.ampdb/loose/" + hash):
                pass
            else:
                new_traj.append(image)
    else:
        new_traj = traj
    return new_traj

def make_snn_params(
    elements, etas, rs_s, g4_eta=4, cutoff=6.5, g4_zeta=[1.0, 4.0], g4_gamma=[1, -1]
    ):
    """
    makes a params file for simple_NN. This is the file containing
    the descriptors. This function makes g2 descriptos for the eta
    and rs values that are input, and g4 descriptors that are log
    spaced between 10 ** -5 and 10 ** -1. The number of these
    that are made is controlled by the `n_g4_eta` variable
    Parameters:
        elements (list):
            a list of elements for which you'd like to make params
            files for
        etas (list):
            the eta values you'd like to use for the descriptors
        rs_s (list):
            a list corresponding to `etas` that contains the rs
            values for each descriptor
        g4_eta (int or list):
            the number of g4 descriptors you'd like to use. if a
            list is passed in the values of the list will be used
            as eta values
        cutoff (float):
            the distance in angstroms at which you'd like to cut
            off the descriptors
    returns:
        None
    """
    params_set = {}
    
    if len(etas) != len(rs_s):
        raise ValueError('the length of the etas list must be equal to the'
                         'length of the rs_s list')
    if type(g4_eta) == int:
        g4_eta = np.logspace(-4, -1, num=g4_eta)
    for element in elements:
        params = {'i':[],'d':[]}
        
        # G2
        for eta, Rs in zip(etas, rs_s):
            for species in range(1, len(elements) + 1):
                params['i'].append([2,species,0])
                params['d'].append([cutoff,eta,Rs,0.0])

        # G4
        for eta in g4_eta:
            for zeta in g4_zeta:
                for lamda in g4_gamma:
                    for i in range(1, len(elements) + 1):
                        for j in range(i, len(elements) + 1):
                            params['i'].append([4,i,j])
                            params['d'].append([cutoff,eta,zeta,lamda])
                            
                            
        params_set[element]={'num':len(params['i']),
                'i':params['i'],
                'd':params['d']}
    return params_set

def wrap_symmetry_functions(atoms, params_set):

    # Adapted from the python code in simple-nn
    x_out = {}
    dx_out = {}
    # da_out = {} # no stress calculation
    
    cart = np.copy(atoms.get_positions(wrap=True), order='C')
    scale = np.copy(atoms.get_scaled_positions(), order='C')
    cell = np.copy(atoms.cell, order='C')

    symbols = np.array(atoms.get_chemical_symbols())
    atom_num = len(symbols)
    atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
    type_num = dict()
    type_idx = dict()
    
    for j,jtem in enumerate(params_set.keys()):
        tmp = symbols==jtem
        atom_i[tmp] = j+1
        type_num[jtem] = np.sum(tmp).astype(np.int64)
        # if atom indexs are sorted by atom type,
        # indexs are sorted in this part.
        # if not, it could generate bug in training process for force training
        type_idx[jtem] = np.arange(atom_num)[tmp]

    for key in params_set:
        params_set[key]['ip']=_gen_2Darray_for_ffi(np.asarray(params_set[key]['i'], dtype=np.intc, order='C'), ffi, "int")
        params_set[key]['dp']=_gen_2Darray_for_ffi(np.asarray(params_set[key]['d'], dtype=np.float64, order='C'), ffi)
        
    atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

    cart_p  = _gen_2Darray_for_ffi(cart, ffi)
    scale_p = _gen_2Darray_for_ffi(scale, ffi)
    cell_p  = _gen_2Darray_for_ffi(cell, ffi)

    for j,jtem in enumerate(params_set.keys()):
        q = type_num[jtem]
        r = type_num[jtem] 

        cal_atoms = np.asarray(type_idx[jtem][:], dtype=np.intc, order='C')
        cal_num = len(cal_atoms)
        cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

        x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
        dx = np.zeros([cal_num, params_set[jtem]['num'] * atom_num * 3], dtype=np.float64, order='C')
        # da = np.zeros([cal_num, params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C') # no stress calculation

        x_p = _gen_2Darray_for_ffi(x, ffi)
        dx_p = _gen_2Darray_for_ffi(dx, ffi)
        # da_p = _gen_2Darray_for_ffi(da, ffi) # no stress calculation

        errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                         atom_i_p, atom_num, cal_atoms_p, cal_num, \
                         params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
                         x_p, dx_p)
                         # , da_p) # no stress calculation
                
        x_out[jtem] = np.array(x).reshape([type_num[jtem], params_set[jtem]['num']])
        dx_out[jtem] = np.array(dx).\
                                    reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
        # da_out[jtem] = np.array(da)

    return x_out, dx_out 

def convert_simple_nn_fps(traj, Gs, cffi_out, forcetraining, cores, save):

    # make the directories
    if not os.path.isdir("./amp-data-fingerprints.ampdb"):
        os.mkdir("./amp-data-fingerprints.ampdb")
    if not os.path.isdir("./amp-data-fingerprints.ampdb/loose"):
        os.mkdir("./amp-data-fingerprints.ampdb/loose")
    if forcetraining:
        if not os.path.isdir("./amp-data-fingerprint-primes.ampdb"):
            os.mkdir("./amp-data-fingerprint-primes.ampdb")
        if not os.path.isdir("./amp-data-fingerprint-primes.ampdb/loose"):
            os.mkdir("amp-data-fingerprint-primes.ampdb/loose")
    for i, image in enumerate(traj):
        x = cffi_out[i]['x']
        dx = cffi_out[i]['dx']
        im_hash = get_hash(image, Gs)
        x_list = reorganize_simple_nn_fp(image, x)
        x_der_dict = None
        if forcetraining:
            x_der_dict = reorganize_simple_nn_derivative(image, dx)
        if save:
            pickle.dump(x_list, open("./amp-data-fingerprints.ampdb/loose/" + im_hash, "wb"))
            if forcetraining:
                pickle.dump(
                    x_der_dict, open("./amp-data-fingerprint-primes.ampdb/loose/" + im_hash, "wb")
                )
    return x_list, x_der_dict

def reorganize_simple_nn_fp(image, x_dict):
    """
    reorganizes the fingerprints from simplen_nn into
    amp format
    Parameters:
        image (ASE atoms object):
            the atoms object used to make the finerprint
        x_dict (dict):
            a dictionary of the fingerprints from simple_nn
    """
    # TODO check for bugs
    # the structure is:
    # [elements][atom i][symetry function #][fp]
    fp_l = []
    sym_dict = OrderedDict()
    syms = image.get_chemical_symbols()
    for sym in syms:
        sym_dict[sym] = []
    for i, sym in enumerate(syms):
        sym_dict[sym].append(i)
    for i, sym in enumerate(syms):
        simple_nn_index = sym_dict[sym].index(i)
        fp = x_dict[sym][simple_nn_index]
        fp_l.append((sym, list(fp)))
    return fp_l

def reorganize_simple_nn_derivative(image, dx_dict):
    """
    reorganizes the fingerprint derivatives from simplen_nn into
    amp format
    Parameters:
        image (ASE atoms object):
            the atoms object used to make the finerprint
        dx_dict (dict):
            a dictionary of the fingerprint derivatives from simple_nn
    """
    # TODO check for bugs
    d = OrderedDict()
    sym_dict = OrderedDict()
    syms = image.get_chemical_symbols()
    for sym in syms:
        sym_dict[sym] = []
    for i, sym in enumerate(syms):
        sym_dict[sym].append(i)
    # the structure is:
    # [elements][atom i][symetry function #][atom j][derivitive in direction]
    for element, full_arr in dx_dict.items():
        for i, arr_t in enumerate(full_arr):
            true_i = sym_dict[element][i]
            for sf in arr_t:
                for j, dir_arr in enumerate(sf):
                    for k, derivative in enumerate(dir_arr):
                        if (j, syms[j], true_i, element, k) not in d:
                            d[(j, syms[j], true_i, element, k)] = []
                        d[(j, syms[j], true_i, element, k)].append(derivative)
    # zero_keys = []
    # for key, derivatives in d.items():
        # zero_check = [a == 0 for a in derivatives]
        # if zero_check == [True] * len(derivatives):
            # zero_keys.append(key)
    # for key in zero_keys:
        # del d[key]
    # d = OrderedDict(d)
    return d

def get_hash(atoms, Gs=None):
    import hashlib

    """Creates a unique signature for a particular ASE atoms object.
    This is used to check whether an image has been seen before. This is just
    an md5 hash of a string representation of the atoms object and symmetry
    functions.
    Parameters
    ----------
    atoms : ASE dict
        ASE atoms object.
    Returns
    -------
        Hash string key of 'atoms'.
    """
    string = str(atoms.pbc)
    try:
        flattened_cell = atoms.cell.array.flatten()
    except AttributeError:  # older ASE
        flattened_cell = atoms.cell.flatten()
    for number in flattened_cell:
        string += "%.15f" % number
    for number in atoms.get_atomic_numbers():
        string += "%3d" % number
    for number in atoms.get_positions().flatten():
        string += "%.15f" % number
    if Gs:
        gs_values = list(Gs.values())
        for number in gs_values[0]:
            string += "%.15f" % number
        for number in gs_values[1]:
            string += "%.15f" % number
        for number in gs_values[2]:
            string += "%.15f" % number
        for number in gs_values[3]:
            string += "%.15f" % number
        for number in gs_values[4]:
            string += "%.15f" % number
        string += "%.15f" % gs_values[5]

    md5 = hashlib.md5(string.encode("utf-8"))
    hash = md5.hexdigest()
    return hash

def stored_fps(traj, Gs, forcetraining):
    image_hash = get_hash(traj[0], Gs)
    with open("amp-data-fingerprints.ampdb/loose/"+image_hash, "rb") as f:
        fps = load(f)
    if forcetraining:
        with open("amp-data-fingerprint-primes.ampdb/loose/"+image_hash, "rb") as f:
            fp_primes = load(f)
    else:
        fp_primes = None
    return fps, fp_primes
