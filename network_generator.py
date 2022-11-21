import itertools, multiprocessing, argparse
import numpy as np
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import AllChem
import selection


def calc_fgp_map(smiles_seq, radius=2):
    fgp_map = dict()
    for smiles in smiles_seq:
        mol = Chem.MolFromSmiles(smiles)
        fgp = AllChem.GetMorganFingerprint(mol, radius)
        for hash_of_fragment in fgp.GetNonzeroElements():
            if hash_of_fragment not in fgp_map:
                fgp_map[hash_of_fragment] = len(fgp_map)
    return fgp_map


def calc_size(smiles_seq):
    sizes = []
    for smiles in smiles_seq:
        mol = Chem.MolFromSmiles(smiles)
        size = mol.GetNumHeavyAtoms()
        sizes.append(size)
    return sizes


def calc_sparse_vector(smiles_seq, radius=2):
    # max_value = max(FGP_MAP_DICT)
    max_value = max(calc_fgp_map(smiles_seq))
    matrix = np.zeros((len(smiles_seq), max_value))
    for smiles_id, smiles in enumerate(smiles_seq):
        mol = Chem.MolFromSmiles(smiles)
        fgp = AllChem.GetMorganFingerprint(mol, radius)
        for hash_of_fragment, num_of_fragments in fgp.GetNonzeroElements().items():
            matrix[smiles_id, hash_of_fragment] = num_of_fragments
    sparse_matrix = sparse.csr_matrix(matrix)
    return sparse_matrix


##############################################################################################################
#                    ! ! !     R E A D       M E      ! ! !
# before running the code you need to provided following data and also adjust calc_synthetic_generation()
# i.e. properly define listed below global variables
BUYABLE_SMILES = []  # numpy array of smiles of buyable compounds
FGP_MAP_DICT = calc_fgp_map(BUYABLE_SMILES)       # dont need to be changed when BUYABLE_SMILES is correctly defined
BUYABLE_SIZES = calc_size(BUYABLE_SMILES)         # as above
BUYABLE_SMILES_SET = set(BUYABLE_SMILES)          # as above
BUYABLE_MTX = calc_sparse_vector(BUYABLE_SMILES)  # as above
RX_DATABASE = []  # list of reactions, reaction should be provided as Rdkit's ChemicalReaction object

# Please remember defined listed above global variable before running the code !!!
# ############################################################################################################


def sparse_tanimoto(query, sparse_table):
    result = sparse.csr_matrix(sparse_table.shape)
    result.indices = sparse_table.indices
    result.indptr = sparse_table.indptr
    result.data = np.min([sparse_table.data, np.take(query, sparse_table.indices)], axis=0)
    result = result.sum(axis=1)
    query_sum = query.sum()
    all_elements = query_sum + sparse_table.sum(axis=1) - result
    result = result / all_elements
    return np.array(result).reshape(-1)


class SparseTanimotoServer():
    def __init__(self, n_proc):
        # initialize queues
        self.n_proc = n_proc
        self.processes = []
        self.fragments = []

    def initialize(self, base):
        if len(self.processes) != 0:
            self.close_processes()
        N = base.shape[0]
        self.qin = multiprocessing.Queue()
        self.qout = multiprocessing.Queue()
        self.batch_size = int(N / self.n_proc)
        self.processes = []
        self.fragments = []
        # initialize workers
        for i in range(self.n_proc):
            if i < self.n_proc - 1:
                beg_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                fragment = base[beg_idx:end_idx]
            else:
                fragment = base[i * self.batch_size:]
            self.fragments.append(fragment)

            def ith_target_function(qin, qout):
                while True:
                    query = qin.get()
                    res = sparse_tanimoto(query, self.fragments[i])
                    qout.put((i, res))

            p = multiprocessing.Process(target=ith_target_function, args=(self.qin, self.qout))
            p.start()
            self.processes.append(p)

    def calc_sim(self, query):
        # sumbitt calculation
        for i in range(self.n_proc):
            self.qin.put(query)
        # collect data
        results = []
        for i in range(self.n_proc):
            results.append(self.qout.get())
        results.sort(key=lambda x: x[0])  # assure correct order of data
        results = np.hstack([x[1] for x in results])
        return results

    def close_processes(self):
        # kill workers
        for i in range(self.n_proc):
            self.processes[i].kill()
            self.processes[i].join()
        del self.processes, self.fragments, self.qin, self.qout
        self.processes = []
        self.fragments = []


def select_reactions(mol_list, reaction_list=RX_DATABASE, substrate_avoid_id_list=None, only_first=False):
    '''
    Selects reactions from reaction_list that match at least one mol from mol_list (taken as substrate).
    * substrate_avoid_id_list constrols which of substrate patterns cannot be used
    * check is 'lazy' - only first template match included
    * if only_first==True, only the first matching reaction is returned (the rest is not checked)
    Returns list of reactions and list of substrate ids (which of multiple templates was matching)'''

    result_reactions, result_substrate_ids = [], []

    for i, rx in enumerate(reaction_list):
        avoid = [] if substrate_avoid_id_list is None else substrate_avoid_id_list[i]
        N_templates = rx.rxn.GetNumReactantTemplates()

        is_matching = False
        for template_id in range(N_templates):
            if template_id in avoid:
                continue
            template = rx.rxn.GetReactantTemplate(template_id)
            is_matching = any(check_has_unmapped_match(mol, template) for mol in mol_list)
            if is_matching:
                result_reactions.append(rx)
                result_substrate_ids.append(avoid + [template_id])
                break
        if is_matching and only_first:
            return result_reactions, result_substrate_ids
    return result_reactions, result_substrate_ids


def generate_combinations_to_check(substrates, no3c=False, no4c=False):
    to_check = {'1c': [(smi, ) for smi in substrates], '2c': [], '3c': [], '4c': []}
    if len(substrates) >= 2:
        to_check['2c'] = list(itertools.combinations(substrates, 2))
    if len(substrates) >= 3 and not no3c:
        to_check['3c'] = list(itertools.combinations(substrates, 3))
    if len(substrates) >= 4 and not no4c:
        to_check['4c'] = list(itertools.combinations(substrates, 4))
    return to_check


def fgp_to_vector(fgp, dict_map):
    vector = np.zeros(len(dict_map))
    for hash_of_fragment, num_of_fragments in fgp.GetNonzeroElements().items():
        if hash_of_fragment in dict_map:
            vector[dict_map[hash_of_fragment]] = num_of_fragments
    return vector


def smiles_to_fgp(smiles, r=2):
    m = Chem.MolFromSmiles(smiles)
    f = AllChem.GetMorganFingerprint(m, r)
    return f


def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_vector(mol)


def mol_to_vector(mol, radius=2):
    fgp = AllChem.GetMorganFingerprint(mol, radius)
    return fgp_to_vector(fgp, FGP_MAP_DICT)


def check_has_unmapped_match(mol, pattern):
    '''Checks whether a mol contains pattern, assuming that match has to contain non-mapped atom'''
    for match in mol.GetSubstructMatches(pattern):
        for x in match:
            if mol.GetAtomWithIdx(x).GetIsotope() == 0:
                return True
    return False


similarity_server = SparseTanimotoServer(10)


def prepare_mask(SIZES, SMILES, forbidden_smiles, maximum_size):
    '''Prepares list of indices for molecules 1) smaller than maximum size and 2) not present in the forbidden_smiles (string with dots).
      SIZES and SMILES should be arrays properly ordered (eg. columns from CSV table)'''
    size_mask = SIZES < maximum_size
    if forbidden_smiles != '':
        smiles_mask = ~np.isin(SMILES, forbidden_smiles.split('.'))
        size_mask = size_mask & smiles_mask
    size_mask = np.where(size_mask)[0]  # np.where(BUYABLE_SIZES<(size*1.5))[0]
    return size_mask


def update_sparse_diff(diff_fgp, new_sparse_fgp):
    diff_fgp -= np.array(new_sparse_fgp.todense()).reshape(-1)
    diff_fgp = np.where(diff_fgp > 0, diff_fgp, 0)
    return diff_fgp


def is_radical(smiles):
    m = Chem.MolFromSmiles(smiles)
    for atom in m.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


def greedy_search(query, base, natoms, th=0.0, max_search=None, forbidden_smiles=[], k=100,
                  indices_list=[], sizes=BUYABLE_SIZES, smiles=BUYABLE_SMILES):
    result = []
    if max_search is None:
        max_search = base.shape[0]
    similarity_val = 1
    n_checked = 0
    diff = query
    seen_similarities = []
    seen_clean_smiles = []
    while similarity_val > th and n_checked < max_search and natoms > -5:
        similarities = similarity_server.calc_sim(diff)
        # search through k-best hits
        kbest_indices = np.argpartition(similarities, -k)[-k:]
        kbest_indices = kbest_indices[np.argsort(similarities[kbest_indices])]
        best_idx = kbest_indices[-1]
        for i in range(k - 1, -1, -1):
            similarity_val = similarities[kbest_indices[i]]
            if indices_list != []:
                idx = indices_list[kbest_indices[i]]
            else:
                idx = kbest_indices[i]
            smi = Chem.CanonSmiles(smiles[idx], useChiral=False)
            curr_size = sizes[idx]
            if (smi not in forbidden_smiles) and (not is_radical(smi)):
                best_idx = idx
                break
        if similarity_val <= th:
            break
        natoms -= curr_size
        result.append(best_idx)
        seen_similarities.append(similarity_val)
        seen_clean_smiles.append(smi)
        diff = update_sparse_diff(diff, base[best_idx])
        n_checked += 1
    return result, seen_clean_smiles


def propose_substrates_for_target(target_smiles, forbidden_smiles_dotted='', num_complements=3, rSize=False):
    TH = 0.0
    generations = []
    # num_complements = 3
    message, error_codes = '', []

    # 1. Prepare molecules, define a 'residue' (target-scaffold)
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_vector = mol_to_vector(target_mol)
    forbidden_smiles = [Chem.CanonSmiles(target_smiles, useChiral=False)]  # do not take stereoisomers

    # 1a. make mask and limit the database
    size = target_mol.GetNumHeavyAtoms()
    if rSize:
        max_atom_ratio = float(rSize)
    else:
        if size <= 12:
            max_atom_ratio = 1.2
        elif size <= 15:
            max_atom_ratio = 0.8
        else:
            max_atom_ratio = 0.6

    size_mask = prepare_mask(BUYABLE_SIZES, BUYABLE_SMILES, forbidden_smiles_dotted, size * max_atom_ratio)
    similarity_server.initialize(BUYABLE_MTX[size_mask])

    # 3. Search for building blocks
    # see https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    similarity_vector = similarity_server.calc_sim(target_vector)
    topK = min(int((size / 2) * num_complements * 3), similarity_vector.shape[0])
    top_K_indices = np.argpartition(similarity_vector, -topK)[-topK:]
    top_K_indices = top_K_indices[np.argsort(similarity_vector[top_K_indices])]
    top_K_indices = size_mask[top_K_indices]  # translate back to global frame
    # 3c. Get first three that matches the reaction
    for idx_ in range(topK-1, -1, -1):
        idx = top_K_indices[idx_]  # we have to go backwards as the list is sorted in ascending order
        complement = BUYABLE_SMILES[idx]
        clean_smiles = Chem.CanonSmiles(BUYABLE_SMILES[idx], useChiral=False)

        # seen already?
        if clean_smiles in forbidden_smiles:
            continue
        forbidden_smiles.append(clean_smiles)  # so stereoisomers are neglected
        mol = Chem.MolFromSmiles(complement)
        # matches any reaction?
        reactions_found = select_reactions([mol])
        if len(reactions_found) == 0:
            continue

        # 3d. Make greedy search for each selected molecule
        query = update_sparse_diff(target_vector.copy(), BUYABLE_MTX[idx])
        natoms = size * 1.2 - BUYABLE_SIZES[idx]
        found, found_smi_canon = greedy_search(query, BUYABLE_MTX, natoms, max_search=int(natoms * 2), th=TH,
                                               forbidden_smiles=forbidden_smiles, indices_list=size_mask,
                                               sizes=BUYABLE_SIZES, smiles=BUYABLE_SMILES)
        forbidden_smiles.extend(found_smi_canon)
        complement += '.' + '.'.join(BUYABLE_SMILES[found])
        generations.append(complement)
        if len(generations) == num_complements:
            message += 'OK'
            break
    else:
        error_codes.append('4')
        message += 'The number of generated substrates is smaller than usually - consider clicking "Generate more"'
        # or using auxiliary reagents (so-called "toolbox")'

    similarity_server.close_processes()
    # no need to translate indices, greedy_search does this
    return '.'.join(generations), message, '.'.join(error_codes)


def calc_synthetic_generation(substrates_to_check):
    """
    Arguments:
    - substrates_to_check - dict with key '1c', '2c', '3c' and '4c'. Each of key is optional
                            '1c' - list of one element tuples with substrate smiles for one-component reactions
                            '2c' - list of tuple with two smiles, candidates for two-component reactions
                            '3c' - list of tuple with 3 smiles for 3-component reactions
                            '4c' - list of tuple with 4 smiles for 4-component reaction
    Return:
    - obtained_product - list of product smiles
    """
    obtained_product = []  # list of product smiles
    # here you need to connect forward synthetic software which generate products (when possible) for given substrates
    # please note '1c', '2c', '3c' and '4c' is only valid compination of substrates and not necessary give any product(s)
    # especially for multicomponent reaction (3c and 4c) it rather rare situation that given substrate set give any product
    return obtained_product


def perform_calculation(target_smiles, number_of_generation, w1, w2, eta, ratio):
    substrates = propose_substrates_for_target(target, eta)
    combination_to_check = generate_combinations_to_check(substrates, no3c=False, no4c=False)
    results = {'target': target_smiles, 'generations': [], 'substrates': substrates}
    args = argparse.Namespace(w1=w1, w2=w2, )
    for gen_num in range(number_of_generation - 1):  # no pruning for last generation
        obtained_product = calc_synthetic_generation(combination_to_check)
        results['current_products'] = obtained_product
        results = selection.perform_pruning(results, args)
        combination_to_check = results.pop('reactions_to_check')
    obtained_product = calc_synthetic_generation(combination_to_check)
    results['generations'].append(obtained_product)
    return results


if __name__ == "__main__":
    do_calc = False  # please, read comment below (in else block) before change this
    # ####################################################################################
    # input (variable) for calculation which need to be defined by user
    number_of_generation: int = 4  # number of generation
    w1: int = 225  # beam width in diversification phase
    w2: int = 150  # beam width in optimization phase
    eta_substrates: int = 0.60  # define how many substraes will be taken to calculation, see paper details
    # product which has this per cent of atom from target will be classified as heavy
    # when all products in beam are heavy then we switch from diversification to optimization phase
    eta_diversity: float = 0.3  # define when calculation switch from diversification to optimization phase, see paper for details
    target = 'c1ccccc1C(CN)C(=O)O'  # smiles of target

    if do_calc:
        results = perform_calculation(target, number_of_generation, w1, w2, eta_substrates, eta_diversity)
    else:
        print("this is code code cannot be run as it is, see comments in the source")
        print("Purpose of the file is to provide very verbose pseudocode, step to make it working")
        print('1. define informarmation about buyable compounds - global variable: BUYABLE_SMILES ')
        print('2. define chemical reaction database global variable: RX_DATABASE ')
        print('3. connect software which can perform in-silico forward synthesis from given substrates')
        print('   function make_synthetic_generation() ')
