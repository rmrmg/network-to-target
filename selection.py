import argparse, json
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

"""
Input data format
{ 'substrates': list  # list of smiles string
  'current_products': list of smiles
  'generations': list of lists of smiles, each list represents one synthetic generation
  'target': smiles
  'optimization_from': int | None  # if set defined at when (which generation) optimization phase was started
}
User doesn't need to provide 'optimization_from', this information will be added to output file when switch from
diversification to optimization takes place. The exact value is for information purpose, the code checks only
in which phase (optimization or diversification) we are.
All lists of smiles should NOT have duplicates and each smiles can be only on one list. For example
if some compound is present in 'substrate', it should be removed from 'generations' and 'current_products'.

The output format is similar to the input, but there are following differences:
1. pruned 'current_products' goes to the end of 'generations' (id. new list is appended ).
2. by default 'current_products' is removed (remains if --dont-clean-current is selected)
3. new key is added to 'reactions_to_check' with proposed substrates for 1, 2, 3, and 4 component reactions.
   The data in 'reactions_to_check' is stored as a dictionary with following keys: '1c', '2c', '3c' and '4c'.
   Proposed substrates are stored as tuples with 1, 2, 3 or 4 smiles and contain all combinations of substrates which should be
   checked by forward-synthesis software (in short FSS, e. g. Allchemy). For example tuple with 1 substrate means FSS
   need to check if the substrate can undergo one-component reactions (e.g. rearrangement, cyclisation),
   tuple with 2 substrates indicates FSS need to check if those two substrates can react with each other in
   two-component reaction (e.g. Diels-Alder, esterification, ...), tuple with 3 and 4 substrates indicates FSS
   needs to check if those 3 or 4 substrates can undergo 3 or 4 components reaction(s).
   If option --disable-3c or --disable-4c is selected substates for 3-component and 4-component reactions respectively
   will not be proposed.

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input file in json format, see code documentation for details')
    parser.add_argument('-o', '--output', type=str, required=True, help='name of output file')
    parser.add_argument('--w1', type=int, default=225, help='beam width when small compounds present (diversification phase)')
    parser.add_argument('--w2', type=int, default=150, help='beam width when only big compounds present (optimization phase)')
    parser.add_argument('--disable-3c', action='store_true', help='dont propose substrates for 3-components reactions')
    parser.add_argument('--disable-4c',  action='store_true', help='dont propose substrates for 4-components reactions')
    parser.add_argument('--dont-clean-current', action='store_true', help='raw synthetic generation is not removed from current_products')
    parser.add_argument('--fingerprint', choices=['ecfp6', 'ecfp4'], default='ecfp6', help='type of molecular fingerprint')
    parser.add_argument('--similarity', choices=['tversky0.2_0.8', 'dice'], default='tversky0.2_0.8',
                        help='type of similarity measure between fingerprint')
    parser.add_argument('--min-size-ratio', type=float, default=0.3,
                        help='size threshold decide when switch from diversification to optimization phase')
    args = parser.parse_args()
    return args


def calc_similarity(smiles, target_info, args):
    mol = Chem.MolFromSmiles(smiles)
    fp = calc_fingerprint(mol, args)
    if args.similarity == 'tversky0.2_0.8':
        simil = DataStructs.TverskySimilarity(target_info['fp'], fp, 0.2, 0.8)
    if args.similarity == 'dice':
        simil = DataStructs.DiceSimilarity(target_info['fp'], fp)
    return simil


def prune_targeted_search_products(products, target_info, beta=200, minimum_size_ratio=0.45, alfa=300):
    """
    Arguments:
        - products - list of smiles; The smiles are products of single synthetic generation.
        - target_info - dict with information about target, as returned by calc_target_info() function
        - alfa - expected number of products when NOT only "heavy molecules" present (W1). 
             Definition of "Heavy molemole": molecule with number of heavy atoms higher than:
             target_size * minimum_size_ration. Note it refers to number of heavy atoms not mass.
        - beta - expected number of products when only "heavy molecules" are retained (W2).
        - minimum_size_ratio - defines when switch between alfa and beta limit takes place (eta_diverse) (see above)
    """
    Nprods = len(products)

    if beta == 0 or (beta >= Nprods and alfa >= Nprods):
        print(f'No pruning width are {alfa} and {beta} whereas number of products is {Nprods}')
        return products, False
    assert (beta > 0) and (alfa > 0), 'alpha and beta should be positive'

    print(f'applying pruning to {Nprods} prods beam width: {alfa} and {beta}; size_ratio: {minimum_size_ratio}')
    # 1. Get partition into top-beta and below
    similarities = []
    for sml in products:
        similarities.append(calc_similarity(sml, target_info, args))

    # 2. Get minimum_size in top-beta
    min_size = target_info['num_atoms']
    partitioned_indices = None
    if beta < Nprods:
        partitioned_indices = np.argpartition(similarities, -beta)
        for idx in partitioned_indices[-beta:]:  # top-beta
            size = Chem.MolFromSmiles(products[idx]).GetNumHeavyAtoms()
            if size < min_size:
                min_size = size
    else:
        min_size = min([Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in products])
    size_th = minimum_size_ratio * target_info['num_atoms']
    print("min_size", min_size, "target_size * ratio", size_th)
    start_opt_phase = False
    prune_idx = None

    if min_size > size_th:
        # the smallest product from selected is above size_th limit - only relatively big molecules
        if beta >= Nprods:
            print(f"no w2 pruning {Nprods} products and width is {beta}")
            return products, False
        prune_idx = beta
        start_opt_phase = True
        if partitioned_indices is None:
            partitioned_indices = np.argpartition(similarities, -beta)
        print(f'pruning optimization mode beam width {beta}')
    elif Nprods > alfa:
        prune_idx = alfa
        print(f'pruning: size cryterion not met, use w1 beam width {alfa}')
        partitioned_indices = np.argpartition(similarities, -alfa)

    if prune_idx is not None:
        # 3. remove bottom_beta
        idx_to_rm = sorted(partitioned_indices[: -prune_idx], reverse=True)
        for idx in idx_to_rm:
            rm_prod = products.pop(idx)
            print('rejecting: ', idx, rm_prod, similarities[idx])
    return products, start_opt_phase


def calc_target_info(smiles, args):
    mol = Chem.MolFromSmiles(smiles)
    fp = calc_fingerprint(mol, args)
    info = {'smiles': smiles, 'mol': mol, 'fp': fp, 'num_atoms': mol.GetNumHeavyAtoms()}
    return info


def calc_fingerprint(mol, args):
    if args.fingerprint == 'ecfp6':
        fp = AllChem.GetMorganFingerprint(mol, 3, useChirality=True, useFeatures=True)
    elif args.fingerprint == 'ecfp4':
        fp = AllChem.GetMorganFingerprint(mol, 3, useChirality=True, useFeatures=True)
    return fp


def dynamic_pruning(calculation_results, target_info, args):
    products = calculation_results['current_products']
    products, start_opt_phase = prune_targeted_search_products(products, target_info, alfa=args.w1, beta=args.w2,
                                                               minimum_size_ratio=args.min_size_ratio)
    calculation_results['generations'].append(products)
    if start_opt_phase:
        is_in_opt_phase = calculation_results.get('optimization_from', 0)
        if not is_in_opt_phase:
            calculation_results['optimization_from'] = len(calculation_results['generations'])
    return calculation_results


def load_results(inpfile):
    data = json.load(open(inpfile))
    return data


def save_results(new_results, outfile):
    try:
        json.dump(new_results, open(outfile, 'w'))
    except:
        print("problem with saving results")
        return False
    return True


def get_candidate_for_reaction(new_results, args):
    reactions_to_check = {'1c': [(smi,) for smi in new_results['generations'][-1]]}
    gr2 = new_results['substrates'][:]
    if new_results.get('optimization_from', None):
        end_divers = new_results['optimization_from']
        for gen in new_results['generations'][:end_divers]:
            gr2.extend(gen)
    else:
        # diversification phase - all combination are allowed
        for gen in new_results['generations']:
            gr2.extend(gen)
    reactions_to_check['2c'] = get_2c_candidates(new_results['generations'][-1], gr2)
    if not args.disable_3c:
        reactions_to_check['3c'] = get_3c_candidates(new_results['generations'][-1], gr2, gr2)
    if not args.disable_4c:
        reactions_to_check['4c'] = get_4c_candidates(new_results['generations'][-1], gr2, gr2, gr2)
    return reactions_to_check


def get_2c_candidates(gr1, gr2):
    pairs = set()
    for smi1 in gr1:
        for smi2 in gr2:
            if smi1 == smi2:
                continue
            pairs.add(tuple(sorted([smi1, smi2])))
    return list(pairs)


def get_3c_candidates(gr1, gr2, gr3):
    gr1_2 = get_2c_candidates(gr1, gr2)
    triples = []
    for smi1_2 in gr1_2:
        for smi3 in gr3:
            if smi3 in smi1_2:
                continue
            triples.append((smi1_2[0], smi1_2[1], smi3))
    return triples


def get_4c_candidates(gr1, gr2, gr3, gr4):
    gr1_2 = get_2c_candidates(gr1, gr2)
    gr3_4 = get_2c_candidates(gr3, gr3)
    quatros = []
    for smi1_2 in gr1_2:
        for smi3_4 in gr3_4:
            setunion = set(smi1_2).union(set(smi3_4))
            if len(setunion) != 4:
                continue
            smituple = (smi1_2[0], smi1_2[1], smi3_4[0], smi3_4[1])
            quatros.append(smituple)
    return quatros


def perform_pruning(calculation_results, args):
    target_info = calc_target_info(calculation_results['target'], args)
    new_results = dynamic_pruning(calculation_results, target_info, args)
    new_results['reactions_to_check'] = get_candidate_for_reaction(new_results, args)
    if not args.dont_clean_current:
        del new_results['current_products']
    return new_results


if __name__ == "__main__":
    args = parse_args()
    calculation_results = load_results(args.input)
    new_results = perform_pruning(calculation_results, args)
    save_results(new_results, args.output)
