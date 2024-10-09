import os
# import sys
# sys.path.append('/m/cs/work/guoy8/RouteScoring/aizynthfinder')
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.scoring.scorers import Scorer
from aizynthfinder.context.scoring import (
    StateScorer,NumberOfReactionsScorer
)

import matplotlib.pyplot as plt
from aizynthfinder.context.scoring.scorers import Scorer, PriceSumScorer, RouteCostScorer
from aizynthfinder.search.mcts import MctsNode
from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.search.mcts import MctsState
from aizynthfinder.chem import TreeMolecule
import pandas as pd


from scscore.scscore.standalone_model_numpy import SCScorer
from aizynthfinder.context.config import Configuration
from aizynthfinder.utils.type_utils import (
    Union,
    Tuple,
    Sequence,
    Iterable,
    TypeVar,
)
from aizynthfinder.chem import Molecule, RetroReaction, FixedRetroReaction
from aizynthfinder.context.stock import StockException
from aizynthfinder.utils.exceptions import ScorerException
from collections import defaultdict
from route_distances.ted.reactiontree import ReactionTreeWrapper
import json
import sys
import numpy as np
import argparse
class WeightedScorer(Scorer):

    def __init__(
        self,
        config: Configuration,
        default_cost: float = 1.0,
        not_in_stock_multiplier: int = 10,
    ) -> None:
        super().__init__(config)
        self._config: Configuration = config
        self.default_cost = default_cost
        self.not_in_stock_multiplier = not_in_stock_multiplier
        self._reverse_order = False
        self.scscorer = SCScorer()
        self.scscorer.restore('scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz')

    def __repr__(self):
        return "weighted scores"

    def _calculate_tree_costs(
        self, tree) -> dict:
        costs = {}
        stability = {}
        for mol in tree.molecules():
            if mol.smiles!=tree.root.smiles:
                if mol not in tree.leafs():
                    _, s = self.scscorer.get_score_from_smi(mol.smiles)
                    stability[mol] = s
                else:
                    try:
                        cost = self._config.stock.price(mol)
                    except StockException:
                        costs[mol] = self.default_cost
                    else:
                        costs[mol] = cost

        max_cost = max(costs.values()) if costs else self.default_cost
        # print([mol.smiles for mol in list(costs.keys())], [mol.smiles for mol in list(stability.keys())])
        return defaultdict(lambda: max_cost * self.not_in_stock_multiplier, costs), stability

    def _score_node(self, node):
        reaction_class = pd.read_csv('reaction_class_summ_20.csv')
        feasibility = 0
        reactions, nodes = node.path_to()
        ws = sum([1.1 ** (n + 1) for n in range(len(reactions))])
        # print(ws)
        i=1
        reaction_list=[]
        for reaction in reactions:
            try:
                w=1.1**i/ws
                # print(w)
                # feasibility += w*reaction_class.loc[reaction_class['reaction_class']==reaction.metadata['classification']]['rank_score'].iloc[0]
                if reaction.metadata['classification'] in list(reaction_class['reaction_class']):
                    feasibility += w*reaction_class.loc[reaction_class['reaction_class']==reaction.metadata['classification']]['rank_score'].iloc[0]
                    reaction_list.append(reaction.metadata['classification'])
                elif reaction.metadata['classification'].split(' ')[0] in list(reaction_class['reaction_class']):
                    reactionclass = reaction.metadata['classification'].split(' ')[0]
                    feasibility += w*reaction_class.loc[reaction_class['reaction_class']==reactionclass]['rank_score'].iloc[0]
                    reaction_list.append(reactionclass)
                i=i+1
            except:
                # print('mapping error, '+'reaction class: '+reaction.metadata['classification']+', '+'reaction smiles: ' + reaction.smiles)
                continue
        prices = {}
        stabilities = {}
        for node in nodes[1:]:
            for mol in node.state.mols:
                if mol in self._config.stock and mol not in prices.keys():
                    try:
                        price = self._config.stock.price(mol)
                    except StockException:
                        prices[mol] = self.default_cost
                    else:
                        prices[mol] = price
                elif mol not in self._config.stock and mol not in stabilities.keys():
                    _, s = self.scscorer.get_score_from_smi(mol.smiles)
                    stabilities[mol] = s
        # print([mol.smiles for mol in list(costs.keys())], [mol.smiles for mol in list(stabilities.keys())])
        max_cost = max(prices.values()) if prices else self.default_cost
        prices = defaultdict(lambda: max_cost * self.not_in_stock_multiplier, prices)
        price = sum([prices[mol] for mol in prices.keys()])
        stability = sum([stabilities[mol] for mol in stabilities.keys()])

        return [price,stability,feasibility,reaction_list]

    def _score_reaction_tree(self, tree):
        reaction_class = pd.read_csv('reaction_class_summ_20.csv')
        feasibility = 0
        # reactions=[reaction.metadata['classification'] for reaction in ]
        ws = sum([1.1 ** (n + 1) for n in range(len([reaction for reaction in tree.reactions()]))])
        # print(ws)
        i=1
        reaction_list=[]
        for reaction in tree.reactions():
            try:
                w=1.1**i/ws
                # print(w)
                if reaction.metadata['classification'] in list(reaction_class['reaction_class']):
                    feasibility += w*reaction_class.loc[reaction_class['reaction_class']==reaction.metadata['classification']]['rank_score'].iloc[0]
                    reaction_list.append(reaction.metadata['classification'])
                elif reaction.metadata['classification'].split(' ')[0] in list(reaction_class['reaction_class']):
                    reactionclass = reaction.metadata['classification'].split(' ')[0]
                    feasibility += w*reaction_class.loc[reaction_class['reaction_class']==reactionclass]['rank_score'].iloc[0]
                    reaction_list.append(reactionclass)
                i=i+1

            except:
                # print('mapping error'+reaction.metadata['classification'])
                continue
        leaf_costs, stability = self._calculate_tree_costs(tree)
        cost = sum([leaf_costs[leaf] for leaf in leaf_costs.keys()])
        try:
            # stability = sum([stability[leaf] for leaf in tree.leafs()])
            stability = sum([stability[leaf] for leaf in stability.keys()])
        except KeyError:
            stability = 0
        return [cost,stability,feasibility,reaction_list]

def main(args):
    filename = "finder.yml"
    finder = AiZynthFinder(configfile=filename)
    finder.stock.select("emolecules")
    finder.expansion_policy.select("uspto-nm")
    finder.filter_policy.select("uspto")
    scorer2 = WeightedScorer(finder.config)
    CostScorer = RouteCostScorer(finder.config)
    with open(f'data/{args.input_file}', 'r') as datafile:
        data = datafile.read()
    routes = json.loads(data)[:10]

    # List to store the contents of each JSON file
    # routes = {}
    # Loop through all files in the folder
    # for filename in os.listdir(folder_path):
    #     # Check if the file is a JSON file
    #     if filename.endswith('.json'):
    #         file_path = os.path.join(folder_path, filename)
    #         # Open and load the JSON file
    #         try:
    #             with open(file_path, 'r') as json_file:
    #                 data = json.load(json_file)
    #                 data = {k.replace('reaction_smiles', 'smiles'): replace_reaction_smiles(v) for k, v in data.items()}
    #                 routes[filename] = data
    #
    #         except json.JSONDecodeError as e:
    #             print(f"Error reading {filename}: {e}")
    #         except Exception as e:
    #             print(f"An error occurred with {filename}: {e}")

    distances_dict={}
    fingerprints=[]

    for n,route in enumerate(routes):
        compound = route['smiles']
        print(n, compound)
        finder.target_smiles = compound
        route_tree = ReactionTree.from_dict(route)
        cost = CostScorer(route_tree)
        features = scorer2(route_tree)
        features.append(0)
        features.append(route)
        features = [[n, cost]+features]
        fingerprints.append(finder.target_mol.fingerprint(2,64))
        distances_dict[compound] = features
        # print(distances)

    if not os.path.exists('data'):
        os.makedirs('data')
    if args.output_file == '':
        args.output_file = f"route_{len(routes)}.json"
    with open(f"data/{args.output_file}", "w") as fp:
        json.dump(distances_dict, fp)
    print(f'The routes feature file is save as data/{args.output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='jmedchem_routes_processed.json', help="Input file name")
    parser.add_argument("--output_file", type=str, default='', help="Output save file")
    args = parser.parse_args()
    main(args)

