""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations
import abc
from collections import defaultdict
from collections.abc import Sequence as SequenceAbc
from typing import TYPE_CHECKING
import pandas as pd
from aizynthfinder.search.mcts import MctsNode
from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.search.mcts import MctsState
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.stock import StockException
from aizynthfinder.utils.exceptions import ScorerException
from scscore.scscore.standalone_model_numpy import SCScorer

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import (
        Union,
        Tuple,
        Sequence,
        Iterable,
        TypeVar,
    )
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.chem import Molecule, RetroReaction, FixedRetroReaction

    _Scoreable = TypeVar("_Scoreable", MctsNode, ReactionTree)
    _Scoreables = Sequence[_Scoreable]
    _ScorerItemType = Union[_Scoreables, _Scoreable]
import numpy as np

class Scorer(abc.ABC):
    """
    Abstract base class for classes that do scoring on MCTS-like nodes or reaction trees.

    The actual scoring is done be calling an instance of
    a scorer class with a ``Node`` or ``ReactionTree`` object as only argument.

    .. code-block::

        scorer = MyScorer()
        score = scorer(node1)

    You can also give a list of such objects to the scorer

    .. code-block::

        scorer = MyScorer()
        scores = scorer([node1, node2])

    :param config: the configuration the tree search
    """

    def __init__(self, config: Configuration = None) -> None:
        self._config = config
        self._reverse_order: bool = True

    def __call__(self, item: _ScorerItemType) -> Union[float, Sequence[float]]:
        if isinstance(item, SequenceAbc):
            return self._score_many(item)
        if isinstance(item, (MctsNode, ReactionTree)):
            return self._score_just_one(item)  # type: ignore
        raise ScorerException(
            f"Unable to score item from class {item.__class__.__name__}"
        )

    def sort(
        self, items: _Scoreables
    ):
            # -> Tuple[_Scoreables, Sequence[float], Sequence[int]]:
        """
        Sort nodes or reaction trees in descending order based on the score

        :param items: the items to sort
        :return: the sorted items and their scores
        """
        scores = self._score_many(items)
        # if isinstance(scores[0],float):
        #     all_score = scores
        # else:
        #     all_score = scores
        #     scores = np.array(scores)[:,:3]
        #     mean = np.mean(scores, axis=0)
        #     std = np.std(scores.astype(float), axis=0)
        #     if 0 not in std:
        #         scores = (scores - mean)/std
        #         scores = -0.4 * scores[:, 0] - 0.3 * scores[:, 1] + 0.3 * scores[:, 2]
        #         scores = scores.tolist()
        #     else:
        #         scores = -0.4 * scores[:, 0] - 0.3 * scores[:, 1] + 0.3 * scores[:, 2]
        #         scores = scores.tolist()

        assert isinstance(scores, SequenceAbc)
        sortidx = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=True
        )
        scores = [scores[idx] for idx in sortidx]
        sorted_items = [items[idx] for idx in sortidx]
        # all_score = [all_score[idx] for idx in sortidx]
        return sorted_items, scores, sortidx #, all_score

    def _score_just_one(self, item: _Scoreable) -> float:
        if isinstance(item, MctsNode):
            return self._score_node(item)
        return self._score_reaction_tree(item)

    def _score_many(self, items: _Scoreables) -> Sequence[float]:
        if isinstance(items[0], MctsNode):
            return self._score_nodes(items)
        if isinstance(items[0], ReactionTree):
            return self._score_reaction_trees(items)
        raise ScorerException(
            f"Unable to score item from class {items[0].__class__.__name__}"
        )

    @abc.abstractmethod
    def _score_node(self, node: MctsNode) -> float:
        pass

    def _score_nodes(self, nodes: _Scoreables) -> Sequence[float]:
        return [self._score_node(node) for node in nodes]

    @abc.abstractmethod
    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        pass

    def _score_reaction_trees(self, trees: _Scoreables) -> Sequence[float]:
        return [self._score_reaction_tree(tree) for tree in trees]


class StateScorer(Scorer):
    """Class for scoring nodes based on the state score"""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._config: Configuration = config

    def __repr__(self) -> str:
        return "state score"

    def _score_node(self, node: MctsNode) -> float:
        return node.state.score

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        mols = [
            TreeMolecule(
                parent=None, transform=tree.depth(leaf) // 2, smiles=leaf.smiles
            )
            for leaf in tree.leafs()
        ]
        state = MctsState(mols, self._config)
        return state.score

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
        # for mol in tree.leafs():

        for mol in tree.molecules():
            if mol in tree.leafs():
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
        return defaultdict(lambda: max_cost * self.not_in_stock_multiplier, costs), stability

    def _calculate_node_costs(
        self, leafs: Union[Sequence[Molecule], Iterable[Molecule]]
    ) -> dict:
        costs = {}
        stability = {}
        for mol,avai in zip(leafs.mols, leafs.in_stock_list):
            if avai:
                try:
                    cost = self._config.stock.price(mol)
                except StockException:
                    costs[mol] = self.default_cost
                else:
                    costs[mol] = cost
            else:
                # continue
                _,s = self.scscorer.get_score_from_smi(mol.smiles)
                stability[mol] = s

        max_cost = max(costs.values()) if costs else self.default_cost
        return defaultdict(lambda: max_cost * self.not_in_stock_multiplier, costs), stability

    def _score_node(self, node):
        reaction_class = pd.read_csv('reaction_class_summ_20.csv')
        feasibility = 0
        reactions=node.actions_to()
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

        leaf_costs,stability = self._calculate_node_costs(node.state)
        # leaf_costs,stability = self._calculate_tree_costs(node.to_reaction_tree())
        cost = sum(leaf_costs[mol] for mol,avai in zip(node.state.mols, node.state.in_stock_list) if avai)
        stability = sum(stability[mol] for mol,avai in zip(node.state.mols, node.state.in_stock_list) if not avai)

        scores = 0.3*stability-0.3*feasibility-0.4*cost
        # score = cost+stability+feasibility
        return [cost,stability,feasibility,reaction_list]

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

        cost = sum(leaf_costs[leaf] for leaf in tree.molecules() if leaf not in tree.leafs())
        try:
            stability = sum(stability[leaf] for leaf in tree.leafs())
        except KeyError:
            stability = 0
        # score = cost+stability+feasibility
        # scores = 0.3 * stability - 0.3 * feasibility - 0.4 * cost
        return [cost,stability,feasibility,reaction_list]

class NumberOfReactionsScorer(Scorer):
    """Class for scoring nodes based on the number of reaction it took to get to a node"""

    def __init__(self, config: Configuration = None) -> None:
        super().__init__(config)
        self._reverse_order = False

    def __repr__(self) -> str:
        return "number of reactions"

    def _score_node(self, node: MctsNode) -> float:
        reactions = node.actions_to()
        return len(reactions)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len(list(tree.reactions()))


class NumberOfPrecursorsScorer(Scorer):
    """Class for scoring nodes based on the number of pre-cursors in a node or route"""

    def __init__(self, config: Configuration = None) -> None:
        super().__init__(config)
        self._reverse_order = False

    def __repr__(self) -> str:
        return "number of pre-cursors"

    def _score_node(self, node: MctsNode) -> float:
        return len(node.state.mols)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len(list(tree.leafs()))


class NumberOfPrecursorsInStockScorer(Scorer):
    """Class for scoring nodes based on the number of pre-cursors in stock a node or route"""

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._stock = config.stock

    def __repr__(self) -> str:
        return "number of pre-cursors in stock"

    def _score_node(self, node: MctsNode) -> float:
        return len([mol for mol in node.state.mols if mol in self._stock])

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return len([mol for mol in tree.leafs() if mol in self._stock])


class AverageTemplateOccurrenceScorer(Scorer):
    """Class for scoring the nodes based on the average occurrence of the templates used to get to a node"""

    def __repr__(self) -> str:
        return "average template occurrence"

    def _calc_average(
        self, reactions: Sequence[Union[FixedRetroReaction, RetroReaction]]
    ) -> float:
        if not reactions:
            return 0.0
        occurrences = [self._get_occurrence(reaction) for reaction in reactions]
        return sum(occurrences) / len(reactions)

    def _score_node(self, node: MctsNode) -> float:
        return self._calc_average(node.actions_to())

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return self._calc_average(list(tree.reactions()))

    @staticmethod
    def _get_occurrence(reaction: Union[FixedRetroReaction, RetroReaction]) -> int:
        return reaction.metadata.get(
            "library_occurrence", reaction.metadata.get("library_occurence", 0)
        )


class PriceSumScorer(Scorer):
    """Scorer that sums the prices of all pre-cursors"""

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

    def __repr__(self) -> str:
        return "sum of prices"

    def _calculate_leaf_costs(
        self, leafs: Union[Sequence[Molecule], Iterable[Molecule]]
    ) -> dict:
        costs = {}
        for mol in leafs:
            if mol not in self._config.stock:
                continue
            try:
                cost = self._config.stock.price(mol)
            except StockException:
                costs[mol] = self.default_cost
            else:
                costs[mol] = cost

        max_cost = max(costs.values()) if costs else self.default_cost
        return defaultdict(lambda: max_cost * self.not_in_stock_multiplier, costs)

    def _score_node(self, node: MctsNode) -> float:
        leaf_costs = self._calculate_leaf_costs(node.state.mols)
        return sum(leaf_costs[mol] for mol in node.state.mols)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        leaf_costs = self._calculate_leaf_costs(tree.leafs())
        return sum(leaf_costs[leaf] for leaf in tree.leafs())


class RouteCostScorer(PriceSumScorer):
    """
    Score based on the cost of molecules and reactions.
    From Badowski et al. Chem Sci. 2019, 10, 4640
    """

    def __init__(
        self,
        config: Configuration,
        reaction_cost: int = 1,
        average_yield: float = 0.8,
        default_cost: int = 1,
        not_in_stock_multiplier: int = 10,
    ) -> None:
        super().__init__(
            config,
            default_cost=default_cost,
            not_in_stock_multiplier=not_in_stock_multiplier,
        )
        self.reaction_cost = reaction_cost
        self.average_yield = average_yield
        self._reverse_order = False

    def __repr__(self) -> str:
        return "route cost"

    def _score_node(self, node: MctsNode) -> float:
        leaf_costs = self._calculate_leaf_costs(node.state.mols)

        reactions, nodes = node.path_to()
        if not reactions:
            return leaf_costs[node.state.mols[0]]

        scores = {id(mol): leaf_costs[mol] for mol in nodes[-1].state.mols}
        for pnode, reaction in zip(nodes[::-1][1:], reactions[::-1]):
            updated_scores = {
                id(mol): scores[id(mol)]
                for mol in pnode.state.mols
                if mol is not reaction.mol
            }
            child_sum = sum(
                1 / self.average_yield * score
                for id_, score in scores.items()
                if id_ not in updated_scores
            )
            updated_scores[id(reaction.mol)] = self.reaction_cost + child_sum
            scores = updated_scores

        return list(scores.values())[0]

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        def _recursive_score(node):
            # This list should contains 0 or 1 elements
            reaction_nodes = list(tree.graph[node])
            if not reaction_nodes:
                return leaf_costs[node]

            child_sum = sum(
                1 / self.average_yield * _recursive_score(child)
                for child in tree.graph[reaction_nodes[0]]
            )
            return self.reaction_cost + child_sum

        leaf_costs = self._calculate_leaf_costs(tree.leafs())
        return _recursive_score(tree.root)
