import os 
import sys
import os.path

if __name__ == "__main__":

    sys.path.insert(1, 
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from pyrankagg.rankagg import FullListRankAggregator

def perform_rank_aggregation(
    rank_aggregation_input: list,
    are_scores: bool = True,
    ):
    """Use pyrankagg to aggregate the items in the given list of experts

    Parameters
    ----------
    rank_aggregation_input : list
        List of expert scores. Each element corresponds to one expert and is a 
        dictionary containing scores for each item from that expert. 

    Returns
    -------
    dict
        The aggregated ranks of the items
    """
    aggregator = FullListRankAggregator() #the package is for rank aggregation
    rank_aggregation_output = aggregator.aggregate_ranks(rank_aggregation_input, areScores=are_scores)
    if isinstance(rank_aggregation_output, tuple):
        rank_aggregation_output = rank_aggregation_output[-1] # TODO:?
    assert isinstance(rank_aggregation_output, dict), f"rank_aggregation_output is NOT A DICT! {rank_aggregation_output}"
    return rank_aggregation_output