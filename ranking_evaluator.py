"""
Ranking evaluator for pySpark DataFrame

Wrapper of pySpark MLLib (RDD api) RankingMetrics
"""

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics

import warnings
import numpy as np

# Default col names
ITEM_BY_RATING = 'item_by_rating'
RATING = 'rating'
ITEM_BY_PREDICTION = 'item_by_prediction'
PREDICTION = 'prediction'


class RankingEvaluator:
    def __init__(self, true_df, pred_df=None, reco_df=None,
                 user_col='userId', item_col='item', rating_col='rating', pred_col='rating',
                 reco_col='recommendations'):
        """Evaluate by using ranking metrics (use PySpark RDD implementation).

        Args:
            true_df (pySpark DataFrame): Groundtruth dataframe containing user_col, item_col, rating_col.
                Should be aggregated by (user_col, item_col) if there exist multiple interactions per user per item.
            pred_df (pySpark DataFrame): Prediction dataframe containing user_col, item_col, pref_col.
            reco_df (pySpark DataFrame): Recommendation list dataframe containing user_col and reco_col.
                reco_col should be a list of (item_col, pred_col) tuples sorted by the predicted ratings.
                (AKA, pySpark ALS recommendation output format).
            user_col (str): User id column name
            item_col (str): Item id column name
            rating_col (str): Rating column name
            pred_col (str): Predicted rating column name
            reco_col (str): Recommendation list column name
        """

        if pred_df is None and reco_df is None:
            raise ValueError("Either prediction dataframe or recommendation dataframe should be provided.")
        elif pred_df is not None and reco_df is not None:
            warnings.warn("Both prediction and recommendation dataframes are provided. \
                Prediction dataframe will be ignored.", Warning)

        try:
            # Data preparation - Sort items by ratings for each user
            true_list_df = true_df.withColumn('groundtruths',
                                              F.collect_list(F.struct(item_col, rating_col)).over(
                                                  Window.partitionBy(user_col).orderBy(F.desc(rating_col))
                                              )) \
                .groupBy(user_col) \
                .agg(F.max('groundtruths').alias('groundtruths'))

            if reco_df is not None:
                pred_list_df = reco_df
            else:
                pred_list_df = pred_df.withColumn(reco_col,
                                                  F.collect_list(F.struct(item_col, pred_col)).over(
                                                      Window.partitionBy(user_col).orderBy(F.desc(pred_col))
                                                  )) \
                    .groupBy(user_col) \
                    .agg(F.max(reco_col).alias(reco_col))

            # Following columns will be cached for the ndcg calculation
            # [list of items sorted by rating], [list of corresponding rating],
            # [list of items sorted by predicted-rating], [list of corresponding predicted-rating]
            self.ratings_predictions = true_list_df.join(pred_list_df, user_col) \
                .select(
                user_col,
                F.col('groundtruths.' + item_col).alias(ITEM_BY_RATING),
                F.col('groundtruths.' + rating_col).alias(RATING),
                F.col(reco_col + '.' + item_col).alias(ITEM_BY_PREDICTION),
                F.col(reco_col + '.' + pred_col).alias(PREDICTION)
            ).cache()

            # convert to RDD as Spark's RankingMetrics is RDD-based implementation
            self.predictions_labels_rdd = self.ratings_predictions.rdd.map(lambda row: (row[3], row[1]))
            self.ranker = RankingMetrics(self.predictions_labels_rdd)

        except Exception as e:
            print(str(e))
            raise

    def ndcgAt(self, k, binary_relevance=True):
        """Normalized Discounted Cumulative Gain (nDCG).

        Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

        Args:
            k (int): top k
            binary_relevance (bool): if False, use ratings as relevance values. Otherwise, will use Spark mllib's ndcg

        Returns:
            nDCG (float): Averaged nDCG over users (max=1, min=0)
        """
        if k == 0:
            return .0

        if binary_relevance:
            # pySpark implementation use indicator-relevance (1 or 0)
            return self.ranker.ndcgAt(k)

        def ndcg(recommend_list, item_list, rating_list, k, log_e):
            """Calculate ndcg for a row (a user)

            Args:
                recommend_list (List): Sorted list of the recommended items by the predictions
                item_list (List): Sorted list of the groundtruth items by the ratings
                rating_list (List): Sorted list of the ratings
                k (int): Given parameter K
                log_e (List): Pre-calculated log values from log(1) to log(k)

            Returns:
                nDCG (float): max=1, min=0
            """
            # make item-rating map for look up
            item_rating = dict(zip(item_list, rating_list))

            # calculate dcg
            dcg = .0
            for i in range(min(k, len(recommend_list))):
                r = item_rating.get(recommend_list[i])
                if r is not None:
                    dcg = dcg + r / log_e[i + 1]

            # calculate idcg (ideal-dcg)
            idcg = .0
            for i in range(min(k, len(rating_list))):
                idcg = idcg + rating_list[i] / log_e[i + 1]

            if idcg == .0:
                # should filter out this case
                return -1.0
            else:
                return dcg / idcg

        ndcg_udf = F.udf(ndcg, DoubleType())

        # Cache logs
        log_e = F.array([F.lit(np.log2(x)) for x in range(1, k + 2)])

        ndcg_df = self.ratings_predictions \
            .select(ndcg_udf(ITEM_BY_PREDICTION, ITEM_BY_RATING, RATING, F.lit(k), log_e).alias('ndcg')) \
            .filter(F.col('ndcg') >= 0.0)

        if ndcg_df.count() == 0:
            raise ValueError("All sample's relevance are zeros")

        # Average across users
        return ndcg_df.agg(F.avg('ndcg').alias('avg_ndcg')).collect()[0].avg_ndcg

    def precisionAt(self, k):
        return self.ranker.precisionAt(k)

    def recallAt(self, k):
        return self.predictions_labels_rdd.map(
            lambda x: float(len(set(x[0][:k]).intersection(set(x[1])))) / float(len(x[1]))
        ).mean()

    def meanAveragePrecision(self):
        return self.ranker.meanAveragePrecision
