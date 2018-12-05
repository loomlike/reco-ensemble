"""
Recommendation Bagging Ensemble
"""

DEBUG = True

from random import randrange, uniform
from pyspark.sql import functions as F
from pyspark.sql import Window


class RecoBagging:
    def __init__(
        self,
        model_cls, num_models=3,
        user_col="user", item_col="item", rating_col="rating",
        **model_params
    ):
        """ Bagging ensemble.

        Args:
            model_cls (cls): Recommendation model class.
                The model interface should be the same as pySpark ALS DataFrame implementation,
                including the output format
            num_models (int): Number of models to combine
            user_col (str): User column
            item_col (str): Item column
            rating_col (str): Rating column
            **model_params: Model parameters
        """
        if num_models < 1:
            raise ValueError("At least one model is required to form an ensemble")

        self.model_cls = model_cls
        self.num_models = num_models
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.model_params = model_params

        # List of trained models
        self.models = []

        # Cache the models' recommendation results
        self.reco_output_df = None

    def fit(self, train_df):
        """ Fit models.

        Args:
            train_df (pySpark.DataFrame): Training set
        """

        # Reset models and outputs
        self.models.clear()
        if self.reco_output_df is not None:
            self.reco_output_df.unpersist()
            self.reco_output_df = None

        for i in range(self.num_models):
            # Randomize model hyper-parameter if the value is given as a range (tuple or list)
            params = {}
            for key, val in self.model_params.items():
                if isinstance(val, (list, tuple)):
                    if isinstance(val[0], int):
                        val = randrange(val[0], val[1])
                    else:
                        val = uniform(val[0], val[1])
                params[key] = val

            if DEBUG:
                print("Training model", i, params)

            model_cls = self.model_cls(**params)
            # Train with Bootstrap sampling
            model = model_cls.fit(train_df.sample(True, 1.0))
            self.models.append(model)

    def recommend_k_items(self, test_df, top_k=10, merge_by="sum", scale=True):
        """ Recommend k items

        Args:
            test_df (pySpark.DataFrame): Test set
            top_k (int): k items to recommend
            merge_by (str): Merging method in the ensemble. One of {"average", "sum", "count"}.
                If None, raw combined results (not aggregated) will be returned.
            scale (bool): Scale the preference prediction value or not

        Returns:
            item_recommendations (pySpark.DataFrame)
        """
        if len(self.models) == 0:
            raise SyntaxError("Train the model before use it")
        if merge_by not in {"average", "avg", "sum", "count", "cnt"}:
            raise ValueError(
                "Cannot handle {}. Use one of 'average', 'sum', or 'count'".format(
                    merge_by
                )
            )

        for i in range(self.num_models):
            if DEBUG:
                print("Recommending by", i)

            recommendations = (
                self.models[i]
                .recommendForUserSubset(test_df, top_k)
                .withColumn("recommendations", F.explode("recommendations"))
                .select(
                    self.user_col,
                    F.col("recommendations." + self.item_col),
                    F.col("recommendations.rating").alias(self.rating_col),
                    F.lit(i).alias("model"),
                )
            )

            # min-max scaling for each recommender output
            if scale:
                min_max = recommendations.agg(
                    F.min(self.rating_col), F.max(self.rating_col)
                ).collect()[0]
                scale = min_max[1] - min_max[0]
                recommendations = recommendations.withColumn(
                    self.rating_col,
                    (F.col(self.rating_col) - F.lit(min_max[0])) / F.lit(scale),
                )

            if i == 0:
                self.reco_output_df = recommendations
            else:
                self.reco_output_df = self.reco_output_df.union(recommendations)

        self.reco_output_df.cache()

        if merge_by == "average" or merge_by == "avg":
            merged = self.reco_output_df.groupBy(self.user_col, self.item_col).agg(
                F.avg(self.rating_col).alias(self.rating_col)
            )
        elif merge_by == "sum":
            merged = self.reco_output_df.groupBy(self.user_col, self.item_col).agg(
                F.sum(self.rating_col).alias(self.rating_col)
            )
        elif merge_by == "count" or merge_by == "cnt":
            merged = self.reco_output_df.groupBy(self.user_col, self.item_col).agg(
                F.count(self.rating_col).alias(self.rating_col)
            )

        # sort by rating for each user, select top k and collect the items as a list
        merged = (
            merged.withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy(self.user_col).orderBy(
                        F.col(self.rating_col).desc()
                    )
                ),
            )
            .where(F.col("rank") <= top_k)
            .withColumn(
                "recommendations",
                F.collect_list(
                    F.struct(self.item_col, F.col(self.rating_col).alias("rating"))
                ).over(Window.partitionBy(self.user_col)),
            )
            .groupBy(self.user_col)
            .agg(F.max("recommendations").alias("recommendations"))
        )

        return merged

    def prediction(self, test_df, merge_by="sum", scale=True):
        """ Predict ratings by averaging M models' predictions

        Args:
            test_df (pySpark.DataFrame): Test set
            merge_by (str): Merging method in the ensemble. One of {"average", "sum", "count"}.
                If None, raw combined results (not aggregated) will be returned.
            scale (bool): Scale the preference prediction value or not

        Returns:
            rating_predictions (pySpark.DataFrame)
        """
        raise NotImplementedError
