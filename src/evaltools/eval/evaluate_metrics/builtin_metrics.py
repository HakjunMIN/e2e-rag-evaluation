from azure.ai.evaluation import (
    CoherenceEvaluator,
    F1ScoreEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator
)

from .base_metric import BaseMetric

   
class BuiltinRatingMetric(BaseMetric):
    @classmethod
    def get_aggregate_stats(cls, df):
        return cls.get_aggregate_stats_for_numeric_rating(df, cls.METRIC_NAME)

class BuiltinViolenceMetric(BaseMetric):
    METRIC_NAME = "violence_score"

    @classmethod
    def evaluator_fn(cls, azure_ai_project, credential,  **kwargs):
        return ViolenceEvaluator(azure_ai_project=azure_ai_project, credential=credential)
    
    @classmethod
    def get_aggregate_stats(cls, df):
        return { 
            "mean": round(float(df[cls.METRIC_NAME].mean()), 2), 
            "max": round(float(df[cls.METRIC_NAME].max()), 2), 
            "min": round(float(df[cls.METRIC_NAME].min()), 2), 
        }
    
class BuiltinRelevanceMetric(BuiltinRatingMetric):
    METRIC_NAME = "gpt_relevance"

    @classmethod
    def evaluator_fn(cls, openai_config, **kwargs):
        return RelevanceEvaluator(openai_config)


class BuiltinCoherenceMetric(BuiltinRatingMetric):
    METRIC_NAME = "gpt_coherence"

    @classmethod
    def evaluator_fn(cls, openai_config, **kwargs):
        return CoherenceEvaluator(openai_config)


class BuiltinGroundednessMetric(BuiltinRatingMetric):
    METRIC_NAME = "gpt_groundedness"

    @classmethod
    def evaluator_fn(cls, openai_config, **kwargs):
        return GroundednessEvaluator(openai_config)


class BuiltinSimilarityMetric(BuiltinRatingMetric):
    METRIC_NAME = "gpt_similarity"

    @classmethod
    def evaluator_fn(cls, openai_config, **kwargs):
        return SimilarityEvaluator(openai_config)


class BuiltinFluencyMetric(BuiltinRatingMetric):
    METRIC_NAME = "gpt_fluency"

    @classmethod
    def evaluator_fn(cls, openai_config, **kwargs):
        return FluencyEvaluator(openai_config)


class BuiltinF1ScoreMetric(BaseMetric):
    METRIC_NAME = "f1_score"

    @classmethod
    def evaluator_fn(cls, **kwargs):
        return F1ScoreEvaluator()

    @classmethod
    def get_aggregate_stats(cls, df):
        return {
            "mean": round(df[cls.METRIC_NAME].mean(), 2),
            "max": round(df[cls.METRIC_NAME].max(), 2),
            "min": round(df[cls.METRIC_NAME].min(), 2),
        }
