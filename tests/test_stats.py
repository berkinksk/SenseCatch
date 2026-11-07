import math

from sklearn.feature_extraction.text import CountVectorizer

from diagnose_rules import mcnemar
from src.sensecatch.nbsvm_transformer import NBLogCountRatio
from zero_shot_baseline import wilson_ci


def test_wilson_interval_matches_a_hand_computed_value():
    assert wilson_ci(0.5, 100) == [0.4038, 0.5962]


def test_wilson_interval_matches_the_reported_stack_interval():
    assert wilson_ci(0.9298, 25000) == [0.9266, 0.9329]


def test_wilson_interval_handles_an_empty_sample():
    assert wilson_ci(0.0, 0) == [0.0, 0.0]


def test_mcnemar_matches_the_continuity_corrected_formula():
    stat, p = mcnemar(10, 2)
    assert stat == 49 / 12
    assert math.isclose(p, math.erfc(math.sqrt(stat / 2)), rel_tol=1e-9)


def test_mcnemar_with_no_disagreements():
    assert mcnemar(0, 0) == (0.0, 1.0)


def test_nbsvm_ratio_follows_the_class_of_each_word():
    docs = ["great fun", "great acting", "awful plot", "awful acting"]
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(docs)
    ratio = NBLogCountRatio().fit(X, [1, 1, 0, 0]).r_.ravel()
    weight = dict(zip(vectorizer.get_feature_names_out(), ratio))
    assert weight["great"] > 0
    assert weight["awful"] < 0
    assert math.isclose(weight["great"], math.log(3))
    assert math.isclose(weight["acting"], 0.0, abs_tol=1e-12)
