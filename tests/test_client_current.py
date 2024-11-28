import os
import pytest
import statistics
from src.client import FrenchSecondHandCarsClient
from tests.data.cars import cars
from tests.constants.common import THRESHOLD, THRESHOLD_INDIVIDUAL

@pytest.fixture
def predictor():
    return FrenchSecondHandCarsClient()

@pytest.fixture
def test_cases():
    return cars

def calculate_precision(predicted: float, actual: float) -> float:
    return 1 - abs(predicted - actual) / actual


@pytest.mark.parametrize("case_index", range(len(cars)))  # Adjust range based on test cases count
def test_individual_predictions(predictor, test_cases, case_index):
    case = test_cases[case_index]
    predicted_price = predictor.predict(case['data'])
    precision = calculate_precision(predicted_price, case['real_price'])
    print(f"{case['data']['carmodel'].upper()}")
    print(f"Predicted price: {predicted_price}")
    print(f"Real price: {case['real_price']}")
    print(f"Precision: {precision:.2%}")
    
    assert precision >= THRESHOLD_INDIVIDUAL, f"Precision below {THRESHOLD_INDIVIDUAL*100}% threshold."

def test_average_precision(predictor, test_cases):
    precisions = []
    for case in test_cases:
        predicted_price = predictor.predict(case['data'])
        precision = calculate_precision(predicted_price, case['real_price'])
        precisions.append(precision)

    avg_precision = statistics.mean(precisions)
    print(f"Average precision: {avg_precision:.2%}")
    assert avg_precision >= THRESHOLD, f"Average precision below {THRESHOLD*100}% threshold."