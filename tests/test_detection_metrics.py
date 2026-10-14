from app.services.detection_metrics import count_matches


def test_one_to_one_class_aware_matching():
    truth = [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))]
    predicted = [(0, (0, 0, 10, 10), .9), (0, (0, 0, 10, 10), .8),
                 (0, (20, 20, 30, 30), .7)]
    assert count_matches(truth, predicted) == {"tp": 1, "fp": 2, "fn": 1}
