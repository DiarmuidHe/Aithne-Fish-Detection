from scripts.live_recall_benchmark import summarize


def test_recall_summary_requires_actual_sixty_percent_observations():
    rows = [{"id": key, "confidence": score} for key, score in
            [("a", .60), ("a", .65), ("a", .61),
             ("b", .599), ("b", .59), ("b", .95), ("c", .50)]]
    original = [dict(row) for row in rows]
    assert summarize([], rows, 4) == {
        "observations_at_60": 4, "tracks_with_3_observations_at_60": 1,
        "accepted_singletons": 1}
    assert rows == original
