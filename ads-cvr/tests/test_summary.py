from main import _summary_fields


def test_summary_fields_include_metrics_added_by_persistence():
    fields = _summary_fields(
        [
            {"trial": 1, "status": "ok", "val_logloss": 0.3},
            {"trial": 2, "status": "ok", "best_val_logloss": 0.2},
        ]
    )

    assert "best_val_logloss" in fields
    assert len(fields) == len(set(fields))
