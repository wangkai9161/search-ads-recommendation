import pandas as pd

from prepare.reader import _hash_column, feature_columns


def test_hashing_uses_field_namespace_and_preserves_missing_bucket():
    values = pd.Series(["123", "same", "", "-1"])

    user_hash = _hash_column(values, 100_000, "user_id")
    product_hash = _hash_column(values, 100_000, "product_id")

    assert user_hash[0] != product_hash[0]
    assert user_hash[1] != product_hash[1]
    assert user_hash[2:].tolist() == [0, 0]
    assert product_hash[2:].tolist() == [0, 0]


def test_no_entity_ids_feature_set_removes_only_user_and_product_ids():
    sparse, dense = feature_columns("no_entity_ids")

    assert "user_id" not in sparse
    assert "product_id" not in sparse
    assert "partner_id" in sparse
    assert dense == ("click_timestamp", "nb_clicks_1week", "product_price")


def test_coarse_context_removes_identity_proxies():
    sparse, _ = feature_columns("coarse_context")

    assert {"device_type", "product_category_1", "product_country"}.issubset(sparse)
    assert {
        "audience_id",
        "product_brand",
        "product_id",
        "product_title",
        "partner_id",
        "user_id",
    }.isdisjoint(sparse)
