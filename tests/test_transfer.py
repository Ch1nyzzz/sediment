from sediment.transfer import canonical_transfer_reflection, transfer_features


def test_transfer_features_collapse_domain_words_to_workflow_tags():
    text = "Disambiguate duplicate patients, preserve fields, then assign an available bed."
    assert {"disambiguate", "preserve", "order", "assign", "availability", "capacity"} \
        <= transfer_features(text)


def test_canonical_transfer_reflection_contains_no_source_domain_terms():
    task = "Delete duplicate patient Alice after removing her active bed reservation."
    evidence = "get_patient -> ok; cancel_reservation -> ok; delete_patient -> ok"
    out = canonical_transfer_reflection(task, evidence)
    assert "patient" not in out and "Alice" not in out and "bed" not in out
    assert "[delete]" in out
