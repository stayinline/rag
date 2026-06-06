from app.services.planner import build_query_plan


def test_build_query_plan_decomposes_compound_query():
    plan = build_query_plan(
        "Compare EGFR resistance and PD-1 adverse events; then summarize monitoring",
        mode="langgraph",
        max_queries=5,
    )

    assert plan.enabled is True
    assert plan.queries[0].startswith("Compare EGFR")
    assert any("PD-1 adverse events" in item for item in plan.queries)
    assert len(plan.queries) >= 3


def test_build_query_plan_disabled_returns_single_query():
    plan = build_query_plan("simple question", mode="none")

    assert plan.enabled is False
    assert plan.queries == ["simple question"]
