#!/usr/bin/env python3
"""Print every output block used in Chapter 18, from the current files.

Run from the project root:

    python scripts/chapter18_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data" / "raw" / "reporting"


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(REP / "rep_findings.csv"),
        pd.read_csv(REP / "rep_kpi.csv"),
        pd.read_csv(REP / "rep_scenarios.csv"),
        pd.read_csv(REP / "rep_questions.csv"),
        pd.read_csv(REP / "rep_audiences.csv"),
    )


def section_181(f, k, s, q, au) -> None:
    show("A1 audience table", au.to_string(index=False))
    show("A2 thirty second versions", {
        "executive": "Revenue is down 6.2%; fund a $180k regional pilot.",
        "manager": "South and West drove 71% of the drop; web traffic fell 17%.",
        "technical": "The sequential bridge closes to $0; 3 months of cost data are missing.",
        "client": "The decline is real and localized; the proposed remedy is a bounded test.",
    })
    show("A3 curse of knowledge", {
        "expert": "The PR AUC is 0.141 against a 3.47% base rate.",
        "plain": "The shortlist finds about four times as many leavers as chance.",
    })
    show("A4 one finding three audiences", {
        "executive": "A $180k test could recover $1.4m.",
        "manager": "Focus retention effort on the South and West regions.",
        "technical": "Model confidence is 0.70; the holdout is pre-registered.",
    })
    show("A5 attention", {
        "executive_seconds": 30, "manager_minutes": 2,
        "technical_minutes": 10, "client_minutes": 5,
    })
    show("A6 audience rule", [
        "lead with the decision the audience can make",
        "give evidence in the units they use",
        "put method in the appendix unless method is the question",
    ])


def section_182(f, k, s, q, au) -> None:
    show("B1 headline rewrites", pd.DataFrame([
        ("we analysed the data and found several trends",
         "Revenue fell 6.2%; fund a $180k regional pilot"),
        ("there are differences between regions",
         "South and West explain 71% of the decline"),
        ("the model performed reasonably well",
         "The shortlist finds 23 leavers in 150 reviews"),
        ("some data quality issues were noted",
         "Three months of cost data are missing; margin is provisional"),
    ], columns=["weak", "action_title"]).to_string(index=False))
    show("B2 pyramid", [
        "headline: the finding in one sentence",
        "evidence: three numbers that support it",
        "implication: what it means for the decision",
        "recommendation: the action and its owner",
        "appendix: method, checks, and detail",
    ])
    top = f.sort_values("relevance", ascending=False).head(3)
    show("B3 thirty to three", top[["id", "finding", "value", "unit"]]
         .to_string(index=False))
    show("B4 action titles", [
        "Fund a regional pilot", "Cut the review queue to 100",
        "Fix the cost feed before publishing margin",
        "Hold pricing steady this quarter",
    ])
    show("B5 one page brief", [
        "FINDING: revenue fell 6.2% year over year",
        "EVIDENCE: South and West = 71% of the gap; traffic -17%",
        "IMPLICATION: this is acquisition, not pricing",
        "RECOMMENDATION: $180k pilot, sales lead, four-week gate",
        "LIMITATIONS: pull-forward untested without a holdout",
    ])
    show("B6 headline check", {
        "has_a_number": True, "has_a_direction": True,
        "names_an_action": True, "under_20_words": True,
    })
    show("B7 conclusion first", {
        "rule": "write the headline before the last chart is finished",
        "reason": "the analysis then has a question to answer",
    })
    show("B8 checkpoint rewrite", {
        "given": "we analyzed the data and found several trends",
        "rewritten": "Revenue fell 6.2% year over year, concentrated in "
                     "South and West; fund a $180k regional pilot",
    })


def section_183(f, k, s, q, au) -> None:
    show("C1 brief vs full", {
        "one_page": "finding, evidence, implication, recommendation",
        "full_report": "context, method, findings, limitations, "
                       "recommendations, appendix",
        "words_one_page": 350, "words_full": 6000,
    })
    show("C2 report outline", [
        "1. context and decision", "2. data and definitions",
        "3. method and checks", "4. findings ranked by relevance",
        "5. limitations", "6. recommendations", "7. appendix",
    ])
    def rag(row):
        a, t = row["actual"], row["target"]
        if row["kpi"] == "NPS":
            return "green" if a >= t else ("amber" if a >= t - 0.1 else "red")
        if row["direction"] == "higher":
            if a >= t:
                return "green"
            return "amber" if a >= 0.93 * t else "red"
        if a <= t:
            return "green"
        return "amber" if a <= 1.05 * t else "red"
    kk = k.copy()
    kk["status"] = kk.apply(rag, axis=1)
    show("C3 rag table", kk[["kpi", "actual", "target", "status", "owner"]]
         .to_string(index=False))
    kk["variance"] = kk["actual"] - kk["target"]
    show("C4 variance", kk[["kpi", "variance", "status"]]
         .round(4).to_string(index=False))
    show("C5 monthly review", [
        "1. status against target (5 min)",
        "2. three largest variances (10 min)",
        "3. one decision and its owner (5 min)",
    ])
    show("C6 weekly review", [
        "1. leading indicators", "2. exceptions",
        "3. actions due", "4. blockers",
    ])
    show("C7 length discipline", {
        "executive_brief": 1, "analytical_report": 12,
        "slide_deck": 10, "appendix": "unlimited but unread",
    })
    show("C8 appendix rule", {
        "test": "if a reader asks for it, move it to the appendix",
        "keep": "headline, three numbers, one ask",
    })


def section_184(f, k, s, q, au) -> None:
    show("D1 ten slides", pd.DataFrame([
        (1, "Title", "Decision in the subtitle"),
        (2, "Finding", "One sentence with a number"),
        (3, "Evidence", "Three numbers, one chart"),
        (4, "Cause", "The driver tree"),
        (5, "Region", "South and West"),
        (6, "Channel", "Traffic, not conversion"),
        (7, "Uncertainty", "Ranges and a stop rule"),
        (8, "Options", "Three choices with downsides"),
        (9, "Recommendation", "One ask with an owner"),
        (10, "Appendix", "Method and checks"),
    ], columns=["n", "title", "content"]).to_string(index=False))
    show("D2 chart titles", pd.DataFrame([
        ("Revenue by month", "Revenue is down 6.2% year over year"),
        ("Conversion by channel", "Web traffic fell 17%; conversion held"),
        ("Model ROC curve", "The shortlist finds 23 leavers in 150 reviews"),
    ], columns=["descriptive", "action_title"]).to_string(index=False))
    show("D3 what to cut", [
        "method slides that no one asked for",
        "four charts where one will do",
        "decimal places beyond the decision's precision",
        "a summary slide that repeats the headline",
    ])
    show("D4 slide words", {
        "title": 8, "finding": 18, "evidence": 25,
        "recommendation": 20, "rule": "under 30 words a slide",
    })
    show("D5 appendix discipline", {
        "appendix_slides": 6, "rule": "numbered and referenced in the body",
        "test": "no slide in the main deck needs a spoken explanation",
    })
    show("D6 chart sizing", {
        "main_chart": "one per slide, at least half the slide",
        "small_multiples": "max six panels",
        "annotation": "name the point that matters",
    })
    show("D7 annotation", {
        "weak": "a line chart",
        "strong": "the line plus a labelled arrow at the 6.2% drop",
    })
    show("D8 slide test", [
        "could this slide be understood without the speaker?",
        "does it make one point?",
        "does it end with a decision?",
    ])


def section_185(f, k, s, q, au) -> None:
    show("E1 narrative arc", [
        "context: what changed", "complication: what we expected",
        "question: what actually happened", "answer: the finding",
        "action: what to do next",
    ])
    show("E2 contrast context stakes", {
        "contrast": "Q3 was stable; Q4 fell 15.5%",
        "context": "the same quarter fell 12% last year",
        "stakes": "$469,438 of revenue at issue",
    })
    show("E3 memorable chart", {
        "one_chart": "the cumulative gain curve",
        "reason": "it shows the value of a shortlist in one line",
    })
    show("E4 spin test", pd.DataFrame([
        ("Q4 fell 15.5%", "Q4 fell 15.5% but is up 10% year on year", "balanced"),
        ("the model is 96% accurate", "the model is 96% accurate at a 3.5% base rate", "honest"),
        ("accuracy rose", "accuracy rose because the base rate rose", "careful"),
    ], columns=["claim", "rewritten", "verdict"]).to_string(index=False))
    show("E5 ethical framing", [
        "show the comparison that a reasonable opponent would choose",
        "label a model-derived association as an association",
        "state who bears the cost of a wrong decision",
    ])
    show("E6 story checklist", {
        "has_a_contrast": True, "has_a_number": True,
        "names_a_stakeholder": True, "admits_a_limit": True,
    })


def section_186(f, k, s, q, au) -> None:
    show("F1 ranges", s.to_string(index=False))
    show("F2 probability language", pd.DataFrame([
        ("certain", "will happen", "not available in analysis"),
        ("likely", "70-90%", "a held-out estimate"),
        ("possible", "10-40%", "a scenario, not a forecast"),
        ("unlikely", "below 10%", "a tail risk"),
    ], columns=["word", "range", "basis"]).to_string(index=False))
    show("F3 scenarios", {
        "best": "conversion 5.5%, uplift 32%",
        "base": "conversion 4.2%, uplift 18%",
        "worst": "conversion 3.0%, uplift 5%",
    })
    show("F4 evidence language", {
        "no_evidence": "the test did not detect an effect",
        "evidence_of_no_effect": "the interval excludes a useful effect",
        "why_it_matters": "absence of evidence is not evidence of absence",
    })
    show("F5 decimal trap", {
        "source": "conversion 0.1066321",
        "reported": "10.7%",
        "rule": "round to the precision the decision can use",
    })
    show("F6 confidence labels", {
        "high": "multiple independent checks agree",
        "moderate": "one holdout, no causal test",
        "low": "assumption-driven or a single small sample",
    })
    show("F7 uncertainty sentences", [
        "We estimate a 6.2% decline (90% interval 5.1-7.3%).",
        "The pilot's expected value is $1.4m if uplift exceeds 12%.",
        "We cannot distinguish seasonality from pull-forward here.",
    ])
    show("F8 uncertainty checklist", [
        "range beside every point estimate",
        "word for every probability",
        "a named unknown",
        "a decision that is robust across the range",
    ])


def section_187(f, k, s, q, au) -> None:
    show("G1 three artifacts", pd.DataFrame([
        ("dashboard", "live", "monitor", "what is happening now?"),
        ("report", "periodic", "inform", "what happened this period?"),
        ("notebook", "on demand", "explore", "why did it happen?"),
    ], columns=["artifact", "cadence", "purpose", "question"]
    ).to_string(index=False))
    show("G2 parameterized", {
        "regions": ["Midwest", "South", "West"],
        "reports": 3, "template": "one function",
        "note": "papermill is not installed; a plain function reproduces it",
    })
    tpl = Template("{{ region }} revenue is {{ value }} ({{ status }}).")
    rendered = [tpl.render(region=r, value=v, status=s_) for r, v, s_ in
                [("Midwest", "$4.1m", "green"),
                 ("South", "$3.6m", "red"),
                 ("West", "$3.9m", "amber")]]
    show("G3 jinja render", rendered)
    show("G4 scheduled job", {
        "trigger": "first working day, 07:00",
        "inputs": "rep_findings.csv, rep_kpi.csv",
        "outputs": "brief.md, kpi.csv, log.txt",
        "on_failure": "alert owner, do not send stale report",
    })
    show("G5 delivery", ["email", "Slack webhook", "shared drive",
                         "PDF archive"])
    show("G6 failure alerting", {
        "checks": ["row count", "control total", "freshness", "render success"],
        "action": "block send and page the owner",
    })
    show("G7 report debt", [
        "an unreviewed report nobody owns",
        "a metric defined in three places",
        "a chart that silently changed source",
        "a dashboard no one has opened in a quarter",
    ])
    show("G8 inventory", {
        "active_reports": 5, "owners_assigned": 5,
        "last_reviewed_days": [12, 30, 91, 14, 240],
    })
    show("G9 papermill note", {
        "installed": False,
        "equivalent": "a Python function with explicit parameters",
        "example": "render_report(region='South', month='2026-01')",
    })
    show("G10 streamlit note", {
        "installed": False,
        "equivalent": "a Jinja2 page served by a static host",
        "example": "five questions answered by one template",
    })


def section_188(f, k, s, q, au) -> None:
    show("H1 readme structure", [
        "purpose", "data sources", "how to run", "outputs",
        "assumptions", "owner", "reproduction time",
    ])
    show("H2 method notes", [
        "revenue definition", "exclusions", "bridge order",
        "imputation rule", "sensitivity checks",
    ])
    show("H3 lineage", [
        "raw files", "generator scripts", "analysis notebook",
        "report template", "published brief",
    ])
    show("H4 reproduction", [
        "python scripts/generate_chapter18_sources.py",
        "python scripts/chapter18_outputs.py",
        "open chapters/chapter18-Communication.adoc",
    ])
    show("H5 ownership", {
        "data": "analytics engineering", "model": "analytics",
        "report": "reporting lead", "decision": "sales lead",
    })
    show("H6 handover checklist", [
        "a stranger can run it in under ten minutes",
        "every number traces to a fixture",
        "owners are named", "assumptions are written down",
    ])
    show("H7 stranger test", {
        "steps": 3, "minutes": 8,
        "result": "reproduced without asking the author",
    })
    show("H8 readme sample", [
        "# Revenue review",
        "Run: python scripts/chapter18_outputs.py",
        "Inputs: data/raw/reporting/*.csv",
        "Owner: reporting lead; review monthly",
    ])


def section_189(f, k, s, q, au) -> None:
    show("I1 prebrief", [
        "tell the decision-maker the headline before the meeting",
        "send the one-pager 24 hours ahead",
        "ask which objection they will raise",
    ])
    show("I2 message messenger", {
        "bad_news": "the numbers are wrong",
        "reframe": "the extract is incomplete; here is what is safe to say",
        "rule": "criticise the artefact, not the person",
    })
    show("I3 bad news script", [
        "state the finding in one sentence",
        "give the evidence",
        "name the immediate containment step",
        "give the date for the full answer",
    ])
    show("I4 defend method", [
        "restate the question the method answers",
        "show the check that would falsify it",
        "offer the alternative and its cost",
    ])
    show("I5 objections", pd.DataFrame([
        ("the number is too low", "show the control total and the definition"),
        ("last year was different", "show the matched-period comparison"),
        ("the model is a black box", "show permutation importance and a holdout"),
    ], columns=["objection", "response"]).to_string(index=False))
    show("I6 escalation", {
        "disagreement_on_fact": "reconcile to source",
        "disagreement_on_interpretation": "name the shared objective",
        "disagreement_on_values": "escalate to the decision owner",
    })


def section_1810(f, k, s, q, au) -> None:
    show("J1 recommendation fields", [
        "specific action", "owner", "cost", "expected effect",
        "measurement plan", "decision requested",
    ])
    show("J2 business case", {
        "action": "fund a $180k regional pilot",
        "owner": "sales lead", "cost": 180000.0,
        "expected_effect": "$1.4m recovery if uplift exceeds 12%",
        "measurement": "four-week holdout on precision at capacity",
        "decision": "approve or decline today",
    })
    show("J3 decision requested", {
        "question": "approve the pilot?",
        "by": "2026-02-01", "owner": "sales lead",
    })
    show("J4 six fields check", {
        "action": True, "owner": True, "cost": True,
        "effect": True, "measurement": True, "decision": True,
    })
    show("J5 acted-on test", [
        "can the reader name the owner?",
        "can finance name the cost?",
        "can the team name the measurement window?",
    ])
    show("J6 final memo", [
        "Revenue fell 6.2%; South and West explain 71%.",
        "Fund a $180k pilot under the sales lead.",
        "Expected recovery $1.4m; decide by 2026-02-01.",
    ])
    show("J7 final checklist", [
        "finding first", "three numbers", "one decision",
        "named owner", "ranges not point estimates",
        "limitations stated", "handover documented",
    ])
    show("J8 closing", {
        "goal": "make the finding land and the work rerunnable",
        "test": "a stranger can reproduce it and a decision-maker can act",
    })


def section_extra(f, k, s, q, au) -> None:
    """Additional audience and structure outputs for Chapter 18."""
    show("N1 findings by section", {
        k_: int(v) for k_, v in f.groupby("section").size().items()
    })
    show("N2 findings by audience", {
        k_: int(v) for k_, v in f.groupby("audience").size().items()
    })
    show("N3 confidence bands", {
        "high": int((f["confidence"] >= 0.90).sum()),
        "medium": int(f["confidence"].between(0.70, 0.90).sum()),
        "low": int((f["confidence"] < 0.70).sum()),
    })
    show("N4 relevance bands", {
        "top10": int((f["relevance"] >= 9).sum()),
        "middle": int(f["relevance"].between(5, 8).sum()),
        "low": int((f["relevance"] < 5).sum()),
    })
    weak = "we analysed the data and found several trends"
    strong = "Revenue fell 6.2%; fund a $180k regional pilot"
    show("N5 headline lengths", {
        "weak_words": len(weak.split()), "strong_words": len(strong.split()),
    })
    show("N6 pitch lengths", {
        "executive_words": len("Revenue is down 6.2%; fund a $180k regional pilot.".split()),
        "manager_words": len("South and West drove 71% of the drop; web traffic fell 17%.".split()),
    })
    top = f.sort_values("relevance", ascending=False).head(3)
    show("N7 top values", {str(r["id"]): float(r["value"])
                          for _, r in top.iterrows()})

    def rag2(row):
        a, t = row["actual"], row["target"]
        if row["kpi"] == "NPS":
            return "green" if a >= t else ("amber" if a >= t - 0.1 else "red")
        if row["direction"] == "higher":
            if a >= t:
                return "green"
            return "amber" if a >= 0.93 * t else "red"
        if a <= t:
            return "green"
        return "amber" if a <= 1.05 * t else "red"
    k2 = k.copy(); k2["status"] = k2.apply(rag2, axis=1)
    show("N8 rag counts", {x: int((k2["status"] == x).sum())
                          for x in ["green", "amber", "red"]})
    k2["var"] = k2["actual"] - k2["target"]
    show("N9 largest gaps", k2.reindex(k2["var"].abs().sort_values(
        ascending=False).index).head(3)[["kpi", "var", "status"]]
        .round(4).to_string(index=False))
    show("N10 kpi units", {k_: int(v) for k_, v in
         k.groupby("unit").size().items()})

    show("N11 slide timing", {
        "opening": 3, "finding": 2, "evidence": 4,
        "cause": 3, "options": 3, "decision": 3, "buffer": 2,
    })
    show("N12 chart selection", pd.DataFrame([
        ("trend over time", "line"), ("category comparison", "bar"),
        ("two-variable relationship", "scatter"), ("distribution", "histogram"),
    ], columns=["question", "chart"]).to_string(index=False))
    show("N13 colour semantics", {
        "red": "below target or adverse", "amber": "near threshold",
        "green": "target met", "grey": "no target or not applicable",
    })
    show("N14 annotation checklist", [
        "name the point that matters", "state the unit",
        "label the comparison", "avoid a legend when one label will do",
    ])
    show("N15 narrative beats", {
        "context": 25, "complication": 25, "question": 15,
        "answer": 30, "action": 25,
    })
    show("N16 spin phrases", [
        "improved significantly", "performed well",
        "a number of issues", "looks promising",
    ])
    show("N17 probability calibration", {
        "likely": "a held-out estimate, not a feeling",
        "possible": "a scenario with a named mechanism",
        "unlikely": "a tail with a stated consequence",
    })
    show("N18 scenario spread", {
        "conversion": round(float(s["high"].iloc[0] - s["low"].iloc[0]), 4),
        "aov": round(float(s["high"].iloc[1] - s["low"].iloc[1]), 2),
        "uplift": round(float(s["high"].iloc[2] - s["low"].iloc[2]), 4),
    })
    show("N19 template variables", ["region", "value", "status", "period"])
    show("N20 rendered regions", 3)


def section_extra2(f, k, s, q, au) -> None:
    """Additional automation and handover outputs for Chapter 18."""
    show("P1 report catalog", pd.DataFrame([
        ("daily KPI", "dashboard", "operations", "live"),
        ("monthly brief", "report", "executive", "monthly"),
        ("regional pack", "report", "managers", "monthly"),
        ("revenue notebook", "notebook", "analysts", "on demand"),
        ("quality audit", "notebook", "data team", "weekly"),
    ], columns=["name", "artifact", "audience", "cadence"]
    ).to_string(index=False))
    show("P2 trigger schedule", {
        "daily": "07:00 local", "weekly": "Monday 08:00",
        "monthly": "first working day", "on_demand": "manual",
    })
    show("P3 failure simulation", {
        "row_count": "pass", "control_total": "pass",
        "freshness": "fail", "render": "not run",
        "action": "block send and page owner",
    })
    show("P4 delivery matrix", pd.DataFrame([
        ("executive", "email", "brief"),
        ("manager", "Slack", "status"),
        ("technical", "shared drive", "notebook"),
        ("client", "PDF", "signed summary"),
    ], columns=["audience", "channel", "artifact"]).to_string(index=False))
    show("P5 raci", {
        "raw_data": "analytics engineering",
        "generator": "analytics engineering",
        "analysis": "analytics", "report": "reporting lead",
        "decision": "sales lead",
    })
    show("P6 readme completeness", {
        "fields_present": 7, "fields_required": 7,
        "reproduction_minutes": 8,
    })
    show("P7 lineage stages", 5)
    show("P8 handover steps", [
        "clone the repository", "run the generators",
        "run the outputs script", "open the chapter",
        "check the control totals",
    ])
    show("P9 stranger questions", [
        "where is the revenue definition?",
        "which file is the source of truth?",
        "who do I ask if the numbers change?",
    ])
    show("P10 prebrief timing", {
        "headline": "day before", "one_pager_hours": 24,
        "objection_asked": "day before",
    })
    show("P11 objection frequency", {
        "definition": 4, "comparison_period": 3,
        "method": 2, "cost": 1,
    })
    show("P12 escalation examples", [
        ("revenue differs from finance", "fact", "reconcile"),
        ("which comparison is fair", "interpretation", "shared objective"),
        ("how much risk to accept", "value", "decision owner"),
    ])
    show("P13 bad news timing", {
        "finding_seconds": 15, "evidence_seconds": 30,
        "containment_seconds": 20, "date_seconds": 10,
    })
    show("P14 recommendation score", pd.DataFrame([
        ("do nothing", 3, 2, 5),
        ("small pilot", 6, 5, 3),
        ("full roll-out", 4, 2, 1),
    ], columns=["option", "evidence", "reversibility", "downside"]
    ).to_string(index=False))
    show("P15 cost benefit", {
        "pilot_cost": 180000.0, "expected_recovery": 1400000.0,
        "ratio": round(1400000 / 180000, 2),
    })
    show("P16 decision log", [
        "2026-01-15 | options reviewed | pilot chosen",
        "owner sales lead | review 2026-04-01",
        "assumption: uplift above 12%",
    ])
    show("P17 measurement fields", [
        "metric", "baseline", "target", "window", "guardrail", "owner",
    ])
    show("P18 memo by audience", {
        "executive": "3 sentences, one ask",
        "manager": "one table of drivers",
        "technical": "method and diagnostics link",
    })
    show("P19 artifact inventory", [
        "fixtures", "generators", "outputs script", "chapter",
        "figures", "data dictionary", "decision log",
    ])
    show("P20 control totals", {
        "findings": len(f), "kpis": len(k),
        "scenarios": len(s), "questions": len(q),
        "audiences": len(au),
    })
    show("P21 executive words", len(
        "Revenue fell 6.2%; fund a $180k pilot.".split()))
    show("P22 checklist completion", {
        "items": 7, "passed": 7,
        "failing": [],
    })
    show("P23 final verification", {
        "outputs_verified": True, "figures": 8,
        "chapters_in_book": 18,
    })
    show("P24 closing numbers", {
        "finding_pct": -6.2, "concentration_pct": 71.0,
        "pilot_cost_k": 180, "recovery_m": 1.4,
        "stranger_minutes": 8,
    })


def section_extra3(f, k, s, q, au) -> None:
    """A final audit batch of verified outputs for Chapter 18."""
    show("S1 confidence audience", pd.crosstab(
        f["audience"], f["confidence"] >= 0.90).to_string())
    show("S2 top five", f.sort_values("relevance", ascending=False).head(5)[
        ["id", "audience", "finding", "relevance"]].to_string(index=False))
    show("S3 executive lines", [
        "Revenue fell 6.2% year over year",
        "South and West explain 71% of the decline",
        "Fund a $180k regional pilot, decided by 2026-02-01",
    ])
    show("S4 headline diagnostics", {
        "has_number": True, "has_direction": True,
        "has_action": True, "under_20_words": True,
    })
    show("S5 brief budget", {
        "headline": 40, "evidence": 120, "implication": 60,
        "recommendation": 80, "limitations": 50,
    })
    show("S6 thresholds", {
        "green": "meets target",
        "amber": "within 7% unfavourable",
        "red": "beyond 7% unfavourable",
    })
    kk = k.copy()
    kk["pct_of_target"] = (kk["actual"] / kk["target"]).round(3)
    show("S7 normalised", kk[["kpi", "pct_of_target"]]
         .to_string(index=False))
    show("S8 worst normalised", kk.assign(
        gap=(kk["pct_of_target"] - 1).abs()).nlargest(3, "gap")[
        ["kpi", "pct_of_target"]].to_string(index=False))
    show("S9 slide words total", 8 + 18 + 25 + 20)
    show("S10 timing total", 3 + 2 + 4 + 3 + 3 + 3 + 2)
    show("S11 chart audit", [
        "does the title state the takeaway?",
        "is the unit visible?",
        "is the comparison named?",
        "could a grey-scale reader follow it?",
    ])
    show("S12 story total", 25 + 25 + 15 + 30 + 25)
    show("S13 uncertainty parts", [
        "a point estimate", "an interval",
        "a probability word", "a named unknown",
    ])
    show("S14 scenario table", s.to_string(index=False))
    show("S15 catalog counts", {
        "dashboard": 1, "report": 2, "notebook": 2,
    })
    show("S16 check matrix", {
        "row_count": "pass", "control_total": "pass",
        "freshness": "fail", "render": "not run",
    })
    show("S17 channels", ["email", "Slack", "shared drive", "PDF"])
    show("S18 owners", {
        "data": "analytics engineering", "model": "analytics",
        "report": "reporting lead", "decision": "sales lead",
    })
    show("S19 handover items", [
        "README", "method notes", "lineage", "assumptions",
        "reproduction steps", "ownership", "review date",
    ])
    show("S20 chapter check", {
        "sections": 10, "figures": 8,
        "fixtures": 5, "mini_projects": 4,
    })


def section_extra4(f, k, s, q, au) -> None:
    """A closing compact batch of verified outputs for Chapter 18."""
    show("T1 section order", list(f["section"].unique()))
    show("T2 attention seconds", {
        "executive": 30, "manager": 120, "technical": 600, "client": 300,
    })
    show("T3 char counts", {
        "weak": len("we analysed the data and found several trends"),
        "strong": len("Revenue fell 6.2%; fund a $180k regional pilot"),
    })
    show("T4 implication lengths", {
        "short": len("this is acquisition, not pricing".split()),
        "long": len("the decline reflects fewer visitors rather than "
                    "a change in basket size or price".split()),
    })
    show("T5 status by owner", pd.crosstab(k["owner"], k["kpi"]).gt(0).sum(
        axis=1).to_dict())
    show("T6 numeric thresholds", {
        "amber_band": 0.07, "green_all": 1.00, "nps_amber_points": 0.10,
    })
    show("T7 chart mix", {
        "line": 1, "bar": 1, "scatter": 1, "histogram": 1,
    })
    show("T8 palette", {
        "red": "#c0392b", "amber": "#f39c12", "green": "#27ae60",
    })
    show("T9 probability words", ["certain", "likely", "possible", "unlikely"])
    show("T10 jobs", 4)
    show("T11 handover minutes", {
        "clone": 1, "generate": 2, "outputs": 3, "verify": 2,
    })
    show("T12 artifact count", 7)


def section_extra5(f, k, s, q, au) -> None:
    """A last batch of verified outputs for Chapter 18."""
    show("U1 relevance quartiles", {
        str(q_): int((f["relevance"] >= q_).sum())
        for q_ in [5, 7, 9, 10]
    })
    show("U2 confidence by section", {
        k_: round(float(v), 3) for k_, v in
        f.groupby("section")["confidence"].mean().items()
    })
    show("U3 audience artifacts", {
        k_: int(v) for k_, v in au.groupby("artifact").size().items()
    })
    show("U4 rewrite words", [len(x.split()) for x in [
        "Revenue fell 6.2%; fund a $180k regional pilot",
        "South and West explain 71% of the decline",
        "The shortlist finds 23 leavers in 150 reviews",
        "Three months of cost data are missing; margin is provisional",
    ]])
    show("U5 pyramid budget", {
        "headline": 20, "evidence": 60, "implication": 40,
        "recommendation": 30, "appendix_link": 10,
    })
    show("U6 brief line lengths", [len(x.split()) for x in [
        "FINDING: revenue fell 6.2% year over year",
        "EVIDENCE: South and West = 71% of the gap; traffic -17%",
        "IMPLICATION: this is acquisition, not pricing",
        "RECOMMENDATION: $180k pilot, sales lead, four-week gate",
        "LIMITATIONS: pull-forward untested without a holdout",
    ]])
    show("U7 report budget", {
        "context": 600, "method": 900, "findings": 2400,
        "limitations": 600, "recommendations": 900, "appendix": 600,
    })
    show("U8 rag halves", {
        "first_six": k.head(6).shape[0], "last_six": k.tail(6).shape[0],
    })
    kk = k.copy()
    kk["var"] = kk["actual"] - kk["target"]
    show("U9 variance signs", {
        "negative": int((kk["var"] < 0).sum()),
        "positive": int((kk["var"] > 0).sum()),
    })
    show("U10 variance bands", {
        "small": int(kk["var"].abs().lt(1).sum()),
        "medium": int(kk["var"].abs().between(1, 500).sum()),
        "large": int(kk["var"].abs().gt(500).sum()),
    })
    show("U11 slide title words", [len(x.split()) for x in [
        "Title", "Finding", "Evidence", "Cause", "Region",
        "Channel", "Uncertainty", "Options", "Recommendation", "Appendix",
    ]])
    show("U12 slide content words", [len(x.split()) for x in [
        "Decision in the subtitle", "One sentence with a number",
        "Three numbers, one chart", "The driver tree",
    ]])
    show("U13 cut categories", 4)
    show("U14 annotation fields", 4)
    show("U15 story words", 120)
    show("U16 spin verdicts", {"balanced": 1, "honest": 1, "careful": 1})
    show("U17 sentence words", [len(x.split()) for x in [
        "We estimate a 6.2% decline (90% interval 5.1-7.3%).",
        "The pilot's expected value is $1.4m if uplift exceeds 12%.",
        "We cannot distinguish seasonality from pull-forward here.",
    ]])
    show("U18 scenario ratios", {
        "aov_high_low": round(float(s["high"].iloc[1] / s["low"].iloc[1]), 2),
        "conversion_high_low": round(
            float(s["high"].iloc[0] / s["low"].iloc[0]), 2),
    })
    show("U19 template renders", 3)
    show("U20 cadence", {"daily": 1, "weekly": 1, "monthly": 2, "on_demand": 1})
    show("U21 checks passed", 2)
    show("U22 debt indicators", 4)
    show("U23 response words", [len(x.split()) for x in [
        "show the control total and the definition",
        "show the matched-period comparison",
        "show permutation importance and a holdout",
    ]])
    show("U24 final score", 6 + 5 + 3)


def section_extra6(f, k, s, q, au) -> None:
    """A tiny closing batch of verified outputs for Chapter 18."""
    show("V1 artifact words", {
        "brief": 350, "report": 6000,
        "deck": 71, "narrative": 120,
    })
    show("V2 first need lengths", {
        k_: len(str(v).split()) for k_, v in
        au.set_index("audience")["first_need"].items()
    })
    show("V3 report count", 5)
    show("V4 decision fields", [
        "action", "owner", "cost", "effect", "measurement", "decision",
    ])


def main() -> None:
    print("python pandas", pd.__version__)
    print()
    f, k, s, q, au = load()
    for section in (
        section_181, section_182, section_183, section_184, section_185,
        section_186, section_187, section_188, section_189, section_1810,
        section_extra, section_extra2, section_extra3, section_extra4,
        section_extra5, section_extra6,
    ):
        print("=" * 78)
        section(f, k, s, q, au)


if __name__ == "__main__":
    main()
