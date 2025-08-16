import polars as pl
from typing import Dict, List


_optional_rules: dict[str, dict] = {
    "fstqpd30_highrisk_flag": {
        "required": ["posfstqpd30lastmonth_3976962P"],
        "expr": (lambda: (pl.col("posfstqpd30lastmonth_3976962P") > 0.2).cast(pl.Int32))()
    },
    "fpd30_flag": {
        "required": ["posfpd30lastmonth_3976960P"],
        "expr": (lambda: (pl.col("posfpd30lastmonth_3976960P") > 0.3).cast(pl.Int32))()
    },
    "dpd_ratio": {
        "required": ["pmts_dpdvalue_108P", "sumoutstandtotal_3546847A"],
        "expr": (
            lambda: pl.col("pmts_dpdvalue_108P") 
                  / (pl.col("sumoutstandtotal_3546847A") + 1e-3)
        )()
    },
    "dpd_always_gt60_24m_flag": {
        "required": ["mindbddpdlast24m_3658935P"],
        "expr": (
            lambda: (pl.col("mindbddpdlast24m_3658935P") > 60).cast(pl.Int32)
        )()
    },
    "dpd_worsening_flag": {
        "required": ["maxdpdlast3m_392P", "maxdpdlast6m_474P"],
        "expr": (
            lambda: (pl.col("maxdpdlast3m_392P") > pl.col("maxdpdlast6m_474P")).cast(pl.Int32)
        )()
    },
    "dpd_early_flag": {
        "required": ["maxdpdinstlnum_3546846P"],
        "expr": (
            lambda: (pl.col("maxdpdinstlnum_3546846P") <= 3).cast(pl.Int32)
        )()
    },
    "dpd_improvement_flag": {
       "required": ["maxdpdlast6m_474P", "maxdpdfrom6mto36m_3546853P"],
       "expr": (
           lambda: (pl.col("maxdpdlast6m_474P") < pl.col("maxdpdfrom6mto36m_3546853P")).cast(pl.Int32)
       )()
   },
   "dpd_neg_flag": {
       "required": ["maxdbddpdtollast6m_4187119P"],
       "expr": (
           lambda: (pl.col("maxdbddpdtollast6m_4187119P") < 0).cast(pl.Int32)
       )()
   },
   "actualdpd_gt60_flag": {
        "required": ["actualdpd_943P"],
        "expr": (
            lambda: (pl.col("actualdpd_943P") > 60).cast(pl.Int32)
        )()
    },
    "dpd_tolerance_ratio": {
        "required": ["actualdpd_943P", "actualdpdtolerance_344P"],
        "expr": (
            lambda: pl.col("actualdpdtolerance_344P") / (pl.col("actualdpd_943P") + 1e-3)
        )()
    },
    "dpd_3m_24m_delta": {
        "required": ["avgdbddpdlast3m_4187120P", "avgdbddpdlast24m_3658932P"],
        "expr": (
            lambda: pl.col("avgdbddpdlast3m_4187120P") - pl.col("avgdbddpdlast24m_3658932P")
        )()
    },
    "dpd_relief_ratio": {
        "required": ["maxdpdlast6m_474P", "maxdpdfrom6mto36m_3546853P"],
        "expr": (
            lambda: pl.col("maxdpdlast6m_474P") 
                  / (pl.col("maxdpdfrom6mto36m_3546853P") + 1e-3)
        )()
    },
    "dpd_worst_gt60_flag": {
        "required": ["maxdpdlast24m_143P", "maxdpdlast12m_727P"],
        "expr": (
            lambda: ((pl.col("maxdpdlast24m_143P") > 60) 
                   | (pl.col("maxdpdlast12m_727P") > 60)).cast(pl.Int32)
        )()
    },
    "dti_ratio": {
        "required": ["currdebt_22A", "maininc_215A"],
        "expr": (
            lambda: pl.col("currdebt_22A") 
                  / (pl.col("maininc_215A") + 1e-3)
        )()
    },
    "disposable_income_ratio": {
        "required": ["maininc_215A", "installmentamount_644A"],
        "expr": (
            lambda: (pl.col("maininc_215A") - pl.col("installmentamount_644A")) 
                  / (pl.col("maininc_215A") + 1e-3)
        )()
    },
    "debt_burden_ratio": {
        "required": ["currdebt_22A", "debtoutstand_525A", "maininc_215A"],
        "expr": (
            lambda: (pl.col("currdebt_22A") + pl.col("debtoutstand_525A")) 
                  / (pl.col("maininc_215A") + 1e-3)
        )()
    },
    "credit_utilization_ratio": {
        "required": ["currdebt_22A", "credacc_credlmt_575A"],
        "expr": (
            lambda: pl.col("currdebt_22A") 
                  / (pl.col("credacc_credlmt_575A") + 1e-3)
        )()
    },
    "repayment_progress_ratio": {
        "required": ["disbursedcredamount_1113A", "currdebt_22A"],
        "expr": (
            lambda: (pl.col("disbursedcredamount_1113A") - pl.col("currdebt_22A"))
                  / (pl.col("disbursedcredamount_1113A") + 1e-3)
        )()
    },
    "cc_burden_ratio": {
        "required": ["credacc_actualbalance_314A", "credacc_credlmt_575A"],
        "expr": (
            lambda: pl.col("credacc_actualbalance_314A")
                  / (pl.col("credacc_credlmt_575A") + 1e-3)
        )()
    },
    "dti_high_flag": {
        "required": ["currdebt_22A", "maininc_215A"],
        "expr": (
            lambda: ((pl.col("currdebt_22A") / (pl.col("maininc_215A") + 1e-3)) > 0.60).cast(pl.Int32)
        )()
    },
    "overdue_active_ratio": {
        "required": ["totaldebtoverduevalue_178A", "totaloutstanddebtvalue_39A"],
        "expr": (
            lambda: pl.col("totaldebtoverduevalue_178A") 
                  / (pl.col("totaloutstanddebtvalue_39A") + 1e-3)
        )()
    },
    "overdue_closed_ratio": {
        "required": ["totaldebtoverduevalue_718A", "totaloutstanddebtvalue_668A"],
        "expr": (
            lambda: pl.col("totaldebtoverduevalue_718A") 
                  / (pl.col("totaloutstanddebtvalue_668A") + 1e-3)
        )()
    },
    "payment_ability_ratio": {
        "required": ["totinstallast1m_4525188A", "sumoutstandtotal_3546847A"],
        "expr": (
            lambda: pl.col("totinstallast1m_4525188A") 
                  / (pl.col("sumoutstandtotal_3546847A") + 1e-3)
        )()
    },
    "overpaid_flag": {
        "required": ["totalsettled_863A", "totaldebt_9A"],
        "expr": (
            lambda: (pl.col("totalsettled_863A") > pl.col("totaldebt_9A")).cast(pl.Int32)
        )()
    },
    "has_revolving_account_flag": {
        "required": ["revolvingaccount_394A"],
        "expr": (
            lambda: (pl.col("revolvingaccount_394A") == 1).cast(pl.Int32)
        )()
    },
    "residual_active_ratio": {
        "required": ["residualamount_856A", "price_1097A"],
        "expr": (
            lambda: pl.col("residualamount_856A") 
                  / (pl.col("price_1097A") + 1e-3)
        )()
    },
    "residual_closed_flag": {
        "required": ["residualamount_488A"],
        "expr": (
            lambda: (pl.col("residualamount_488A") > 0).cast(pl.Int32)
        )()
    },
    "overdue_active_flag": {
        "required": ["pmts_overdue_1140A"],
        "expr": (
            lambda: (pl.col("pmts_overdue_1140A") > 0).cast(pl.Int32)
        )()
    },
    "overdue_closed_flag": {
        "required": ["pmts_overdue_1152A"],
        "expr": (
            lambda: (pl.col("pmts_overdue_1152A") > 0).cast(pl.Int32)
        )()
    },
    "payment_level_overdue_flag": {
        "required": ["pmts_pmtsoverdue_635A"],
        "expr": (
            lambda: (pl.col("pmts_pmtsoverdue_635A") > 0).cast(pl.Int32)
        )()
    },
}



def build_combo_exprs(
    lf: pl.LazyFrame,
    rules: Dict[str, dict],
) -> List[pl.Expr]:
    """
    Dynamically generate a list of usable pl.Expr based on _optional_rules.
    Three expr types are supported: str / pl.Expr / callable,
    and all are finalized with .alias(name).
    """
    cols = set(lf.columns)
    exprs: List[pl.Expr] = []

    for name, rule in rules.items():
        # If exist, skip
        if name in cols:
            continue
        # If required columns are not all present, skip
        if not all(c in cols for c in rule["required"]):
            continue

        raw = rule["expr"]
        #  Expr
        if isinstance(raw, str):
            expr = eval(raw)
        elif isinstance(raw, pl.Expr):
            expr = raw
        elif callable(raw):
            expr = raw()
        else:
            raise ValueError(f"Unsupported expr type {type(raw)} for {name}")

        if not isinstance(expr, pl.Expr):
            raise ValueError(f"Expression for {name} must be pl.Expr, got {type(expr)}")

        # name
        exprs.append(expr.alias(name))

    return exprs