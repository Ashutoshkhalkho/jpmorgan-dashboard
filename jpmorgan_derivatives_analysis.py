"""
============================================================
  JP Morgan Chase - Stock & Derivatives Dashboard
  Skills: Python, Data Analysis, Statistics, Algorithm Development,
          Credit Analysis, Financial Derivatives, Critical Thinking
  Author: Ashutosh | JP Morgan Job Simulation Graduate
============================================================
"""

import math
import random
import statistics
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. DATA GENERATION — Simulated Stock Data
# ─────────────────────────────────────────────

def generate_stock_data(ticker: str, days: int = 252, start_price: float = 100.0) -> list[dict]:
    """
    Generates simulated daily OHLCV stock data using geometric Brownian motion.
    Demonstrates: Algorithm Development, Statistics, Python Programming
    """
    random.seed(42)
    data = []
    price = start_price
    mu = 0.0002       # Daily drift (mean return)
    sigma = 0.015     # Daily volatility

    for i in range(days):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        shock = random.gauss(0, 1)
        daily_return = math.exp((mu - 0.5 * sigma**2) + sigma * shock)
        price *= daily_return

        high = price * random.uniform(1.001, 1.02)
        low  = price * random.uniform(0.98, 0.999)
        volume = random.randint(1_000_000, 5_000_000)

        data.append({
            "date":   date.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "open":   round(price * random.uniform(0.998, 1.002), 2),
            "high":   round(high, 2),
            "low":    round(low, 2),
            "close":  round(price, 2),
            "volume": volume,
        })

    return data


# ─────────────────────────────────────────────
# 2. STATISTICAL ANALYSIS MODULE
# ─────────────────────────────────────────────

def compute_statistics(data: list[dict]) -> dict:
    """
    Computes key statistical metrics on stock closing prices.
    Demonstrates: Statistics, Data Analysis, Critical Thinking
    """
    closes = [d["close"] for d in data]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]

    mean_return     = statistics.mean(returns)
    std_return      = statistics.stdev(returns)
    variance        = statistics.variance(returns)
    sharpe_ratio    = (mean_return / std_return) * math.sqrt(252) if std_return != 0 else 0
    max_drawdown    = _compute_max_drawdown(closes)
    annualized_vol  = std_return * math.sqrt(252)
    total_return    = (closes[-1] - closes[0]) / closes[0] * 100

    return {
        "mean_daily_return":   round(mean_return * 100, 4),
        "std_daily_return":    round(std_return * 100, 4),
        "variance":            round(variance * 10000, 6),
        "sharpe_ratio":        round(sharpe_ratio, 3),
        "annualized_volatility": round(annualized_vol * 100, 2),
        "max_drawdown_pct":    round(max_drawdown * 100, 2),
        "total_return_pct":    round(total_return, 2),
        "min_price":           round(min(closes), 2),
        "max_price":           round(max(closes), 2),
        "current_price":       round(closes[-1], 2),
    }


def _compute_max_drawdown(prices: list[float]) -> float:
    """Calculates the maximum peak-to-trough drawdown."""
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ─────────────────────────────────────────────
# 3. MOVING AVERAGES ALGORITHM
# ─────────────────────────────────────────────

def compute_moving_averages(data: list[dict], windows: list[int] = [20, 50]) -> list[dict]:
    """
    Computes Simple Moving Averages (SMA) for given windows.
    Demonstrates: Algorithm Development, Python Programming
    """
    closes = [d["close"] for d in data]
    result = []

    for i, row in enumerate(data):
        entry = {"date": row["date"], "close": row["close"]}
        for w in windows:
            if i >= w - 1:
                entry[f"SMA_{w}"] = round(statistics.mean(closes[i-w+1:i+1]), 2)
            else:
                entry[f"SMA_{w}"] = None
        result.append(entry)

    return result


def generate_signals(ma_data: list[dict]) -> list[dict]:
    """
    Golden Cross / Death Cross signal generator.
    Demonstrates: Algorithm Development, Critical Thinking
    """
    signals = []
    for i in range(1, len(ma_data)):
        row = ma_data[i]
        prev = ma_data[i-1]
        signal = "HOLD"

        if (row.get("SMA_20") and row.get("SMA_50") and
            prev.get("SMA_20") and prev.get("SMA_50")):
            if prev["SMA_20"] < prev["SMA_50"] and row["SMA_20"] > row["SMA_50"]:
                signal = "BUY 🟢 (Golden Cross)"
            elif prev["SMA_20"] > prev["SMA_50"] and row["SMA_20"] < row["SMA_50"]:
                signal = "SELL 🔴 (Death Cross)"

        if signal != "HOLD":
            signals.append({"date": row["date"], "price": row["close"], "signal": signal})

    return signals


# ─────────────────────────────────────────────
# 4. BLACK-SCHOLES OPTIONS PRICING
# ─────────────────────────────────────────────

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Black-Scholes model for European options pricing.
    Demonstrates: Financial Derivatives, Algorithm Development, Statistics

    Args:
        S     : Current stock price
        K     : Strike price
        T     : Time to expiration (years)
        r     : Risk-free rate (decimal)
        sigma : Volatility (decimal)
        option_type : 'call' or 'put'
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    def norm_cdf(x):
        """Approximation of the standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if option_type == "call":
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    # Greeks
    delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
    gamma = math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
    theta = (-(S * sigma * math.exp(-d1**2 / 2)) / (2 * math.sqrt(T) * math.sqrt(2 * math.pi))
             - r * K * math.exp(-r * T) * norm_cdf(d2 if option_type == "call" else -d2)) / 365
    vega  = S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi) / 100

    return {
        "option_type":  option_type.upper(),
        "stock_price":  S,
        "strike_price": K,
        "time_to_exp":  f"{T} years",
        "risk_free_r":  f"{r*100}%",
        "volatility":   f"{sigma*100}%",
        "option_price": round(price, 4),
        "delta":        round(delta, 4),
        "gamma":        round(gamma, 6),
        "theta":        round(theta, 4),
        "vega":         round(vega, 4),
    }


# ─────────────────────────────────────────────
# 5. CREDIT RISK SCORING
# ─────────────────────────────────────────────

def credit_risk_score(credit_score: int, debt_to_income: float,
                      loan_amount: float, annual_income: float,
                      employment_years: int) -> dict:
    """
    Custom credit risk scoring algorithm.
    Demonstrates: Credit Analysis, Algorithm Development, Critical Thinking
    """
    score = 0

    # Credit Score component (max 40 pts)
    if credit_score >= 750: score += 40
    elif credit_score >= 700: score += 30
    elif credit_score >= 650: score += 20
    elif credit_score >= 600: score += 10
    else: score += 0

    # DTI component (max 25 pts)
    if debt_to_income < 0.20: score += 25
    elif debt_to_income < 0.35: score += 15
    elif debt_to_income < 0.43: score += 8
    else: score += 0

    # Loan-to-Income ratio (max 20 pts)
    lti = loan_amount / annual_income
    if lti < 2: score += 20
    elif lti < 3.5: score += 12
    elif lti < 5: score += 5
    else: score += 0

    # Employment stability (max 15 pts)
    if employment_years >= 5: score += 15
    elif employment_years >= 2: score += 10
    elif employment_years >= 1: score += 5
    else: score += 0

    # Risk classification
    if score >= 80: risk = "LOW RISK ✅"
    elif score >= 55: risk = "MODERATE RISK ⚠️"
    elif score >= 35: risk = "HIGH RISK 🔶"
    else: risk = "VERY HIGH RISK 🔴"

    return {
        "composite_score":    score,
        "max_score":          100,
        "risk_classification": risk,
        "credit_score_input": credit_score,
        "debt_to_income":     f"{debt_to_income*100:.1f}%",
        "loan_to_income":     round(loan_amount / annual_income, 2),
        "employment_years":   employment_years,
        "recommended_action": "APPROVE" if score >= 55 else "REJECT / REVIEW",
    }


# ─────────────────────────────────────────────
# 6. MAIN — FULL ANALYSIS RUN
# ─────────────────────────────────────────────

def run_full_analysis():
    print("=" * 60)
    print("  JP MORGAN CHASE — STOCK & DERIVATIVES DASHBOARD")
    print("  Skills Showcase | Ashutosh")
    print("=" * 60)

    tickers = ["AAPL", "JPM", "TSLA"]
    for ticker in tickers:
        print(f"\n{'─'*50}")
        print(f"  📈 STOCK: {ticker}")
        print(f"{'─'*50}")

        data   = generate_stock_data(ticker, days=252, start_price=random.uniform(80, 300))
        stats  = compute_statistics(data)
        ma_data = compute_moving_averages(data)
        signals = generate_signals(ma_data)

        print(f"  Current Price    : ${stats['current_price']}")
        print(f"  Total Return     : {stats['total_return_pct']}%")
        print(f"  Sharpe Ratio     : {stats['sharpe_ratio']}")
        print(f"  Annualized Vol   : {stats['annualized_volatility']}%")
        print(f"  Max Drawdown     : {stats['max_drawdown_pct']}%")
        print(f"  Mean Daily Ret   : {stats['mean_daily_return']}%")

        print(f"\n  Trading Signals ({len(signals)} found):")
        for s in signals[:3]:
            print(f"    {s['date']} | ${s['price']} | {s['signal']}")

    # Options Pricing
    print(f"\n{'─'*50}")
    print("  📊 BLACK-SCHOLES OPTIONS PRICING")
    print(f"{'─'*50}")
    for opt_type in ["call", "put"]:
        result = black_scholes(S=150, K=155, T=0.5, r=0.05, sigma=0.25, option_type=opt_type)
        print(f"\n  {result['option_type']} OPTION:")
        for k, v in result.items():
            print(f"    {k:<18}: {v}")

    # Credit Risk
    print(f"\n{'─'*50}")
    print("  🏦 CREDIT RISK ANALYSIS")
    print(f"{'─'*50}")
    applicants = [
        {"name": "Client A", "credit_score": 760, "dti": 0.22, "loan": 200000, "income": 90000, "emp": 6},
        {"name": "Client B", "credit_score": 620, "dti": 0.45, "loan": 350000, "income": 60000, "emp": 1},
    ]
    for a in applicants:
        result = credit_risk_score(a["credit_score"], a["dti"], a["loan"], a["income"], a["emp"])
        print(f"\n  {a['name']}:")
        print(f"    Score      : {result['composite_score']}/100")
        print(f"    Risk Level : {result['risk_classification']}")
        print(f"    Decision   : {result['recommended_action']}")

    print(f"\n{'='*60}")
    print("  ✅ Analysis Complete | JP Morgan Job Simulation")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_full_analysis()
