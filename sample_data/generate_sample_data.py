"""
InsightAI - Sample data generator.

Produces three realistic datasets for demonstration and testing:
  * sales.csv            (Sales Analytics - time series + categories)
  * ecommerce.csv        (E-commerce Analytics - products, orders, text reviews)
  * customer_feedback.csv (Customer support feedback - text sentiment)

Run:  python sample_data/generate_sample_data.py
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
random.seed(7)
np.random.seed(7)


def gen_sales(n: int = 10000) -> pd.DataFrame:
    regions = ["North", "South", "East", "West"]
    products = ["Widget A", "Widget B", "Widget C", "Gadget X", "Gadget Y", "Service Plan"]
    segments = ["Enterprise", "SMB", "Consumer"]
    channels = ["Online", "Retail", "Partners"]
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, 900, n)]
    base_price = np.random.uniform(5, 500, n)
    qty = np.random.poisson(2, n) + 1
    revenue = base_price * qty
    revenue[-50:] *= np.random.uniform(3.0, 5.0, 50)  # inject a few spikes
    df = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "order_date": dates,
        "region": np.random.choice(regions, n, p=[0.2, 0.3, 0.25, 0.25]),
        "product": np.random.choice(products, n, p=[0.25, 0.2, 0.15, 0.2, 0.1, 0.1]),
        "segment": np.random.choice(segments, n, p=[0.4, 0.35, 0.25]),
        "channel": np.random.choice(channels, n, p=[0.5, 0.3, 0.2]),
        "quantity": qty,
        "unit_price": np.round(base_price, 2),
        "revenue": np.round(revenue, 2),
        "discount_pct": np.round(np.random.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.3, 0.12, 0.06, 0.02]), 1),
    })
    df.loc[random.sample(range(n), 300), "region"] = None
    df.loc[random.sample(range(n), 120), "revenue"] = None
    return df


def gen_ecommerce(n: int = 8000) -> pd.DataFrame:
    categories = ["Electronics", "Clothing", "Home", "Beauty", "Books", "Sports"]
    brands = ["Brand Alpha", "Brand Beta", "Brand Gamma", "Brand Delta"]
    cities = ["Lahore", "Karachi", "Islamabad", "Dubai", "London", "New York"]
    reviews = {
        "Electronics": ["excellent quality", "works great", "battery dies fast", "poor build", "love it",
                        "average at best", "highly recommend", "disappointed with durability"],
        "Clothing": ["fits perfectly", "nice fabric", "too small", "great value", "seams came apart",
                     "comfortable and stylish", "color faded after wash"],
        "Home": ["looks premium", "assembly was hard", "very useful", "cheap materials", "would buy again"],
        "Beauty": ["skin felt great", "caused irritation", "nice scent", "too expensive", "recommend"],
        "Books": ["well written", "fast delivery", "print quality good", "boring content", "informative"],
        "Sports": ["good grip", "broke after week", "great for training", "poor durability"],
    }
    start = datetime(2023, 6, 1)
    dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, 600, n)]
    cat = np.random.choice(categories, n)
    data = {
        "order_id": [f"E-{i:07d}" for i in range(1, n + 1)],
        "order_date": dates,
        "category": cat,
        "brand": np.random.choice(brands, n),
        "city": np.random.choice(cities, n),
        "customer_rating": np.clip(np.random.normal(3.8, 1.1, n), 1, 5).astype(int),
        "price": np.round(np.random.uniform(10, 400, n), 2),
        "quantity": np.random.poisson(1, n) + 1,
        "review": [random.choice(reviews[c]) for c in cat],
        "is_active": np.random.choice([True, False], n, p=[0.85, 0.15]),
        "customer_id": [f"C-{random.randint(1, 1500):d}" for _ in range(n)],
    }
    df = pd.DataFrame(data)
    df.loc[random.sample(range(n), 200), "review"] = None
    df.loc[random.sample(range(n), 150), "price"] = None
    return df


def gen_feedback(n: int = 1500) -> pd.DataFrame:
    support_topics = ["Billing", "Shipping", "Product", "Account", "Technical"]
    positive = ["excellent service", "very helpful agent", "quick resolution", "loved the support",
                "best experience", "great communication", "resolved fast"]
    negative = ["terrible experience", "never received my order", "rude agent", "billing gone wrong",
                "app keeps crashing", "very slow refund", "worst support", "unacceptable wait time"]
    neutral = ["okay experience", "it was fine", "standard support", "nothing special"]

    def rand_text():
        pool = positive if random.random() < 0.4 else (negative if random.random() < 0.35 else neutral)
        return random.choice(pool)

    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, 240, n)]
    df = pd.DataFrame({
        "ticket_id": [f"TK-{i:05d}" for i in range(1, n + 1)],
        "created_at": dates,
        "topic": np.random.choice(support_topics, n),
        "channel": np.random.choice(["Email", "Chat", "Phone", "Social"], n),
        "region": np.random.choice(["NA", "EU", "APAC", "MEA"], n),
        "satisfaction_score": np.clip(np.random.normal(3.4, 1.3, n), 1, 5).astype(int),
        "response_text": [rand_text() for _ in range(n)],
    })
    return df


def write_all():
    sales = gen_sales()
    ecom = gen_ecommerce()
    fb = gen_feedback()
    sales.to_csv(HERE / "sales.csv", index=False)
    ecom.to_csv(HERE / "ecommerce.csv", index=False)
    fb.to_csv(HERE / "customer_feedback.csv", index=False)
    print(f"sales.csv          {sales.shape} -> {HERE/'sales.csv'}")
    print(f"ecommerce.csv      {ecom.shape} -> {HERE/'ecommerce.csv'}")
    print(f"customer_feedback.csv {fb.shape} -> {HERE/'customer_feedback.csv'}")


if __name__ == "__main__":
    write_all()
