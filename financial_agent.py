"""
financial_agent.py — Financial ReAct Agent
==========================================
A multi-tool agent that:

  1. Fetches stock price history (yfinance) to find the day with the
     largest open→close price swing in the last month.
  2. Retrieves relevant news from NewsAPI for that date.
  3. Translates headlines to Russian and generates an analytical summary
     via an LLM (Groq Llama-3.3-70b).

Also includes:
  • Function-call pipeline demonstration (raw Groq API)
  • LangChain tool integration example
  • SQLite text-to-SQL agent (smolagents)

Usage
-----
    export GROQ_API_KEY="gsk_..."
    export NEWS_API_KEY="..."

    python financial_agent.py --ticker AAPL --name Apple
    python financial_agent.py --demo tools
    python financial_agent.py --demo sql
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
HEADERS      = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

FAST_MODEL = "llama-3.1-8b-instant"
STRONG_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Low-level LLM call
# ---------------------------------------------------------------------------

def llm(messages: List[Dict], model: str = STRONG_MODEL, max_tokens: int = 500, temperature: float = 0.3) -> str:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Tool 1: Biggest price change (yfinance)
# ---------------------------------------------------------------------------

def get_biggest_price_change(ticker: str) -> Dict[str, Any]:
    """
    Find the trading day in the last month with the largest absolute
    open→close price difference for ticker.

    Returns
    -------
    {"ticker": str, "date": str (YYYY-MM-DD), "change": float, "open": float, "close": float}
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    df = yf.Ticker(ticker).history(period="1mo")
    if df.empty:
        return {"error": f"No data for ticker {ticker}"}

    df["abs_change"] = (df["Close"] - df["Open"]).abs()
    idx = df["abs_change"].idxmax()
    row = df.loc[idx]
    return {
        "ticker": ticker,
        "date":   idx.strftime("%Y-%m-%d"),
        "change": round(float(row["abs_change"]), 2),
        "open":   round(float(row["Open"]),  2),
        "close":  round(float(row["Close"]), 2),
    }


# ---------------------------------------------------------------------------
# Tool 2: Relevant news (NewsAPI)
# ---------------------------------------------------------------------------

def get_company_news(company_name: str, date: str, max_articles: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve up to max_articles news articles about company_name from
    NewsAPI on or before date(YYYY-MM-DD).

    Returns a list of dicts with keys "title" and "description".
    """
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={company_name}"
        f"&from={date}"
        f"&to={date}"
        f"&language=en"
        f"&sortBy=popularity"
        f"&apiKey={NEWS_API_KEY}"
    )
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        return [{"title": f"Error fetching news: {e}", "description": ""}]

    articles = []
    for a in data.get("articles", []):
        if a.get("title") and a["title"] != "[Removed]":
            articles.append({
                "title":       a["title"],
                "description": (a.get("description") or "")[:300],
            })
            if len(articles) >= max_articles:
                break

    return articles or [{"title": f"No news found for {company_name} on {date}", "description": ""}]


# ---------------------------------------------------------------------------
# Tool 3: Translation helper
# ---------------------------------------------------------------------------

def translate_to_russian(text: str) -> str:
    if not text.strip():
        return ""
    return llm(
        [{"role": "user", "content": f"Переведи на русский (только перевод, без пояснений):\n{text}"}],
        model=FAST_MODEL,
        max_tokens=300,
    )


# ---------------------------------------------------------------------------
# ReAct-style Financial Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — финансовый аналитик.
Твоя задача — найти день с наибольшим изменением цены акции за последний месяц,
выяснить новости за этот день и объяснить причины изменения цены.

Отвечай на русском языке. Будь конкретным и аргументированным.
"""

ANALYSIS_TEMPLATE = """Проанализируй следующие данные и объясни, что могло вызвать такое движение цены.

Компания: {company} ({ticker})
Дата наибольшего изменения: {date}
Цена открытия: ${open}
Цена закрытия: ${close}
Изменение: ${change}

Новости за этот день:
{news_block}

Напиши аналитический вывод: какие новости или события могли повлиять на цену акций?
Если новостей нет — предположи возможные причины на основе рыночного контекста.
"""


def financial_agent(ticker: str, company_name: str, verbose: bool = True) -> str:
    """
    Run the full financial analysis pipeline.

    Returns a formatted Russian-language report.
    """
    if verbose:
        print(f"\n[Agent] Шаг 1: Получение ценовых данных для {ticker} …")
    price_data = get_biggest_price_change(ticker)
    if "error" in price_data:
        return f"Ошибка: {price_data['error']}"

    if verbose:
        print(f"[Agent] Шаг 2: Максимальное изменение на {price_data['date']} (${price_data['change']})")
        print(f"[Agent] Шаг 3: Поиск новостей за {price_data['date']} …")

    articles = get_company_news(company_name, price_data["date"])

    if verbose:
        print(f"[Agent] Найдено новостей: {len(articles)}")
        print("[Agent] Шаг 4: Перевод и анализ …")

    # Build news block
    news_lines = []
    for a in articles:
        title_ru = translate_to_russian(a["title"])
        desc_ru  = translate_to_russian(a["description"]) if a["description"] else ""
        news_lines.append(f"• {title_ru}")
        if desc_ru:
            news_lines.append(f"  {desc_ru}")
    news_block = "\n".join(news_lines) if news_lines else "Новостей не найдено"

    prompt = ANALYSIS_TEMPLATE.format(
        company=company_name,
        ticker=ticker,
        date=price_data["date"],
        open=price_data["open"],
        close=price_data["close"],
        change=price_data["change"],
        news_block=news_block,
    )
    analysis = llm([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])

    report = (
        f"\n{'=' * 60}\n"
        f"Финансовый анализ: {company_name} ({ticker})\n"
        f"{'=' * 60}\n"
        f"Дата: {price_data['date']}\n"
        f"Открытие: ${price_data['open']}  →  Закрытие: ${price_data['close']}\n"
        f"Изменение: ${price_data['change']}\n\n"
        f"Новости:\n{news_block}\n\n"
        f"Анализ:\n{analysis}\n"
    )
    return report


# ---------------------------------------------------------------------------
# Demo: Function-call tools (raw Groq API)
# ---------------------------------------------------------------------------

GET_USER_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "get_user_info_from_db",
        "description": "Get a user's job, pets, hobbies, or city from the database by their name.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_name": {"type": "string", "description": "Name of the person to look up"}
            },
            "required": ["person_name"],
        },
    },
}

DB = {
    "ilya":   {"job": "Software Developer", "pets": "dog"},
    "farruh": {"job": "Senior Data Architect", "hobby": "travelling, hiking"},
    "timur":  {"job": "DeepSchool Founder", "city": "Novosibirsk"},
}


def get_user_info_from_db(person_name: str) -> Dict:
    return DB.get(person_name.lower(), {"error": f"No info about {person_name}"})


FUNCTION_REGISTRY = {"get_user_info_from_db": get_user_info_from_db}


def run_function_call_pipeline(question: str = "What do you know about Timur?"):
    print("\n" + "=" * 60)
    print("Demo: Function-Call Pipeline")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": question},
    ]
    payload = {
        "model": FAST_MODEL,
        "messages": messages,
        "tools": [GET_USER_INFO_TOOL],
        "tool_choice": "auto",
        "max_tokens": 200,
    }
    r = requests.post(GROQ_URL, headers=HEADERS, json=payload)
    r.raise_for_status()
    resp = r.json()

    msg = resp["choices"][0]["message"]
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        print("Model answered directly:", msg.get("content"))
        return

    # Execute tool
    tc    = tool_calls[0]
    fname = tc["function"]["name"]
    fargs = json.loads(tc["function"]["arguments"])
    result = FUNCTION_REGISTRY[fname](**fargs)
    print(f"Tool called: {fname}({fargs}) → {result}")

    # Feed result back
    messages.append(msg)
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    final = llm(messages, model=FAST_MODEL)
    print(f"Final answer: {final}")


# ---------------------------------------------------------------------------
# Demo: SQLite text-to-SQL agent
# ---------------------------------------------------------------------------

def setup_demo_db(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS jobs  (id INTEGER PRIMARY KEY, job TEXT NOT NULL)")
    conn.executemany("INSERT OR IGNORE INTO users VALUES (?,?)", [
        (1,"alice"),(2,"bob"),(3,"charlie"),(4,"dave"),(5,"eve"),
        (6,"frank"),(7,"grace"),(8,"heidi"),(9,"ivan"),(10,"judy"),
    ])
    conn.executemany("INSERT OR IGNORE INTO jobs VALUES (?,?)", [
        (1,"engineer"),(2,"designer"),(3,"manager"),(4,"developer"),(5,"analyst"),
        (6,"engineer"),(7,"support"),(8,"engineer"),(9,"engineer"),(10,"marketing"),
    ])
    conn.commit()
    return conn


def sql_query(conn: sqlite3.Connection, query: str) -> str:
    try:
        rows = conn.execute(query).fetchall()
        return "\n".join(str(r) for r in rows) if rows else "No results"
    except Exception as e:
        return f"SQL error: {e}"


def run_sql_agent():
    """
    Simple ReAct-style loop: ask the LLM to write SQL, execute it, feed
    results back until the model produces a final answer.
    """
    print("\n" + "=" * 60)
    print("Demo: Text-to-SQL Agent")
    print("=" * 60)

    conn = setup_demo_db()
    question = ("Find the most popular job title and list the names of all "
                "users who hold that job.")

    system = (
        "You have access to a SQLite database with two tables:\n"
        "  users(id, username)\n"
        "  jobs(id, job)\n"
        "Joined on id.\n\n"
        "When you need to query the database, write ONLY a JSON object like:\n"
        '  {"sql": "SELECT ..."}\n'
        "When you have the final answer, write it in plain text."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": question},
    ]

    for step in range(6):
        reply = llm(messages, model=FAST_MODEL, max_tokens=300, temperature=0)
        messages.append({"role": "assistant", "content": reply})
        print(f"\n[Step {step+1}] Model: {reply[:200]}")

        # Try to extract SQL
        try:
            obj = json.loads(reply)
            if "sql" in obj:
                result = sql_query(conn, obj["sql"])
                print(f"         DB result: {result}")
                messages.append({"role": "user", "content": f"Query result:\n{result}"})
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        # No SQL → model gave final answer
        print(f"\n[Answer] {reply}")
        break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--name",   default="Apple")
    parser.add_argument("--demo",   choices=["tools", "sql"], default=None)
    args = parser.parse_args()

    if args.demo == "tools":
        run_function_call_pipeline()
    elif args.demo == "sql":
        run_sql_agent()
    else:
        if not GROQ_API_KEY:
            print("⚠  Set GROQ_API_KEY")
        if not NEWS_API_KEY:
            print("⚠  Set NEWS_API_KEY (free at newsapi.org)")
        report = financial_agent(args.ticker, args.name)
        print(report)
