# Financial ReAct Agent

An LLM-powered financial analyst that autonomously researches stock price anomalies and explains them with news context.

## Architecture

```
User query (ticker + company name)
        │
        ▼
┌───────────────────────────┐
│  Tool 1: yfinance          │  → biggest open→close swing in last month
└───────────────────────────┘
        │  date + Δprice
        ▼
┌───────────────────────────┐
│  Tool 2: NewsAPI           │  → up to 5 relevant articles for that date
└───────────────────────────┘
        │  headlines + descriptions
        ▼
┌───────────────────────────┐
│  Tool 3: LLM Translation   │  → Russian headlines
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  LLM Analysis              │  → Groq Llama-3.3-70b analytical summary
└───────────────────────────┘
```

## Included demos

| Demo | Description |
|------|-------------|
| Default | Full financial analysis for any stock ticker |
| `--demo tools` | Raw Groq function-call pipeline walkthrough |
| `--demo sql` | Text-to-SQL ReAct agent on a SQLite database |

## Setup

```bash
pip install yfinance requests groq
export GROQ_API_KEY="gsk_..."
export NEWS_API_KEY="..."   # Free at https://newsapi.org/
```

## Usage

```bash
# Analyse Apple stock
python financial_agent.py --ticker AAPL --name Apple

# Analyse Tesla
python financial_agent.py --ticker TSLA --name Tesla

# Function-call demo
python financial_agent.py --demo tools

# SQL agent demo
python financial_agent.py --demo sql
```

## Example output

```
==============================
Финансовый анализ: Apple (AAPL)
==============================
Дата: 2025-05-09
Открытие: $196.40  →  Закрытие: $210.62
Изменение: $14.22

Новости:
• Apple сообщила о рекордной выручке за квартал
  Выручка выросла на 8% год к году благодаря …

Анализ:
Значительный рост цены акций Apple 9 мая 2025 года объясняется выходом
квартального отчёта, превысившего ожидания аналитиков …
```

## Function-call design

Tools are described as JSON schemas passed to the model.  
The model's `tool_calls` response is parsed, executed locally, and fed back into the conversation before the final generation — demonstrating the full ReAct (Reason + Act) loop.
