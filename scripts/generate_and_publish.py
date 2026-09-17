import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from google import genai
from google.genai import errors


# ============================================================
# CONFIG
# ============================================================

DEVTO_API = "https://dev.to/api"
DEVTO_ARTICLES = f"{DEVTO_API}/articles"

# Primary + fallback models.
# If the primary model is temporarily unavailable (503),
# the script retries before falling back.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]

HISTORY_FILE = Path("data/topic_history.json")

MAX_TRENDING_ARTICLES = 17
MAX_HISTORY = 100
MAX_HISTORY_FOR_PROMPT = 30

MAX_RETRIES_PER_MODEL = 3

MIN_ARTICLE_LENGTH = 1000
TARGET_ARTICLE_MIN = 1500
TARGET_ARTICLE_MAX = 2500


# ============================================================
# ENVIRONMENT
# ============================================================

DEVTO_API_KEY = os.getenv("DEVTO_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not DEVTO_API_KEY:
    print("ERROR: DEVTO_API_KEY is missing")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is missing")
    sys.exit(1)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# TOPIC HISTORY
# ============================================================

def load_history():
    """
    Load previously published topics.

    If the file is missing, empty, malformed, or not a JSON
    array, start with an empty history rather than failing.
    """

    if not HISTORY_FILE.exists():
        print("Topic history does not exist. Starting fresh.")
        return []

    try:
        raw = HISTORY_FILE.read_text(encoding="utf-8").strip()

        if not raw:
            print("Topic history is empty. Starting fresh.")
            return []

        data = json.loads(raw)

        if not isinstance(data, list):
            print(
                "Topic history is not a JSON array. "
                "Starting fresh."
            )
            return []

        print(
            f"Loaded {len(data)} previous published topics."
        )

        return data

    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Could not read topic history: {exc}"
        )
        print("Starting with empty topic history.")
        return []


def save_history(history):
    """
    Persist topic history as a real JSON array.
    """

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = history[-MAX_HISTORY:]

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )


# ============================================================
# DEV.TO
# ============================================================

def get_trending_articles():
    """
    Fetch currently rising DEV.to articles.
    """

    print("Fetching DEV.to rising articles...")

    response = requests.get(
        DEVTO_ARTICLES,
        params={
            "state": "rising",
            "per_page": MAX_TRENDING_ARTICLES
        },
        headers={
            "api-key": DEVTO_API_KEY
        },
        timeout=30
    )

    response.raise_for_status()

    articles = response.json()

    print(
        f"Found {len(articles)} rising DEV.to articles."
    )

    return articles


# ============================================================
# TREND PREPARATION
# ============================================================

def prepare_topics(articles):
    """
    Reduce DEV.to API response to the fields useful for topic
    analysis.
    """

    topics = []

    for article in articles:
        topics.append(
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "tags": article.get("tag_list", []),
                "reactions": article.get(
                    "positive_reactions_count",
                    0
                ),
                "comments": article.get(
                    "comments_count",
                    0
                ),
                "published_at": article.get(
                    "published_at"
                ),
            }
        )

    return topics


# ============================================================
# PROMPT
# ============================================================

def build_prompt(topics, history):

    # Keep prompt reasonably small.
    topic_text = json.dumps(
        topics,
        indent=2,
        ensure_ascii=False
    )

    history_text = json.dumps(
        history[-MAX_HISTORY_FOR_PROMPT:],
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are an experienced Staff Software Engineer and
technical writer creating an original article for DEV.to.

You have strong knowledge of:

- React
- JavaScript
- TypeScript
- Frontend architecture
- Micro Frontends
- Module Federation
- System Design
- Distributed Systems
- Go
- AI
- LLMs
- AI Agents
- MCP
- Developer Productivity
- Software Architecture

Your task is to analyze CURRENT DEV.to rising articles,
identify the strongest relevant trend, and create ONE
original article based on that trend.

IMPORTANT ORIGINALITY RULES:

Do NOT copy or closely rewrite any existing article.

Do NOT reuse:

- another article's title
- sentences
- paragraph structure
- examples
- code
- conclusions

Instead:

1. Identify the underlying trend.
2. Choose a distinct engineering angle.
3. Add substantially different technical value.
4. Explain the subject deeply enough to teach an engineer.

============================================================
CURRENT DEV.TO RISING ARTICLES
============================================================

{topic_text}

============================================================
PREVIOUSLY PUBLISHED TOPICS
============================================================

Avoid repeating these topics unless there is clearly a
new and substantially different angle.

{history_text}

============================================================
CONTENT PREFERENCES
============================================================

Prioritize topics that are:

- currently trending
- technically useful
- practical
- relevant to software engineers
- useful to experienced engineers
- useful to Staff+ engineers
- relevant to modern frontend/backend engineering
- relevant to AI engineering

Potential areas include:

AI coding agents
AI-assisted development
LLM architecture
Agentic systems
MCP
RAG
AI coding workflows
React architecture
Micro Frontends
Frontend performance
System Design
Distributed Systems
Event-driven architecture
Developer productivity
Software architecture
Engineering leadership

Do not force one of these topics if the current DEV.to
trend points somewhere else.

============================================================
ARTICLE REQUIREMENTS
============================================================

Target approximately {TARGET_ARTICLE_MIN}-{TARGET_ARTICLE_MAX}
words.

The article should include:

1. Strong, specific title
2. Compelling opening
3. The engineering problem
4. Clear explanation of the concept
5. How it works
6. Practical examples
7. Code examples when useful
8. Mermaid diagrams when useful
9. Real-world trade-offs
10. Common mistakes
11. When to use it
12. When NOT to use it
13. Practical recommendations
14. Strong conclusion

Avoid generic AI-generated filler.

Prefer:

- concrete examples
- engineering reasoning
- architectural decisions
- trade-offs
- implementation details
- diagrams
- code
- practical recommendations

The article should sound like a thoughtful senior engineer
teaching another engineer.

Do NOT claim personal experience that was not provided.

Do NOT write things such as:

"As a Staff Engineer, I..."

unless the statement is actually supported by provided
information.

Do NOT mention:

- AI generated content
- this prompt
- the automation
- the DEV.to source articles
- content generation

Do NOT use clickbait.

============================================================
DEV.TO METADATA
============================================================

Generate:

title
description
tags

Use 3-5 relevant DEV.to tags.

Tags must:

- be lowercase
- contain only letters/numbers
- be suitable for DEV.to

Examples:

javascript
react
ai
systemdesign
webdev

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not wrap the JSON in Markdown fences.

Use EXACTLY this structure:

{{
  "title": "...",
  "description": "...",
  "tags": ["...", "...", "..."],
  "body_markdown": "..."
}}
"""


# ============================================================
# GEMINI RETRY / FALLBACK
# ============================================================

def generate_with_retry(prompt):
    """
    Generate content with retries and model fallback.

    503 errors are treated as temporary availability failures.
    """

    last_error = None

    for model in GEMINI_MODELS:

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):

            try:
                print(
                    f"Calling Gemini: "
                    f"{model} "
                    f"(attempt {attempt}/{MAX_RETRIES_PER_MODEL})"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.8,
                        "response_mime_type": "application/json",
                    },
                )

                if not response or not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                print(
                    f"Gemini generation succeeded using {model}."
                )

                return response.text

            except errors.ServerError as exc:
                last_error = exc

                print(
                    f"Gemini returned server error "
                    f"{getattr(exc, 'code', 'unknown')}: {exc}"
                )

                if attempt < MAX_RETRIES_PER_MODEL:

                    # 10s, 20s, 40s
                    delay = 10 * (2 ** (attempt - 1))

                    print(
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(delay)

            except errors.APIError as exc:
                last_error = exc

                status_code = getattr(
                    exc,
                    "code",
                    None
                )

                print(
                    f"Gemini API error "
                    f"{status_code}: {exc}"
                )

                # Only retry likely temporary errors.
                retryable = status_code in {
                    429,
                    500,
                    502,
                    503,
                    504
                }

                if retryable and attempt < MAX_RETRIES_PER_MODEL:

                    delay = 10 * (2 ** (attempt - 1))

                    print(
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(delay)

                else:
                    break

            except Exception as exc:
                print(
                    f"Unexpected Gemini error: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

        print(
            f"Model {model} failed after "
            f"{MAX_RETRIES_PER_MODEL} attempts."
        )

        print(
            "Trying fallback model..."
        )

    raise RuntimeError(
        "All Gemini models failed."
    ) from last_error


# ============================================================
# RESPONSE CLEANUP
# ============================================================

def clean_json_response(text):
    """
    Remove common Markdown code fencing accidentally returned
    by the model.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()

    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ============================================================
# ARTICLE GENERATION
# ============================================================

def generate_article(topics, history):

    prompt = build_prompt(
        topics,
        history
    )

    print("Generating article with Gemini...")

    raw_text = generate_with_retry(prompt)

    text = clean_json_response(raw_text)

    try:
        article = json.loads(text)

    except json.JSONDecodeError as exc:

        print(
            "Gemini returned invalid JSON."
        )

        print(
            "Raw response:"
        )

        print(text)

        raise ValueError(
            "Gemini response could not be parsed as JSON."
        ) from exc

    required_fields = [
        "title",
        "description",
        "tags",
        "body_markdown"
    ]

    for field in required_fields:

        if field not in article:
            raise ValueError(
                f"Generated article is missing: {field}"
            )

    return article


# ============================================================
# ARTICLE VALIDATION
# ============================================================

def validate_article(article):

    if not isinstance(article, dict):
        raise ValueError(
            "Generated article is not an object."
        )

    title = article.get("title")

    description = article.get("description")

    tags = article.get("tags")

    body = article.get("body_markdown")

    if not isinstance(title, str):
        raise ValueError(
            "Article title must be a string."
        )

    if not isinstance(description, str):
        raise ValueError(
            "Article description must be a string."
        )

    if not isinstance(body, str):
        raise ValueError(
            "Article body must be a string."
        )

    if not isinstance(tags, list):
        raise ValueError(
            "Article tags must be an array."
        )

    title = title.strip()
    description = description.strip()
    body = body.strip()

    if len(title) < 10:
        raise ValueError(
            "Article title is suspiciously short."
        )

    if len(description) < 20:
        raise ValueError(
            "Article description is suspiciously short."
        )

    if len(body) < MIN_ARTICLE_LENGTH:
        raise ValueError(
            f"Article body is too short "
            f"({len(body)} characters)."
        )

    # Keep maximum of five tags.
    tags = tags[:5]

    cleaned_tags = []

    for tag in tags:

        if not isinstance(tag, str):
            continue

        tag = tag.strip().lower()

        # DEV.to-friendly tags only.
        if not re.fullmatch(r"[a-z0-9]+", tag):
            continue

        if tag not in cleaned_tags:
            cleaned_tags.append(tag)

    if not cleaned_tags:
        raise ValueError(
            "Article has no valid DEV.to tags."
        )

    article["title"] = title
    article["description"] = description
    article["body_markdown"] = body
    article["tags"] = cleaned_tags[:5]

    print()
    print("Article validation passed.")
    print()
    print("TITLE:")
    print(title)
    print()
    print("DESCRIPTION:")
    print(description)
    print()
    print("TAGS:")
    print(", ".join(article["tags"]))
    print()
    print(
        f"BODY LENGTH: {len(body)} characters"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate_title(title, history):

    normalized_title = title.strip().lower()

    for item in history:

        previous_title = item.get(
            "title",
            ""
        )

        if (
            isinstance(previous_title, str)
            and previous_title.strip().lower()
            == normalized_title
        ):
            return True

    return False


# ============================================================
# PUBLISH
# ============================================================

def publish_article(article):

    payload = {
        "article": {
            "title": article["title"],
            "description": article["description"],
            "body_markdown": article["body_markdown"],
            "published": True,
            "tags": article["tags"][:5]
        }
    }

    print()
    print("Publishing article to DEV.to...")

    response = requests.post(
        DEVTO_ARTICLES,
        headers={
            "api-key": DEVTO_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    if response.status_code not in (200, 201):

        print(
            f"DEV.to API error: "
            f"{response.status_code}"
        )

        print(response.text)

        sys.exit(1)

    result = response.json()

    print()
    print("========================================")
    print("ARTICLE PUBLISHED")
    print("========================================")

    print(
        f"Title: {result.get('title')}"
    )

    print(
        f"URL:   {result.get('url')}"
    )

    print(
        f"ID:    {result.get('id')}"
    )

    print("========================================")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "      DAILY DEV.TO AUTO PUBLISHER"
    )

    print(
        "========================================"
    )

    print()

    # --------------------------------------------------------
    # Load history
    # --------------------------------------------------------

    history = load_history()

    # --------------------------------------------------------
    # Get current DEV.to trends
    # --------------------------------------------------------

    articles = get_trending_articles()

    if not articles:
        raise RuntimeError(
            "DEV.to returned no rising articles."
        )

    topics = prepare_topics(
        articles
    )

    # --------------------------------------------------------
    # Generate article
    # --------------------------------------------------------

    article = generate_article(
        topics,
        history
    )

    # --------------------------------------------------------
    # Validate article
    # --------------------------------------------------------

    validate_article(
        article
    )

    # --------------------------------------------------------
    # Avoid exact duplicate titles
    # --------------------------------------------------------

    if is_duplicate_title(
        article["title"],
        history
    ):
        raise RuntimeError(
            "Generated article title already exists "
            "in topic history. Refusing to publish."
        )

    # --------------------------------------------------------
    # Publish
    # --------------------------------------------------------

    result = publish_article(
        article
    )

    # --------------------------------------------------------
    # Save history ONLY after successful publish
    # --------------------------------------------------------

    history.append(
        {
            "title": article["title"],
            "tags": article["tags"],
            "url": result.get("url"),
            "devto_id": result.get("id"),
        }
    )

    save_history(
        history
    )

    print()
    print(
        "Topic history updated successfully."
    )

    print(
        "Daily DEV.to publishing completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "Execution interrupted."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print(
            "========================================"
        )
        print(
            "WORKFLOW FAILED"
        )
        print(
            "========================================"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "========================================"
        )

        sys.exit(1)
