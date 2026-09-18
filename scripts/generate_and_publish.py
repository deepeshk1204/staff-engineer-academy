import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from google import genai
from google.genai import errors, types


# ============================================================
# CONFIG
# ============================================================

DEVTO_API = "https://dev.to/api"
DEVTO_ARTICLES = f"{DEVTO_API}/articles"

# Model order.
#
# If the primary model is temporarily unavailable, the script
# retries it and then moves to the fallback model.
#
# You can change this order later if another model proves more
# reliable for your API quota.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

HISTORY_FILE = Path("data/topic_history.json")

MAX_TRENDING_ARTICLES = 17

MAX_HISTORY = 100
MAX_HISTORY_FOR_PROMPT = 30

# Gemini availability retry configuration.
MAX_RETRIES_PER_MODEL = 3

# Retry the entire article generation if Gemini returns
# incomplete/truncated Markdown.
MAX_ARTICLE_GENERATION_ATTEMPTS = 3

# Minimum article size accepted by the publisher.
MIN_ARTICLE_LENGTH = 1000

# Keep articles practical rather than unnecessarily long.
TARGET_ARTICLE_MIN = 1400
TARGET_ARTICLE_MAX = 1800

# Explicit output budget.
#
# This is deliberately much larger than the expected article
# size because Markdown code blocks and Mermaid diagrams can
# consume additional tokens.
ARTICLE_MAX_OUTPUT_TOKENS = 12000

# Metadata is tiny, so a much smaller budget is sufficient.
METADATA_MAX_OUTPUT_TOKENS = 500


# ============================================================
# GEMINI METADATA SCHEMA
# ============================================================

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
        },
        "description": {
            "type": "string",
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "title",
        "description",
        "tags",
    ],
}


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
    Load previously published topic history.

    If the file is missing, empty, malformed, or not a JSON
    array, start with an empty history instead of failing.
    """

    if not HISTORY_FILE.exists():
        print(
            "Topic history does not exist. Starting fresh."
        )
        return []


    try:
        raw = HISTORY_FILE.read_text(
            encoding="utf-8"
        ).strip()


        if not raw:
            print(
                "Topic history is empty. Starting fresh."
            )
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


    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        print(
            f"Could not read topic history: {exc}"
        )

        print(
            "Starting with empty topic history."
        )

        return []


def save_history(history):
    """
    Save topic history as a JSON array.
    """

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    history = history[-MAX_HISTORY:]


    HISTORY_FILE.write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


# ============================================================
# DEV.TO
# ============================================================

def get_trending_articles():
    """
    Fetch currently rising DEV.to articles.
    """

    print(
        "Fetching DEV.to rising articles..."
    )


    response = requests.get(
        DEVTO_ARTICLES,
        params={
            "state": "rising",
            "per_page": MAX_TRENDING_ARTICLES,
        },
        headers={
            "api-key": DEVTO_API_KEY,
        },
        timeout=30,
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
    Reduce DEV.to API responses to the fields useful for
    trend analysis.
    """

    topics = []


    for article in articles:

        topics.append(
            {
                "title": article.get(
                    "title"
                ),
                "description": article.get(
                    "description"
                ),
                "tags": article.get(
                    "tag_list",
                    [],
                ),
                "reactions": article.get(
                    "positive_reactions_count",
                    0,
                ),
                "comments": article.get(
                    "comments_count",
                    0,
                ),
                "published_at": article.get(
                    "published_at"
                ),
            }
        )


    return topics


# ============================================================
# ARTICLE PROMPT
# ============================================================

def build_article_prompt(
    topics,
    history,
):
    """
    Build the prompt for generating the actual article.

    IMPORTANT:
    The article is generated as plain Markdown.

    We intentionally DO NOT put the article inside JSON.
    This prevents large Markdown/code blocks from breaking
    JSON escaping or being truncated before the JSON closes.
    """

    topic_text = json.dumps(
        topics,
        indent=2,
        ensure_ascii=False,
    )


    history_text = json.dumps(
        history[
            -MAX_HISTORY_FOR_PROMPT:
        ],
        indent=2,
        ensure_ascii=False,
    )


    return f"""
You are an experienced Staff Software Engineer and technical
writer creating one original, publish-ready article for DEV.to.

Analyze the CURRENT DEV.to rising articles below to identify
an underlying engineering trend.

Then choose ONE distinct engineering angle and write an
original article that teaches the topic deeply.

============================================================
CURRENT DEV.TO RISING ARTICLES
============================================================

{topic_text}

============================================================
PREVIOUSLY PUBLISHED TOPICS
============================================================

Avoid repeating these topics unless there is clearly a new
and substantially different engineering angle.

{history_text}

============================================================
AREAS OF INTEREST
============================================================

Prioritize technically useful topics around:

- React
- JavaScript
- TypeScript
- Frontend Architecture
- Micro Frontends
- Module Federation
- Frontend Performance
- System Design
- Distributed Systems
- Event-driven Architecture
- Go
- AI
- LLMs
- AI Agents
- MCP
- RAG
- AI-assisted development
- Developer Productivity
- Software Architecture

Do not force one of these topics if the current DEV.to trend
clearly suggests a better engineering topic.

============================================================
ORIGINALITY
============================================================

Do NOT copy or closely rewrite any source article.

Do NOT reuse:

- source article titles
- source article sentences
- source article paragraph structures
- source article examples
- source article code
- source article conclusions

Instead:

1. Identify the underlying trend.
2. Choose a distinct engineering angle.
3. Add substantially different technical value.
4. Explain the subject deeply enough to teach an engineer.

============================================================
ARTICLE REQUIREMENTS
============================================================

Target approximately {TARGET_ARTICLE_MIN}-{TARGET_ARTICLE_MAX}
words.

The article should contain:

1. Strong H1 title
2. Compelling introduction
3. The engineering problem
4. Explanation of the core concept
5. How it works
6. Practical examples
7. Code examples where useful
8. Mermaid diagrams where useful
9. Real-world trade-offs
10. Common mistakes
11. When to use it
12. When NOT to use it
13. Practical recommendations
14. Strong conclusion

Prefer:

- concrete examples
- engineering reasoning
- architecture decisions
- implementation details
- trade-offs
- code
- diagrams
- practical recommendations

Avoid generic filler.

Write for experienced software engineers.

The article should sound like a thoughtful senior engineer
teaching another engineer.

Do NOT claim personal experience, metrics, or results that were
not provided.

Do NOT write things such as:

"As a Staff Engineer, I..."

unless that specific statement is supported by the supplied
information.

Do NOT mention:

- this prompt
- the automation
- content generation
- AI-generated content
- the source DEV.to articles

Do NOT use clickbait.

============================================================
MARKDOWN OUTPUT
============================================================

Return ONLY the final publish-ready Markdown article.

IMPORTANT:

- Start with the H1 title.
- Do NOT return JSON.
- Do NOT put the entire article inside a code fence.
- Code examples may use normal fenced code blocks.
- Mermaid diagrams may use Mermaid fenced blocks.
- Do NOT add commentary before the article.
- Do NOT add commentary after the article.
- Do NOT stop in the middle of a sentence.
- Do NOT stop in the middle of a code block.
- Do NOT stop in the middle of a Markdown table.
- End naturally with the conclusion.

Prioritize a complete article over additional detail.
"""


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_retry(
    prompt,
    response_schema=None,
    max_output_tokens=None,
):
    """
    Generate Gemini content with model fallback and retry.

    Temporary errors:
    - 429
    - 500
    - 502
    - 503
    - 504

    are retried before moving to the next model.
    """

    last_error = None


    for model in GEMINI_MODELS:

        for attempt in range(
            1,
            MAX_RETRIES_PER_MODEL + 1,
        ):

            try:

                print(
                    f"Calling Gemini: "
                    f"{model} "
                    f"(attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES_PER_MODEL})"
                )


                config_kwargs = {
                    "temperature": 0.8,
                }


                if max_output_tokens is not None:

                    config_kwargs[
                        "max_output_tokens"
                    ] = max_output_tokens


                if response_schema is not None:

                    config_kwargs[
                        "response_mime_type"
                    ] = "application/json"

                    config_kwargs[
                        "response_schema"
                    ] = response_schema


                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        **config_kwargs
                    ),
                )


                if not response:

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )


                print(
                    f"Gemini generation succeeded "
                    f"using {model}."
                )


                log_finish_reason(
                    response
                )


                return response


            except errors.ServerError as exc:

                last_error = exc


                code = getattr(
                    exc,
                    "code",
                    "unknown",
                )


                print(
                    f"Gemini returned server error "
                    f"{code}: {exc}"
                )


                if attempt < MAX_RETRIES_PER_MODEL:

                    delay = 10 * (
                        2 ** (attempt - 1)
                    )


                    print(
                        f"Retrying in "
                        f"{delay} seconds..."
                    )


                    time.sleep(delay)


            except errors.APIError as exc:

                last_error = exc


                code = getattr(
                    exc,
                    "code",
                    None,
                )


                print(
                    f"Gemini API error "
                    f"{code}: {exc}"
                )


                retryable = code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }


                if (
                    retryable
                    and attempt < MAX_RETRIES_PER_MODEL
                ):

                    delay = 10 * (
                        2 ** (attempt - 1)
                    )


                    print(
                        f"Retrying in "
                        f"{delay} seconds..."
                    )


                    time.sleep(delay)

                else:

                    break


            except Exception as exc:

                print(
                    "Unexpected Gemini error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                raise


        print(
            f"Model {model} failed after "
            f"{MAX_RETRIES_PER_MODEL} attempts."
        )


        if model != GEMINI_MODELS[-1]:

            print(
                "Trying fallback model..."
            )


    raise RuntimeError(
        "All Gemini models failed."
    ) from last_error


# ============================================================
# GEMINI RESPONSE DIAGNOSTICS
# ============================================================

def log_finish_reason(response):
    """
    Print generation finish reason when available.

    This is extremely useful for diagnosing output truncation.
    """

    try:

        candidates = getattr(
            response,
            "candidates",
            None,
        )


        if not candidates:

            print(
                "Gemini finish reason: "
                "unavailable"
            )

            return


        candidate = candidates[0]


        finish_reason = getattr(
            candidate,
            "finish_reason",
            None,
        )


        print(
            f"Gemini finish reason: "
            f"{finish_reason}"
        )


        usage = getattr(
            response,
            "usage_metadata",
            None,
        )


        if usage:

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                None,
            )


            output_tokens = getattr(
                usage,
                "candidates_token_count",
                None,
            )


            total_tokens = getattr(
                usage,
                "total_token_count",
                None,
            )


            print(
                "Gemini token usage: "
                f"prompt={prompt_tokens}, "
                f"output={output_tokens}, "
                f"total={total_tokens}"
            )


    except Exception as exc:

        print(
            f"Could not read Gemini diagnostics: "
            f"{exc}"
        )


# ============================================================
# RESPONSE TEXT
# ============================================================

def get_response_text(response):
    """
    Safely extract text from a Gemini response.
    """

    text = getattr(
        response,
        "text",
        None,
    )


    if not text:

        raise ValueError(
            "Gemini returned an empty text response."
        )


    return text.strip()


# ============================================================
# MARKDOWN CLEANUP
# ============================================================

def clean_markdown(text):
    """
    Remove an accidental outer Markdown fence.

    We only remove a fence if the entire response appears to
    have been wrapped in one.
    """

    text = (
        text or ""
    ).strip()


    if text.startswith(
        "```markdown"
    ):

        text = text[
            len("```markdown"):
        ].lstrip()


        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ].rstrip()


    elif text.startswith(
        "```md"
    ):

        text = text[
            len("```md"):
        ].lstrip()


        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ].rstrip()


    return text.strip()


# ============================================================
# ARTICLE COMPLETENESS CHECK
# ============================================================

def article_looks_truncated(markdown):
    """
    Detect common signs that Gemini stopped before completing
    the article.
    """

    if not markdown:

        return True


    if len(markdown) < MIN_ARTICLE_LENGTH:

        return True


    # Unclosed fenced code block.
    if markdown.count("```") % 2 != 0:

        return True


    # Unclosed tilde fence.
    if markdown.count("~~~") % 2 != 0:

        return True


    lines = markdown.splitlines()


    if not lines:

        return True


    last_line = lines[-1].strip()


    # Markdown table appears to have been cut off.
    if (
        last_line.startswith("|")
        and not last_line.endswith("|")
    ):

        return True


    # Strong indicators of an abruptly terminated generation.
    suspicious_endings = (
        ",",
        ":",
        ";",
        "(",
        "[",
        "{",
        "=",
        "->",
        "=>",
        "```",
        "|",
    )


    if last_line.endswith(
        suspicious_endings
    ):

        return True


    # A very short final sentence is suspicious if the article
    # otherwise looks substantial.
    words = last_line.split()


    if (
        len(markdown) > 3000
        and 0 < len(words) <= 2
    ):

        return True


    return False


# ============================================================
# ARTICLE GENERATION
# ============================================================

def generate_article(
    topics,
    history,
):
    """
    Generate a complete Markdown article.

    If the output appears truncated, generate it again rather
    than attempting to repair incomplete content.
    """

    base_prompt = build_article_prompt(
        topics,
        history,
    )


    print(
        "Generating article with Gemini..."
    )


    last_error = None


    for attempt in range(
        1,
        MAX_ARTICLE_GENERATION_ATTEMPTS + 1,
    ):

        try:

            print(
                f"Article generation attempt "
                f"{attempt}/"
                f"{MAX_ARTICLE_GENERATION_ATTEMPTS}"
            )


            prompt = base_prompt


            if attempt > 1:

                prompt += """

IMPORTANT RETRY INSTRUCTION:

The previous generation was incomplete.

Generate a NEW complete article.

Use fewer examples and less prose if necessary.

A complete conclusion is more important than additional
detail.

Do not stop until the article has a natural ending.
"""


            response = generate_with_retry(
                prompt,
                max_output_tokens=ARTICLE_MAX_OUTPUT_TOKENS,
            )


            markdown = clean_markdown(
                get_response_text(
                    response
                )
            )


            print(
                f"Generated article length: "
                f"{len(markdown)} characters"
            )


            if article_looks_truncated(
                markdown
            ):

                raise ValueError(
                    "Generated Markdown appears "
                    "truncated or incomplete."
                )


            if not re.search(
                r"^#\s+.+",
                markdown,
                re.MULTILINE,
            ):

                raise ValueError(
                    "Generated article is missing "
                    "an H1 title."
                )


            print(
                "Article completeness check passed."
            )


            return markdown


        except ValueError as exc:

            last_error = exc


            print(
                f"Article generation validation "
                f"failed: {exc}"
            )


            if attempt < MAX_ARTICLE_GENERATION_ATTEMPTS:

                print(
                    "Retrying article generation..."
                )


                time.sleep(2)


    raise RuntimeError(
        "Gemini failed to produce a complete "
        "article after "
        f"{MAX_ARTICLE_GENERATION_ATTEMPTS} attempts."
    ) from last_error



# ============================================================
# METADATA GENERATION
# ============================================================

def generate_metadata(article_markdown):
    """Generate DEV.to metadata locally without a second Gemini call."""
    print("Generating DEV.to metadata locally...")

    lines = article_markdown.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ") and line[2:].strip()), None)
    if not title:
        raise ValueError("Generated article is missing an H1 title.")

    paragraphs = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("|"):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    description = next((p for p in paragraphs if len(p) >= 80), paragraphs[0] if paragraphs else "A practical engineering guide for software developers.")
    description = re.sub(r"\s+", " ", description).strip()
    description = " ".join(re.split(r"(?<=[.!?])\s+", description)[:2]).strip()
    if len(description) > 300:
        description = description[:297].rsplit(" ", 1)[0] + "..."

    text = " " + article_markdown.lower() + " "
    tag_rules = [
        ("react", ["react", "reactjs"]),
        ("javascript", ["javascript"]),
        ("typescript", ["typescript"]),
        ("frontend", ["frontend", "front-end"]),
        ("systemdesign", ["system design"]),
        ("architecture", ["architecture", "architectural"]),
        ("microfrontend", ["micro frontend", "microfrontend"]),
        ("webpack", ["webpack", "module federation"]),
        ("performance", ["performance", "latency", "rendering"]),
        ("distributed", ["distributed system", "distributed systems"]),
        ("golang", ["golang"]),
        ("ai", ["artificial intelligence", " ai ", "ai-assisted"]),
        ("llm", ["llm", "large language model"]),
        ("agents", ["ai agent", "ai agents", "agentic"]),
        ("mcp", ["model context protocol", "mcp"]),
        ("rag", ["retrieval augmented generation"]),
        ("devtools", ["developer tooling", "developer productivity"]),
    ]
    tags = []
    for tag, keywords in tag_rules:
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
        if len(tags) >= 5:
            break
    if not tags:
        tags = ["programming", "software", "webdev"]

    print("Metadata generated locally: title={!r}, tags={}".format(title, ", ".join(tags)))
    return {"title": title, "description": description, "tags": tags[:5]}

# ============================================================
# ARTICLE VALIDATION
# ============================================================

def validate_article(
    article,
):
    """
    Validate everything immediately before publishing.
    """

    if not isinstance(
        article,
        dict,
    ):

        raise ValueError(
            "Generated article is not an object."
        )


    title = article.get(
        "title"
    )


    description = article.get(
        "description"
    )


    tags = article.get(
        "tags"
    )


    body = article.get(
        "body_markdown"
    )


    if not isinstance(
        title,
        str,
    ):

        raise ValueError(
            "Article title must be a string."
        )


    if not isinstance(
        description,
        str,
    ):

        raise ValueError(
            "Article description must be a string."
        )


    if not isinstance(
        body,
        str,
    ):

        raise ValueError(
            "Article body must be a string."
        )


    if not isinstance(
        tags,
        list,
    ):

        raise ValueError(
            "Article tags must be an array."
        )


    title = title.strip()
    description = description.strip()
    body = body.strip()


    # Use the article's actual H1 as the canonical title.
    h1_match = re.search(
        r"^#\s+(.+?)\s*$",
        body,
        re.MULTILINE,
    )


    if h1_match:

        title = h1_match.group(
            1
        ).strip()


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
            "Article body is too short "
            f"({len(body)} characters)."
        )


    # Validate tags.
    cleaned_tags = []


    for tag in tags[:5]:

        if not isinstance(
            tag,
            str,
        ):

            continue


        tag = tag.strip().lower()


        if not re.fullmatch(
            r"[a-z0-9]+",
            tag,
        ):

            continue


        if tag not in cleaned_tags:

            cleaned_tags.append(
                tag
            )


    if not cleaned_tags:

        raise ValueError(
            "Article has no valid DEV.to tags."
        )


    article["title"] = title
    article["description"] = description
    article["body_markdown"] = body
    article["tags"] = cleaned_tags[:5]


    print()
    print(
        "Article validation passed."
    )
    print()


    print(
        f"TITLE: {title}"
    )


    print(
        f"DESCRIPTION: {description}"
    )


    print(
        f"TAGS: {', '.join(article['tags'])}"
    )


    print(
        f"BODY LENGTH: {len(body)} characters"
    )


    print()


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate_title(
    title,
    history,
):
    """
    Prevent publishing an exact title already in history.
    """

    normalized_title = (
        title.strip().lower()
    )


    for item in history:

        previous_title = item.get(
            "title",
            "",
        )


        if (
            isinstance(
                previous_title,
                str,
            )
            and previous_title.strip().lower()
            == normalized_title
        ):

            return True


    return False


# ============================================================
# DEV.TO PUBLISH
# ============================================================

def publish_article(
    article,
):
    """
    Publish the final validated article to DEV.to.
    """

    payload = {
        "article": {
            "title": article[
                "title"
            ],
            "description": article[
                "description"
            ],
            "body_markdown": article[
                "body_markdown"
            ],
            "published": True,
            "tags": article[
                "tags"
            ][:5],
        }
    }


    print(
        "Publishing article to DEV.to..."
    )


    response = requests.post(
        DEVTO_ARTICLES,
        headers={
            "api-key": DEVTO_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )


    if response.status_code not in (
        200,
        201,
    ):

        print(
            f"DEV.to API error: "
            f"{response.status_code}"
        )


        print(
            response.text
        )


        raise RuntimeError(
            "DEV.to publish failed with "
            f"HTTP {response.status_code}"
        )


    result = response.json()


    print()
    print(
        "========================================"
    )


    print(
        "ARTICLE PUBLISHED"
    )


    print(
        "========================================"
    )


    print(
        f"Title: {result.get('title')}"
    )


    print(
        f"URL:   {result.get('url')}"
    )


    print(
        f"ID:    {result.get('id')}"
    )


    print(
        "========================================"
    )


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
    # Generate article as plain Markdown
    # --------------------------------------------------------

    article_markdown = generate_article(
        topics,
        history,
    )


    # --------------------------------------------------------
    # Generate small structured metadata
    # --------------------------------------------------------

    metadata = generate_metadata(
        article_markdown
    )


    # --------------------------------------------------------
    # Combine metadata + Markdown
    # --------------------------------------------------------

    article = {
        "title": metadata[
            "title"
        ],
        "description": metadata[
            "description"
        ],
        "tags": metadata[
            "tags"
        ],
        "body_markdown": article_markdown,
    }


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_article(
        article
    )


    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    if is_duplicate_title(
        article["title"],
        history,
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
            "title": article[
                "title"
            ],
            "tags": article[
                "tags"
            ],
            "url": result.get(
                "url"
            ),
            "devto_id": result.get(
                "id"
            ),
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
