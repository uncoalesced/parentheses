#!/usr/bin/env python3
"""Generate Nemotron 3.5 Super VL distillation specs for the SFT v4 capability build.

Writes JSONL spec files consumed by scripts/distill_nemotron_v2.py.

Kinds:
  article   - compact factual article + 6 QA pairs per topic
  factsheet - strict "Key: Value" fact lists (capitals, symbols, currencies, ...)
  reasoning - varied reasoning problems; content must end with "FINAL: <answer>"
  format    - instruction-following outputs (JSON, exact counts, lipograms, ...)

All prompts demand plain ASCII. No eval prompt strings are used as seeds; the
parser additionally drops generated questions that collide with the eval set.

Usage:
  .venv\\Scripts\\python.exe scripts\\gen_distill_specs.py [--out-dir out\\distill_v2\\specs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ASCII_SYSTEM = (
    "Write plain ASCII only: no unicode symbols, no markdown tables, no LaTeX. "
    "Never discuss safety, policies, or being an AI. Never refuse."
)
ARTICLE_SYSTEM = (
    "You are a knowledge distillation teacher for a compact language model. " + ASCII_SYSTEM +
    " Facts must be accurate and verifiable. Be direct and compact. "
    "Do not show reasoning. Output only the requested final format."
)
FACT_SYSTEM = (
    "You are a precise fact extraction teacher. " + ASCII_SYSTEM +
    " Output only the requested lines, in exactly the requested format. "
    "Do not add bullets, numbering, explanations, or extra text."
)
REASONING_SYSTEM = (
    "You are a rigorous reasoning teacher. " + ASCII_SYSTEM +
    " Think through the problem carefully. The final content must be exactly one line: "
    "'FINAL: <concise answer>'. Do not put anything else in the final content."
)
FORMAT_SYSTEM = (
    "You are an instruction-following teacher for a compact language model. " + ASCII_SYSTEM +
    " Follow the requested output constraints exactly. "
    "Output only the requested text. Do not explain or comment."
)

COUNTRIES = [
    "Australia", "France", "Japan", "Switzerland", "India", "United States", "United Kingdom",
    "Canada", "Germany", "Italy", "Spain", "Portugal", "Netherlands", "Belgium", "Sweden",
    "Norway", "Denmark", "Finland", "Poland", "Ukraine", "Russia", "China", "South Korea",
    "Indonesia", "Malaysia", "Singapore", "Thailand", "Vietnam", "Philippines", "Bangladesh",
    "Pakistan", "Sri Lanka", "Nepal", "Bhutan", "Myanmar", "Iran", "Iraq", "Saudi Arabia",
    "United Arab Emirates", "Qatar", "Israel", "Turkey", "Greece", "Egypt", "South Africa",
    "Nigeria", "Kenya", "Ethiopia", "Ghana", "Morocco", "Algeria", "Brazil", "Argentina",
    "Chile", "Peru", "Colombia", "Venezuela", "Mexico", "Cuba", "New Zealand", "Fiji",
    "Papua New Guinea", "Iceland", "Ireland", "Austria", "Hungary", "Czech Republic",
    "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Albania", "Moldova", "Belarus",
    "Lithuania", "Latvia", "Estonia", "Luxembourg", "Malta", "Cyprus", "Mongolia",
    "Kazakhstan", "Uzbekistan", "Azerbaijan", "Georgia", "Armenia", "Jordan", "Lebanon",
    "Yemen", "Oman", "Bahrain", "Cambodia", "Laos", "Brunei", "Tanzania", "Uganda",
    "Zimbabwe", "Zambia", "Mozambique", "Angola", "Cameroon", "Senegal", "Tunisia", "Libya",
    "Sudan", "Bolivia", "Ecuador", "Uruguay", "Paraguay", "Panama", "Costa Rica", "Guatemala",
    "Jamaica", "Haiti", "Dominican Republic", "Trinidad and Tobago", "Bahamas", "Afghanistan",
    "Syria", "Kuwait", "North Macedonia", "Montenegro", "Bosnia and Herzegovina", "Slovakia",
    "Eswatini", "Madagascar", "Mauritius", "Maldives",
]

ELEMENTS = [
    "Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Sodium", "Magnesium", "Aluminium",
    "Silicon", "Phosphorus", "Sulfur", "Chlorine", "Potassium", "Calcium", "Iron", "Copper",
    "Zinc", "Silver", "Gold", "Mercury", "Lead", "Tin", "Nickel", "Chromium", "Manganese",
    "Cobalt", "Titanium", "Platinum", "Uranium", "Radium", "Neon", "Argon", "Krypton",
    "Xenon", "Iodine", "Bromine", "Fluorine", "Lithium", "Beryllium", "Boron",
]

COMPOUNDS = [
    "water", "table salt", "carbon dioxide", "carbon monoxide", "ammonia", "methane",
    "sulfuric acid", "hydrochloric acid", "nitric acid", "sodium bicarbonate",
    "hydrogen peroxide", "ozone", "glucose", "ethanol", "acetic acid", "calcium carbonate",
    "rust", "bleach", "dry ice", "laughing gas",
]

PEOPLE = [
    "William Shakespeare", "Leonardo da Vinci", "Michelangelo", "Albert Einstein",
    "Isaac Newton", "Charles Darwin", "Marie Curie", "Nikola Tesla", "Thomas Edison",
    "Galileo Galilei", "Nicolaus Copernicus", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven",
    "Pablo Picasso", "Vincent van Gogh", "Mahatma Gandhi", "Nelson Mandela", "Abraham Lincoln",
    "George Washington", "Napoleon Bonaparte", "Winston Churchill", "John F. Kennedy",
    "Elon Musk", "Jensen Huang", "Satya Nadella", "Sundar Pichai", "Tim Cook", "Bill Gates",
    "Steve Jobs", "Linus Torvalds", "James Gosling", "Guido van Rossum", "Dennis Ritchie",
    "Bjarne Stroustrup", "Brendan Eich", "Alan Turing", "Ada Lovelace", "Grace Hopper",
    "Tim Berners-Lee", "Droupadi Murmu",
]

ARTICLE_TOPICS = (
    [(c, f"the country {c}") for c in [
        "Australia", "France", "Japan", "Switzerland", "India", "United States", "United Kingdom",
        "Canada", "Germany", "Italy", "Spain", "China", "Russia", "Brazil", "Egypt",
        "South Africa", "Kenya", "Mexico", "Argentina", "South Korea", "Indonesia",
        "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Singapore", "Malaysia", "Thailand",
        "Vietnam", "Philippines", "Saudi Arabia", "United Arab Emirates", "Israel", "Turkey",
        "Greece", "Poland", "Ukraine", "Ireland", "Iceland", "New Zealand",
    ]]
    + [(p, f"the planet {p}") for p in [
        "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    ]]
    + [(x, f"the celestial object or mission {x}") for x in [
        "the Sun", "the Moon", "the Milky Way", "the Apollo 11 mission",
        "the International Space Station", "black holes",
    ]]
    + [(e, f"the chemical element {e}") for e in [
        "Gold", "Silver", "Iron", "Copper", "Oxygen", "Carbon", "Hydrogen", "Sodium",
        "Potassium", "Calcium",
    ]]
    + [(p, f"the person {p}") for p in [
        "William Shakespeare", "Leonardo da Vinci", "Albert Einstein", "Isaac Newton",
        "Marie Curie", "Jensen Huang", "Satya Nadella", "Droupadi Murmu", "Linus Torvalds",
        "James Gosling", "Mahatma Gandhi", "Nelson Mandela",
    ]]
    + [(t, f"the technology or company {t}") for t in [
        "NVIDIA", "Microsoft", "Linux", "the Java programming language",
        "the HTTP protocol", "the Python programming language", "Apple Inc.",
        "Google", "the C programming language", "JSON",
    ]]
    + [(g, f"the place or natural feature {g}") for g in [
        "Mangalore", "Karnataka", "Kerala", "Tamil Nadu", "the Pacific Ocean",
        "the Indian Ocean", "the Himalayas", "the Sahara desert", "the Nile river",
        "Canberra",
    ]]
    + [(d, f"the Dravidian language or culture topic {d}") for d in [
        "the Tulu language", "the Kannada language", "the Tamil language",
        "the Telugu language", "the Malayalam language", "the Koya language",
        "the Kuvi language", "Sangam literature", "the Tirukkural", "Yakshagana",
    ]]
    + [(s, f"the scientific concept {s}") for s in [
        "photosynthesis", "gravity", "electricity", "DNA", "the water cycle",
        "the CPU", "the GPU", "the atom", "the solar system", "the food chain",
    ]]
)

HTTP_CODES = [
    "200", "201", "204", "301", "302", "304", "400", "401", "403", "404", "405", "408",
    "409", "410", "418", "429", "500", "502", "503", "504",
]

ANTONYMS = [
    "scarce", "hot", "cold", "big", "small", "fast", "slow", "happy", "sad", "hard",
    "soft", "light", "dark", "early", "late", "young", "old", "rich", "poor", "strong",
    "weak", "clean", "dirty", "empty", "full", "open", "closed", "begin", "end", "buy",
    "sell", "win", "lose", "remember", "forget", "accept", "reject", "ancient", "modern",
    "simple", "complex", "brave", "afraid", "generous", "stingy", "quiet", "loud", "smooth", "rough",
]

SHAPES = ["triangle", "square", "rectangle", "pentagon", "hexagon", "heptagon", "octagon",
          "nonagon", "decagon", "circle", "rhombus", "trapezoid"]

DRAVIDIAN_FACTS = [
    ("Mangalore", "a major port city in Karnataka, India"),
    ("Tulu", "a Dravidian language spoken mainly in coastal Karnataka"),
    ("Kannada", "a Dravidian language and the official language of Karnataka"),
    ("Tamil", "a Dravidian language and the official language of Tamil Nadu"),
    ("Telugu", "a Dravidian language and the official language of Andhra Pradesh and Telangana"),
    ("Malayalam", "a Dravidian language and the official language of Kerala"),
    ("Koya", "a Dravidian language spoken by tribal communities in central India"),
    ("Kuvi", "a Dravidian language spoken in Odisha and Andhra Pradesh"),
    ("Kolami", "a Dravidian language spoken in Maharashtra and Telangana"),
    ("Gondi", "a Dravidian language spoken by the Gond people of central India"),
    ("Kerala", "a state on the south-western coast of India"),
    ("Karnataka", "a state in south-western India whose capital is Bengaluru"),
    ("Tamil Nadu", "a state in southern India whose capital is Chennai"),
    ("Hampi", "a UNESCO World Heritage site in Karnataka, capital of the Vijayanagara Empire"),
    ("Mysore Palace", "a royal palace in Mysuru, Karnataka"),
    ("Onam", "a major harvest festival celebrated in Kerala"),
    ("Pongal", "a Tamil harvest festival celebrated in January"),
    ("Ugadi", "the Kannada and Telugu New Year festival"),
    ("Yakshagana", "a traditional theatre form from coastal Karnataka"),
    ("Kathakali", "a classical dance-drama from Kerala"),
]


def _spec(id_: str, kind: str, system: str, prompt: str, max_tokens: int = 1800,
          temperature: float = 0.2, meta: dict | None = None) -> dict:
    d = {"id": id_, "kind": kind, "system": system, "prompt": prompt,
         "max_tokens": max_tokens, "temperature": temperature}
    if meta:
        d["meta"] = meta
    return d


def article_specs() -> list[dict]:
    out = []
    for i, (name, desc) in enumerate(ARTICLE_TOPICS, 1):
        prompt = (
            f"Write a compact factual article of 80 to 120 words about {desc}. "
            "Include key names, dates, places, and numbers. "
            "Then write exactly 6 question-answer pairs based on the article; "
            "at least two answers must be a single word. "
            "Use exactly this format and nothing else:\n"
            "ARTICLE: <article>\n"
            "QA: <question> || <answer>\n"
            "QA: <question> || <answer>\n"
            "QA: <question> || <answer>\n"
            "QA: <question> || <answer>\n"
            "QA: <question> || <answer>\n"
            "QA: <question> || <answer>"
        )
        out.append(_spec(f"article-{i:03d}", "article", ARTICLE_SYSTEM, prompt,
                         max_tokens=2400, meta={"topic": name}))
    return out


def fact_specs() -> list[dict]:
    out = []

    def batches(prefix, items, size, template, kind):
        for b in range(0, len(items), size):
            group = items[b:b + size]
            out.append(_spec(
                f"{prefix}-{b // size + 1:02d}", "factsheet", FACT_SYSTEM,
                template + "\n" + ", ".join(group),
                max_tokens=1500, temperature=0.1,
                meta={"type": kind, "items": group},
            ))

    batches("facts-capitals", COUNTRIES, 20,
            "For each country below, output exactly one line: Country: Capital city. "
            "Do not add bullets, numbering, explanations, or extra text.", "capitals")
    batches("facts-elements", ELEMENTS, 20,
            "For each chemical element below, output exactly one line: Element: Symbol. "
            "Do not add bullets, numbering, explanations, or extra text.", "elements")
    batches("facts-compounds", COMPOUNDS, 20,
            "For each compound below, output exactly one line: Compound: Chemical formula using plain ASCII. "
            "Do not add bullets, numbering, explanations, or extra text.", "compounds")
    batches("facts-currencies", COUNTRIES[:45], 15,
            "For each country below, output exactly one line: Country: Currency. "
            "Do not add bullets, numbering, explanations, or extra text.", "currencies")
    batches("facts-people", PEOPLE[:40], 10,
            "For each person below, output exactly one line: Person: one factual sentence about them. "
            "Use plain ASCII. Do not add bullets, numbering, explanations, or extra text.", "people")
    batches("facts-antonyms", ANTONYMS, 25,
            "For each word below, output exactly one line: word: its most common antonym. "
            "Use lowercase. Do not add bullets, numbering, explanations, or extra text.", "antonyms")
    batches("facts-http", HTTP_CODES, 20,
            "For each HTTP status code below, output exactly one line: code: official reason phrase. "
            "Do not add bullets, numbering, explanations, or extra text.", "http")
    batches("facts-shapes", SHAPES, 12,
            "For each shape below, output exactly one line: shape: number of sides as a digit. "
            "For a circle write 0. Do not add bullets, numbering, explanations, or extra text.", "shapes")

    drav = [f"{k}: {v}" for k, v in DRAVIDIAN_FACTS]
    out.append(_spec(
        "facts-dravidian-01", "factsheet", FACT_SYSTEM,
        "Rewrite each item below as exactly one line in the format: Subject: one factual sentence. "
        "Do not add bullets, numbering, explanations, or extra text.\n" + "\n".join(drav),
        max_tokens=1500, temperature=0.1, meta={"type": "dravidian"},
    ))

    geo = [
        "Pacific Ocean", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Southern Ocean",
        "Asia", "Africa", "Europe", "North America", "South America", "Antarctica", "Australia",
    ]
    out.append(_spec(
        "facts-geo-01", "factsheet", FACT_SYSTEM,
        "For each place below, output exactly one line: Place: one factual sentence about it "
        "(location, size rank, or a defining feature). Use plain ASCII. "
        "Do not add bullets, numbering, explanations, or extra text.\n" + ", ".join(geo),
        max_tokens=1500, temperature=0.1, meta={"type": "geo"},
    ))
    return out


def reasoning_specs() -> list[dict]:
    prompts: list[str] = []

    for i in range(1, 21):
        a = 2 + (i % 7)
        b = 3 + (i * 5) % 19
        c = a * (2 + i % 9) + b
        prompts.append(f"Solve for x and give the value only: {a}x + {b} = {c}.")

    seqs = []
    for i in range(1, 21):
        start = 2 + i
        step = 2 + (i % 5)
        nums = [start + step * k for k in range(5)]
        if i % 2 == 0:
            nums = [start, start + 2, start + 5, start + 9, start + 14]
        seqs.append(nums)
    for nums in seqs:
        joined = ", ".join(str(n) for n in nums)
        prompts.append(
            f"Find the next number in this sequence and state the rule: {joined}. "
            "End with the answer as a number."
        )

    for i in range(1, 16):
        hours = 1 + i % 7
        prompts.append(
            f"How many minutes are in {hours}.5 hours? Give the number only."
        )
    for i in range(1, 11):
        h = (i * 2) % 12 or 12
        m = (i * 5) % 60
        prompts.append(
            f"A clock shows {h}:{m:02d}. What is the smaller angle between the hour and "
            "minute hands, in degrees? Give the number only."
        )
    pairs = [(0.9, 0.89), (0.75, 0.8), (1.2, 1.19), (3.4, 3.399), (0.5, 0.55),
             (2.25, 2.3), (7.1, 7.09), (10.02, 10.2), (0.111, 0.12), (5.0, 5.05)]
    for a, b in pairs:
        prompts.append(f"Which is larger: {a} or {b}? Answer briefly.")

    syllogisms = [
        ("zeps", "morks", "tivs"), ("bloops", "razzies", "lazzies"), ("florps", "glints", "naxes"),
        ("quibs", "vants", "drells"), ("plinks", "stobs", "wurms"), ("graves", "plints", "zobs"),
        ("trens", "skells", "vurns"), ("mibs", "polems", "daxes"), ("shrons", "flomes", "quist"),
        ("brells", "clarns", "sloves"), ("prins", "troves", "mules"), ("glims", "snarks", "bokers"),
        ("fens", "lorbs", "tweens"), ("kilps", "drobes", "sprats"), ("nurds", "velts", "clines"),
    ]
    for a, b, c in syllogisms:
        prompts.append(f"All {a} are {b}. All {b} are {c}. Are all {a} {c}? Answer yes or no and explain briefly.")

    for i in range(1, 16):
        total = 8 + i
        remain = 2 + (i % 6)
        prompts.append(
            f"A basket has {total} fruits. All but {remain} are eaten. How many remain? "
            "Give the number only."
        )
    for i in range(1, 11):
        rel = "before" if i % 2 else "after"
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i % 7]
        prompts.append(
            f"If yesterday was {i % 5 + 1} days {rel} {day}, what day is today? "
            "Answer with the day name only."
        )
    for i in range(1, 11):
        prompts.append(
            f"In a race, you overtake the runner in position {i % 5 + 1}. "
            "What position are you in now? Answer with the number only."
        )
    for i in range(1, 11):
        prompts.append(
            "Which is heavier: 1 kilogram of steel or 1 kilogram of feathers? "
            "Answer in one short sentence."
        )
    word_problems = [
        ("A tank is filled by pipe A in 6 hours and pipe B in 8 hours. "
         "If both are opened together, how long will they take to fill the tank? Give the answer in hours."),
        ("A train travels 240 km in 3 hours. What is its average speed in km per hour? Give the number only."),
        ("A shop sells a shirt for 800 rupees after a 20 percent discount. What was the original price? Give the number only."),
        ("A sum doubles in 8 years at simple interest. What is the annual interest rate in percent? Give the number only."),
        ("A father is 4 times as old as his son. In 20 years he will be twice as old. "
         "How old is the son now? Give the number only."),
        ("A mixture contains milk and water in the ratio 3 to 2. If there are 30 liters of milk, "
         "how many liters of water are there? Give the number only."),
        ("A worker completes a job in 12 days. Another worker completes it in 18 days. "
         "Working together, how many days will they take? Give the answer as a decimal."),
        ("A car travels at 60 km per hour for 2.5 hours. How many km does it cover? Give the number only."),
        ("A rectangle is 12 cm long and 5 cm wide. What is its area in square cm? Give the number only."),
        ("A circle has a radius of 7 cm. What is its circumference in cm, using pi = 22/7? Give the number only."),
        ("A bag has 5 red and 3 blue balls. What is the probability of drawing a red ball? Give a fraction."),
        ("A number is increased by 25 percent and becomes 100. What was the original number? Give the number only."),
        ("A sum of 5000 rupees earns 8 percent simple interest per year. How much interest in 2 years? Give the number only."),
        ("A boat goes 30 km downstream in 2 hours and returns in 3 hours. What is the speed of the stream? Give the number only."),
        ("A cube has an edge of 3 cm. What is its volume in cubic cm? Give the number only."),
        ("A train 120 m long passes a pole in 6 seconds. What is its speed in m per second? Give the number only."),
        ("A person spends 40 percent of his salary and saves 3000 rupees. What is his salary? Give the number only."),
        ("A team wins 60 percent of 25 matches. How many matches did it win? Give the number only."),
        ("A container holds 2.5 liters. How many 250 ml glasses can it fill? Give the number only."),
        ("A sequence starts at 5 and each term is 3 more than the previous term. What is the 10th term? Give the number only."),
    ]
    prompts.extend(word_problems)

    readings = [
        ("Ravi left home at 7:30 and reached school at 8:10. How many minutes did the journey take? Give the number only.", 40),
        ("Meena bought 3 notebooks at 45 rupees each and paid with a 200 rupee note. How much change did she get? Give the number only.", 65),
        ("A library has 4 shelves with 25 books each. 17 books are borrowed. How many remain? Give the number only.", 83),
        ("A farmer plants 12 rows with 9 saplings each. 7 saplings die. How many survive? Give the number only.", 101),
        ("A bus starts with 32 passengers. At the first stop 9 get off and 5 board. How many are on the bus? Give the number only.", 28),
        ("Anita reads 15 pages on Monday and 1.5 times as many on Tuesday. How many pages on Tuesday? Give the number only.", 22),
        ("A shop had 90 apples. It sold 35 in the morning and 28 in the evening. How many are left? Give the number only.", 27),
        ("A clock gains 2 minutes every hour. How many minutes does it gain in 12 hours? Give the number only.", 24),
        ("A water tank holds 500 liters and leaks 5 liters per hour. How much remains after 6 hours? Give the number only.", 470),
        ("A student scores 78, 85, and 92 in three tests. What is the average score? Give the number only.", 85),
        ("A rectangle has a perimeter of 30 cm and a length of 9 cm. What is its width in cm? Give the number only.", 6),
        ("A pizza is cut into 8 equal slices. 3 slices are eaten. What fraction remains? Give a fraction.", "5/8"),
        ("A man walks 3 km north and then 4 km east. How far is he from the start, in km? Give the number only.", 5),
        ("A printer prints 18 pages per minute. How long to print 90 pages, in minutes? Give the number only.", 5),
        ("A school has 480 students, 45 percent of them girls. How many girls? Give the number only.", 216),
    ]
    for text, _ in readings:
        prompts.append(text)

    science = [
        "Why does ice float on water? Explain in two short sentences.",
        "What happens to the boiling point of water at higher altitude? Explain briefly.",
        "Why is the sky blue? Explain briefly.",
        "What is the difference between mass and weight? Explain in one sentence.",
        "Why do metals conduct electricity? Explain briefly.",
        "What is the difference between evaporation and boiling? Explain briefly.",
        "Why does a mirror reflect light? Explain briefly.",
        "What is the role of chlorophyll in photosynthesis? Explain briefly.",
        "Why does the Moon look bigger near the horizon? Explain briefly.",
        "What is Ohm's law? State it in one sentence.",
    ]
    prompts.extend(science)

    out = []
    for i, p in enumerate(prompts, 1):
        out.append(_spec(f"reason-{i:03d}", "reasoning", REASONING_SYSTEM, p,
                         max_tokens=2600, temperature=0.2))
    return out


def format_specs() -> list[dict]:
    prompts: list[str] = []
    json_jobs = [
        ('name, age, and is_student', '"Alex", 20, and true'),
        ('city, country, and population', '"Tokyo", "Japan", and 14000000'),
        ('title, author, and year', '"The Hobbit", "J.R.R. Tolkien", and 1937'),
        ('product, price, and in_stock', '"laptop", 799.99, and false'),
        ('id, name, and tags', '7, "widget", and ["small", "blue"]'),
        ('name, languages, and active', '"Mira", ["English", "Kannada"], and true'),
        ('item, quantity, and unit', '"rice", 5, "kg"'),
        ('first_name, last_name, and email', '"Ravi", "Kumar", and "ravi@example.com"'),
        ('project, stars, and license', '"parentheses", 120, and "research-only"'),
        ('name, age, and hobbies', '"Sara", 31, and ["reading", "cycling"]'),
        ('country, capital, and currency', '"Japan", "Tokyo", and "yen"'),
        ('brand, model, and year', '"Toyota", "Corolla", and 2020'),
        ('user, score, and passed', '"neo", 88, and true'),
        ('title, duration_minutes, and rating', '"Inception", 148, and 8.8'),
        ('name, role, and team', '"Asha", "engineer", and "platform"'),
    ]
    for i, (keys, vals) in enumerate(json_jobs, 1):
        prompts.append(
            f"Return valid JSON containing the keys {keys}. Use {vals} as their respective values. "
            "Return only the JSON object, nothing else."
        )
    capitals = ["France", "Japan", "Australia", "India", "Egypt", "Canada", "Brazil", "Kenya"]
    for c in capitals:
        prompts.append(f"Answer this question using exactly three words: What is the capital of {c}?")
    prompts.extend([
        "Answer the next question in lowercase only: WHAT IS THE LARGEST PLANET?",
        "Answer the next question in lowercase only: WHAT IS THE CHEMICAL SYMBOL FOR GOLD?",
        "Answer the next question in lowercase only: WHICH PLANET IS CLOSEST TO THE SUN?",
        "Answer the next question in lowercase only: WHO WROTE ROMEO AND JULIET?",
        "Answer the next question in lowercase only: WHAT IS THE CAPITAL OF CANADA?",
    ])
    explain = [
        "Explain the difference between a CPU and a GPU in fewer than 30 words.",
        "Explain what the internet is in fewer than 25 words.",
        "Explain why sleep matters in fewer than 20 words.",
        "Explain what a database is in fewer than 25 words.",
        "Explain the difference between weather and climate in fewer than 30 words.",
        "Explain what an API is in fewer than 25 words.",
        "Explain why exercise is useful in fewer than 20 words.",
        "Explain what machine learning is in fewer than 30 words.",
        "Explain what a compiler does in fewer than 25 words.",
        "Explain the water cycle in fewer than 30 words.",
        "Explain what inflation means in fewer than 25 words.",
        "Explain why trees are important in fewer than 20 words.",
        "Explain what a GPU is used for in fewer than 25 words.",
        "Explain the difference between RAM and storage in fewer than 30 words.",
        "Explain what open source software is in fewer than 25 words.",
    ]
    prompts.extend(explain)
    lipo_topics = [
        "a garden in spring", "a rainy morning", "a train journey", "a quiet library",
        "a mountain hike", "a city at night", "a cup of tea", "a football match",
        "a visit to the beach", "a winter morning", "a busy market", "a bird in flight",
    ]
    for t in lipo_topics:
        prompts.append(
            f"Write a paragraph of 60 to 90 words about {t} without using the letter e. "
            "Do not use any word containing the letter e. Output only the paragraph."
        )
    prompts.extend([
        "List four primary colors as a JSON array only.",
        "List three oceans as a comma-separated list only.",
        "Write a CSV with header name,score and two rows: Asha,91 and Ravi,84. Output only the CSV.",
        "Output exactly the word OK and nothing else.",
        "Output exactly three bullet points about saving water, each starting with a hyphen.",
        "Output a valid JSON array of the numbers 1, 2, and 3, nothing else.",
        "Write the word 'parentheses' five times separated by commas, nothing else.",
        "Output the days Monday to Friday as a JSON array of strings, nothing else.",
        "Format the number 1234567 with commas and output only the result.",
        "Output a two-column CSV with header country,capital and rows France,Paris and Japan,Tokyo. Nothing else.",
    ])
    out = []
    for i, p in enumerate(prompts, 1):
        out.append(_spec(f"format-{i:03d}", "format", FORMAT_SYSTEM, p,
                         max_tokens=1200, temperature=0.2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path("out") / "distill_v2" / "specs"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = {
        "articles": article_specs(),
        "factsheets": fact_specs(),
        "reasoning": reasoning_specs(),
        "format": format_specs(),
    }
    total = 0
    for name, specs in groups.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for s in specs:
                f.write(json.dumps(s, ensure_ascii=True) + "\n")
        print(f"[specs] {name}: {len(specs)} -> {path}")
        total += len(specs)
    print(f"[specs] total {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
