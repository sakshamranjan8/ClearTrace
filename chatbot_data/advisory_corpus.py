"""
Small hand-written advisory/regulation corpus for the chatbot to ground on
(Section 7C). In the full version this would be a FAISS/LangChain vector
store over real CPCB/GRAP documents; for the MVP a keyword-scored lookup
over these short docs is enough to make answers grounded and demoable.
"""

DOCS = [
    {
        "id": "grap_stage3",
        "title": "GRAP Stage III (Severe) actions",
        "text": (
            "Under Graded Response Action Plan Stage III, authorities restrict non-essential "
            "construction and demolition, restrict BS-III petrol and BS-IV diesel four-wheelers, "
            "and advise schools to consider suspending outdoor activities such as sports and assembly."
        ),
    },
    {
        "id": "grap_stage2",
        "title": "GRAP Stage II (Very Poor) actions",
        "text": (
            "Under Stage II, mechanized road sweeping and water sprinkling are intensified, "
            "diesel generator use is restricted where power is available, and parking fees are "
            "increased to discourage private vehicle use."
        ),
    },
    {
        "id": "school_advisory",
        "title": "School guidance during severe AQI",
        "text": (
            "When AQI is severe, schools are advised to move assemblies and physical education "
            "indoors, keep windows closed, and avoid outdoor sports practice until air quality improves."
        ),
    },
    {
        "id": "exercise_advisory",
        "title": "Outdoor exercise guidance",
        "text": (
            "Jogging or other strenuous outdoor exercise is not recommended when AQI is above 200 (Poor) "
            "or higher, especially in early morning hours when pollutant concentration near roads is elevated. "
            "Prefer indoor exercise or early evening hours if AQI is moderate."
        ),
    },
    {
        "id": "vulnerable_groups",
        "title": "Guidance for vulnerable groups",
        "text": (
            "Children, elderly people, pregnant women, and people with asthma or respiratory conditions "
            "should limit outdoor exposure earlier, at Moderate AQI levels, rather than waiting until Poor "
            "or Very Poor levels."
        ),
    },
    {
        "id": "mask_guidance",
        "title": "Mask guidance",
        "text": (
            "An N95 or KN95 mask with a good facial seal reduces inhaled particulate matter significantly "
            "more than a surgical or cloth mask during high-AQI days."
        ),
    },
    {
        "id": "construction_dust",
        "title": "Construction and dust sources",
        "text": (
            "Unmitigated construction and demolition sites are a major localized source of PM10 and PM2.5 "
            "dust. Sites are required to use dust screens, water sprinkling, and covered material transport."
        ),
    },
]


def keyword_search(query, top_k=3):
    q_tokens = set(query.lower().split())
    scored = []
    for doc in DOCS:
        text_tokens = set((doc["title"] + " " + doc["text"]).lower().split())
        overlap = len(q_tokens & text_tokens)
        if overlap:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]] or DOCS[:top_k]
