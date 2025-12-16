"""Test queries for functional testing of the RAG system."""

TEST_QUERIES = [
    # Billing questions
    {
        "category": "Billing",
        "question": "What payment methods do you accept?",
        "expected_topics": ["credit card", "debit card", "net banking", "UPI", "mobile wallet"]
    },
    {
        "category": "Billing",
        "question": "When will I receive my monthly bill?",
        "expected_topics": ["billing cycle", "5th day", "email", "SMS"]
    },
    {
        "category": "Billing",
        "question": "How do I dispute a billing error?",
        "expected_topics": ["30 days", "customer care", "email", "mobile app"]
    },
    
    # Data and FUP questions
    {
        "category": "Data/FUP",
        "question": "What is Fair Usage Policy?",
        "expected_topics": ["FUP", "unlimited data", "speed reduced", "high-speed data"]
    },
    {
        "category": "Data/FUP",
        "question": "How do I check my data usage?",
        "expected_topics": ["mobile app", "SMS", "DATA", "54321"]
    },
    {
        "category": "Data/FUP",
        "question": "Can I purchase additional high-speed data after exhausting my FUP limit?",
        "expected_topics": ["data booster", "1 GB", "5 GB", "10 GB"]
    },
    
    # Roaming questions
    {
        "category": "Roaming",
        "question": "Do I need to pay extra for domestic roaming?",
        "expected_topics": ["FREE", "domestic roaming", "across India"]
    },
    {
        "category": "Roaming",
        "question": "What are the international roaming charges?",
        "expected_topics": ["roaming pack", "pay-per-use", "incoming calls", "data"]
    },
    {
        "category": "Roaming",
        "question": "How do I activate international roaming?",
        "expected_topics": ["24-48 hours", "mobile app", "website", "customer care", "security deposit"]
    },
    
    # Plan activation questions
    {
        "category": "Plan Activation",
        "question": "How long does it take to activate a new connection?",
        "expected_topics": ["prepaid", "2-4 hours", "postpaid", "24-48 hours"]
    },
    {
        "category": "Plan Activation",
        "question": "Can I change my plan anytime?",
        "expected_topics": ["mobile app", "website", "customer care", "immediate", "next billing cycle"]
    },
    {
        "category": "Plan Activation",
        "question": "How do I port my number to your network?",
        "expected_topics": ["PORT", "1900", "UPC", "3-5 working days"]
    },
    
    # General questions
    {
        "category": "General",
        "question": "What happens if I miss my payment due date?",
        "expected_topics": ["15 days", "grace period", "3 days", "suspended", "late payment"]
    },
    {
        "category": "General",
        "question": "How do I contact customer support?",
        "expected_topics": ["1800", "customer care", "email", "live chat", "mobile app"]
    },
    {
        "category": "General",
        "question": "What documents do I need for a new connection?",
        "expected_topics": ["photo ID", "Aadhaar", "PAN", "Passport", "proof of address"]
    }
]
