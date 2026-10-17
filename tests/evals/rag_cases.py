EVAL_CASES = [
    {
        "question": "How many vacation days do employees receive?",
        "answerable": True,
        "expected_source": "sample.txt",
        "expected_chunk_index": 4,
    },
    {
        "question": "What do employees have to report to Security Operations?",
        "answerable": True,
        "expected_source": "sample.txt",
        "expected_chunk_index": 7,
    },
    {
        "question": "Can customers refund $2000 or more?",
        "answerable": True,
        "expected_source": "sample.txt",
        "expected_chunk_index": 10,
    },
    {
        "question": "Who won the 2022 FIFA World Cup?",
        "answerable": False,
    },
    {
        "question": "What currency do they use in Japan?",
        "answerable": False,
    },
    {
        "question": "What is the best burger in Bali?",
        "answerable": False,
    },
]