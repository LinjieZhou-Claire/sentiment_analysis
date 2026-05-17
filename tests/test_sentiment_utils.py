import unittest

from sentiment_utils import (
    SENTIMENT_LABELS,
    analyze_batch,
    fallback_analyze,
    generate_sample_reviews,
    normalize_label,
    summarize_results,
)


class SentimentUtilsTest(unittest.TestCase):
    def test_normalize_label_maps_common_model_outputs(self):
        self.assertEqual(normalize_label("positive"), "Positive")
        self.assertEqual(normalize_label("NEGATIVE"), "Negative")
        self.assertEqual(normalize_label("neutral"), "Neutral")
        self.assertEqual(normalize_label("LABEL_2"), "Positive")
        self.assertEqual(normalize_label("LABEL_1"), "Neutral")
        self.assertEqual(normalize_label("LABEL_0"), "Negative")

    def test_fallback_analyze_detects_chinese_sentiment_keywords(self):
        positive = fallback_analyze("物流很快，屏幕清晰，电池也很耐用")
        negative = fallback_analyze("手机半小时就没电了，画质太垃圾")
        neutral = fallback_analyze("今天收到手机，包装里有说明书和充电线")

        self.assertEqual(positive.label, "Positive")
        self.assertEqual(negative.label, "Negative")
        self.assertEqual(neutral.label, "Neutral")
        self.assertGreaterEqual(positive.score, 0)
        self.assertLessEqual(positive.score, 1)
        self.assertGreaterEqual(negative.score, 0)
        self.assertLessEqual(negative.score, 1)
        self.assertGreaterEqual(neutral.score, 0)
        self.assertLessEqual(neutral.score, 1)

    def test_generate_sample_reviews_returns_mixed_dataset(self):
        reviews = generate_sample_reviews()
        self.assertGreaterEqual(len(reviews), 10)
        self.assertLessEqual(len(reviews), 15)
        self.assertTrue(all("id" in item and "review" in item for item in reviews))
        self.assertTrue(any("好" in item["review"] or "满意" in item["review"] for item in reviews))
        self.assertTrue(any("差" in item["review"] or "失望" in item["review"] for item in reviews))

    def test_analyze_batch_and_summarize_results_use_all_labels(self):
        reviews = [
            {"id": 1, "review": "非常满意，拍照清晰"},
            {"id": 2, "review": "续航太差，半天就没电"},
            {"id": 3, "review": "包装完整，今天刚收到"},
        ]

        rows = analyze_batch(reviews, analyzer=fallback_analyze)
        summary = summarize_results(rows)

        self.assertEqual([row["sentiment"] for row in rows], ["Positive", "Negative", "Neutral"])
        self.assertEqual(set(summary), set(SENTIMENT_LABELS))
        self.assertEqual(summary["Positive"], 1)
        self.assertEqual(summary["Negative"], 1)
        self.assertEqual(summary["Neutral"], 1)


if __name__ == "__main__":
    unittest.main()
