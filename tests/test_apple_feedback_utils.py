import unittest

from apple_feedback_utils import infer_topic, recommended_action, risk_level, split_messages


class AppleFeedbackUtilsTest(unittest.TestCase):
    def test_infer_topic_maps_product_experience_keywords(self):
        self.assertEqual(infer_topic("iPhone 电池掉电太快"), "续航")
        self.assertEqual(infer_topic("AirPods 在地铁里会断一下"), "连接")
        self.assertEqual(infer_topic("客服回复很及时"), "服务")
        self.assertEqual(infer_topic("外观设计不错"), "综合体验")

    def test_risk_level_escalates_negative_high_confidence_feedback(self):
        self.assertEqual(risk_level("Negative", 0.91, "待处理", 100), "红色预警")
        self.assertEqual(risk_level("Negative", 0.7, "观察中", 100), "重点关注")
        self.assertEqual(risk_level("Positive", 0.93, "已回复", 100), "常规")
        self.assertEqual(risk_level("Neutral", 0.68, "待处理", 100), "重点关注")

    def test_recommended_action_matches_sentiment_and_risk(self):
        self.assertEqual(recommended_action("Negative", "红色预警"), "转人工客服并同步产品团队")
        self.assertEqual(recommended_action("Negative", "重点关注"), "进入负面评论复核队列")
        self.assertEqual(recommended_action("Positive", "常规"), "沉淀为产品卖点语料")
        self.assertEqual(recommended_action("Neutral", "常规"), "继续观察")

    def test_split_messages_removes_blank_lines(self):
        raw_text = "第一条评论\n\n 第二条评论 \n"
        self.assertEqual(split_messages(raw_text), ["第一条评论", "第二条评论"])


if __name__ == "__main__":
    unittest.main()
