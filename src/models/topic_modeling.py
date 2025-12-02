"""
主题建模模块
使用LDA (Latent Dirichlet Allocation) 提取评论主题
"""

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import joblib
import logging

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    LDA主题建模器
    """

    def __init__(self, num_topics=10, num_words=10, passes=10, iterations=100):
        """
        Args:
            num_topics: 主题数量
            num_words: 每个主题显示的词数
            passes: 训练轮数
            iterations: 迭代次数
        """
        self.num_topics = num_topics
        self.num_words = num_words
        self.passes = passes
        self.iterations = iterations

        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def preprocess_for_lda(self, texts):
        """
        为LDA预处理文本

        Args:
            texts: 文本列表

        Returns:
            list: 分词后的文档列表
        """
        logger.info("Preprocessing texts for LDA...")

        processed_docs = []
        for text in texts:
            # 分词并转小写
            tokens = text.lower().split()

            # 移除停用词和短词
            tokens = [token for token in tokens
                      if token not in STOPWORDS and len(token) > 3]

            processed_docs.append(tokens)

        return processed_docs

    def train(self, texts):
        """
        训练LDA模型

        Args:
            texts: 文本列表

        Returns:
            dict: 训练结果
        """
        logger.info("Training LDA model...")

        # 预处理
        processed_docs = self.preprocess_for_lda(texts)

        # 创建词典
        logger.info("Creating dictionary...")
        self.dictionary = corpora.Dictionary(processed_docs)

        # 过滤极端词
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)

        logger.info(f"Dictionary size: {len(self.dictionary)}")

        # 创建语料库
        logger.info("Creating corpus...")
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

        # 训练LDA模型
        logger.info(f"Training LDA with {self.num_topics} topics...")
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=self.passes,
            iterations=self.iterations,
            per_word_topics=True
        )

        # 计算困惑度和一致性
        perplexity = self.lda_model.log_perplexity(self.corpus)
        logger.info(f"Perplexity: {perplexity:.4f}")

        # 计算主题一致性
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        logger.info(f"Coherence Score: {coherence_score:.4f}")

        # 打印主题
        self.print_topics()

        return {
            'perplexity': perplexity,
            'coherence_score': coherence_score,
            'num_topics': self.num_topics
        }

    def print_topics(self):
        """打印所有主题"""
        logger.info("\n" + "="*60)
        logger.info("发现的主题:")
        logger.info("="*60)

        for idx, topic in self.lda_model.print_topics(num_words=self.num_words):
            print(f"\nTopic {idx + 1}:")
            print(topic)

    def get_topic_words(self, topic_id, num_words=None):
        """
        获取主题的关键词

        Args:
            topic_id: 主题ID
            num_words: 返回的词数

        Returns:
            list: (词, 权重) 元组列表
        """
        if num_words is None:
            num_words = self.num_words

        return self.lda_model.show_topic(topic_id, topn=num_words)

    def get_document_topics(self, text):
        """
        获取文档的主题分布

        Args:
            text: 文本

        Returns:
            list: (主题ID, 概率) 元组列表
        """
        # 预处理
        tokens = text.lower().split()
        tokens = [token for token in tokens
                  if token not in STOPWORDS and len(token) > 3]

        # 转换为BOW
        bow = self.dictionary.doc2bow(tokens)

        # 获取主题分布
        topics = self.lda_model.get_document_topics(bow)

        return sorted(topics, key=lambda x: x[1], reverse=True)

    def predict_dominant_topic(self, text):
        """
        预测文档的主导主题

        Args:
            text: 文本

        Returns:
            int: 主导主题ID
        """
        topics = self.get_document_topics(text)
        if topics:
            return topics[0][0]
        return -1

    def analyze_corpus_topics(self, texts):
        """
        分析语料库的主题分布

        Args:
            texts: 文本列表

        Returns:
            pd.DataFrame: 主题分布数据框
        """
        logger.info("Analyzing topic distribution in corpus...")

        results = []
        for i, text in enumerate(texts):
            topics = self.get_document_topics(text)

            if topics:
                dominant_topic = topics[0][0]
                topic_prob = topics[0][1]

                # 获取主题关键词
                keywords = [word for word, _ in self.get_topic_words(dominant_topic, 5)]

                results.append({
                    'document_id': i,
                    'dominant_topic': dominant_topic,
                    'topic_probability': topic_prob,
                    'topic_keywords': ', '.join(keywords),
                    'text': text[:200]  # 截取前200字符
                })

        return pd.DataFrame(results)

    def visualize_topics(self, save_path=None):
        """
        可视化主题（使用pyLDAvis）

        Args:
            save_path: 保存路径（HTML文件）

        Returns:
            pyLDAvis对象
        """
        logger.info("Generating topic visualization...")

        vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

        if save_path:
            pyLDAvis.save_html(vis, str(save_path))
            logger.info(f"Visualization saved to {save_path}")

        return vis

    def create_topic_wordclouds(self, save_dir=None):
        """
        为每个主题创建词云

        Args:
            save_dir: 保存目录

        Returns:
            list: 词云图像列表
        """
        logger.info("Creating word clouds for topics...")

        wordclouds = []

        for topic_id in range(self.num_topics):
            # 获取主题词及权重
            topic_words = self.get_topic_words(topic_id, num_words=50)

            # 创建词频字典
            word_freq = {word: float(weight) for word, weight in topic_words}

            # 生成词云
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)

            # 绘图
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_id + 1}', fontsize=16)

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / f'topic_{topic_id + 1}_wordcloud.png',
                            bbox_inches='tight', dpi=150)
                logger.info(f"Saved wordcloud for topic {topic_id + 1}")

            wordclouds.append(wordcloud)
            plt.close()

        return wordclouds

    def get_topic_distribution(self):
        """
        获取语料库中主题的整体分布

        Returns:
            np.array: 主题分布数组
        """
        topic_counts = np.zeros(self.num_topics)

        for doc_bow in self.corpus:
            topics = self.lda_model.get_document_topics(doc_bow)
            for topic_id, prob in topics:
                topic_counts[topic_id] += prob

        return topic_counts / topic_counts.sum()

    def plot_topic_distribution(self, save_path=None):
        """
        绘制主题分布图

        Args:
            save_path: 保存路径
        """
        topic_dist = self.get_topic_distribution()

        plt.figure(figsize=(12, 6))
        plt.bar(range(1, self.num_topics + 1), topic_dist)
        plt.xlabel('Topic ID', fontsize=12)
        plt.ylabel('Proportion', fontsize=12)
        plt.title('Topic Distribution in Corpus', fontsize=14)
        plt.xticks(range(1, self.num_topics + 1))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved topic distribution plot to {save_path}")

        plt.show()

    def save(self, model_path, dictionary_path=None):
        """保存模型"""
        logger.info(f"Saving LDA model to {model_path}")
        self.lda_model.save(str(model_path))

        if dictionary_path:
            self.dictionary.save(str(dictionary_path))

    def load(self, model_path, dictionary_path=None):
        """加载模型"""
        logger.info(f"Loading LDA model from {model_path}")
        self.lda_model = LdaModel.load(str(model_path))

        if dictionary_path:
            self.dictionary = corpora.Dictionary.load(str(dictionary_path))


def find_optimal_num_topics(texts, min_topics=5, max_topics=20, step=5):
    """
    找到最优的主题数量

    Args:
        texts: 文本列表
        min_topics: 最小主题数
        max_topics: 最大主题数
        step: 步长

    Returns:
        dict: 不同主题数的一致性分数
    """
    logger.info("Finding optimal number of topics...")

    coherence_scores = {}

    for num_topics in range(min_topics, max_topics + 1, step):
        logger.info(f"Testing {num_topics} topics...")

        modeler = TopicModeler(num_topics=num_topics, passes=5)
        results = modeler.train(texts)

        coherence_scores[num_topics] = results['coherence_score']

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker='o')
    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel('Coherence Score', fontsize=12)
    plt.title('Coherence Score vs Number of Topics', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 找到最佳主题数
    optimal_num = max(coherence_scores, key=coherence_scores.get)
    logger.info(f"Optimal number of topics: {optimal_num}")

    return coherence_scores


def main():
    """示例：主题建模"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from config import PROCESSED_DATA_DIR, MODEL_DIR

    # 加载数据
    logger.info("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / 'reviews.csv')

    # 采样
    df = df.sample(n=min(5000, len(df)), random_state=42)
    texts = df['text'].tolist()

    # 训练主题模型
    print("\n" + "="*60)
    print("训练 LDA 主题模型")
    print("="*60)

    modeler = TopicModeler(num_topics=10, num_words=10, passes=10)
    results = modeler.train(texts)

    # 保存模型
    modeler.save(
        MODEL_DIR / 'lda_model.pkl',
        MODEL_DIR / 'lda_dictionary.pkl'
    )

    # 分析主题分布
    topic_df = modeler.analyze_corpus_topics(texts[:100])
    print("\n主题分布示例:")
    print(topic_df.head(10))

    # 创建词云
    modeler.create_topic_wordclouds(save_dir=MODEL_DIR / 'topic_wordclouds')

    # 绘制主题分布
    modeler.plot_topic_distribution(save_path=MODEL_DIR / 'topic_distribution.png')

    # 测试单个文档
    test_text = "The food was delicious and the service was excellent!"
    topics = modeler.get_document_topics(test_text)
    print(f"\n测试文本: {test_text}")
    print(f"主题分布: {topics}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
