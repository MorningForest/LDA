from collections import OrderedDict
import numpy as np
import random
import os
import configparser
import codecs
from stanfordcorenlp import StanfordCoreNLP
import log.logger as logger
import prettytable as pt

class Document:
    '''
    文檔，保存文檔單詞和個數
    '''
    def __init__(self):
        self.words = []
        self.length = 0
        self.docuName = None

class DataPreProcessing:
    '''
    总保存文档，得到单词集和映射
    '''
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()  #单词,id映射
        self.id2word = None #id, 单词映射

class LDA:
    def __init__(self, dpre, topic_num, alpha, beta, iters_time, top_words_num, log):
        self.dpre = dpre    #获取预处理参数
        self.K = topic_num  #聚类的个数
        self.alpha = alpha  #超参数α（alpha）
        self.beta = beta    #超参数 β(beta)
        self.iter_times = iters_time    #迭代次数
        self.top_words_num = top_words_num #每个类特征词个数top_words_num
        self.log = log

        self.p = np.zeros(self.K) #概率向量,存储采样的临时变量
        # 词在主题上的分布,单词-主题先验分布P(word,topic)
        self.wordTopic = np.zeros((self.dpre.words_count, self.K), dtype=int)
        # 每个topic上词的数量 主题条件概率分布P(word|topic)
        self.TopicWordSum = np.zeros(self.K, dtype=int)
        # 每个doc中各个topic词的个数 文档-主题先验分布P(doc, topic)
        self.docsTopic = np.zeros((self.dpre.docs_count, self.K), dtype=int)
        # 每个doc中词的总数 文档单词矩阵 单词条件概率分布P(topic|doc)
        self.docsWordSum = np.zeros(self.dpre.docs_count, dtype=int)
        # 文档中的词分布 文档-单词联合分布
        self.TopicWord = np.array([
            [0 for y in range(self.dpre.docs[x].length)]
            for x in range(self.dpre.docs_count)
        ])
        #随机分配类型，为每个文档中的各个单词分配主题
        for x in range(self.TopicWord.shape[0]):
            self.docsWordSum[x] = self.dpre.docs[x].length#每个doc中词个数
            for y in range(self.dpre.docs[x].length):
                topic = random.randint(0, self.K-1)#随机选取一个主题
                self.TopicWord[x][y] = topic #文档中词-主题分布
                self.wordTopic[self.dpre.docs[x].words[y]][topic] += 1
                self.TopicWordSum[topic] += 1
                self.docsTopic[x][topic] += 1
        #主题分布
        self.theta = np.array([
            [0.0 for y in range(self.K)]
            for _ in range(self.dpre.docs_count)
        ])
        #词语分布
        self.phi = np.array([
            [0.0 for y in range(self.dpre.words_count)]
            for _ in range(self.K)
        ])


    def _sampling(self, i, j):
        '''
        Parameters
        ----------
        i   文档下标
        j   词语下标
        Returns 新主题
        -------
        '''
        #换主题
        topic = self.TopicWord[i][j]
        word = self.dpre.docs[i].words[j]#单词编号
        self.wordTopic[word][topic] -= 1
        self.TopicWordSum[topic] -= 1
        self.docsWordSum[i] -= 1
        self.docsTopic[i][topic] -= 1

        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        self.p = (self.wordTopic[word]+self.beta)/(self.TopicWordSum[topic]+Vbeta) * \
                 (self.docsTopic[i]+self.alpha)/(self.docsWordSum[i]+Kalpha)
        #随机更新topic
        p = np.squeeze(np.asarray(self.p / np.sum(self.p)))
        topic = np.argmax(np.random.multinomial(1, p))

        self.wordTopic[word][topic] += 1
        self.TopicWordSum[topic] += 1
        self.docsWordSum[i] += 1
        self.docsTopic[i][topic] += 1
        return topic

    def _est(self):
        '''
        Returns 计算出主题-词语联合概率分布
        -------
        '''
        #Gibbs Sampling
        self.log.info("Gabbs Samlping ....")
        for x in range(self.iter_times):
            for i in range(self.dpre.docs_count):
                for j in range(self.dpre.docs[i].length):
                    topic = self._sampling(i, j)
                    self.TopicWord[i][j] = topic
        self.log.info("计算文档-主题分布")
        self._theta()
        self.log.info("计算词分布")
        self._phi()
        self.log.info("保存模型")
        self._save()

    def _theta(self):
        #计算文档-主题分布
        for i in range(self.dpre.docs_count):
            self.theta[i] = (self.docsTopic[i]+self.alpha)/(self.docsWordSum[i]+self.K*self.alpha)

    def _phi(self):
        #计算词语-主题分布
        for i in range(self.K):
            self.phi[i] = (self.wordTopic.T[i]+self.beta)/(self.TopicWordSum[i]+self.dpre.words_count*self.beta)

    def _save(self):
        '''
        Returns 保存模型参数
        -------
        '''
        conf = configparser.ConfigParser()
        conf.read("settings.conf")
        phifile = conf.get('Setting', 'phifile')
        thetafile = conf.get('Setting', 'thetafile')
        paramfile = conf.get('Setting', 'paramfile')
        topNfile = conf.get('Setting', 'topNfile')
        tassginfile = conf.get('Setting', 'tassginfile')

        self.log.info(u"文章-主题分布已保存到%s" % thetafile)
        with codecs.open(thetafile, 'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        # 保存phi词-主题分布
        self.log.info(u"词-主题分布已保存到%s" % phifile)
        with codecs.open(phifile, 'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        # 保存参数设置
        self.log.info(u"参数设置已保存到%s" % paramfile)
        with codecs.open(paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
        # 保存每个主题topic的词
        self.log.info(u"主题topN词已保存到%s" % topNfile)

        with codecs.open(topNfile, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in range(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
        # 保存最后退出时，文章的词分派的主题的结果
        self.log.info(u"文章-词-主题分派结果已保存到%s" % tassginfile)
        with codecs.open(tassginfile, 'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.TopicWord[x][y]) + '\t')
                f.write('\n')
        self.log.info(u"模型训练完成.")

    def perplexity(self, docs=None):
        '''
        Parameters  docs
        ----------
        Returns 计算困惑度
        -------
        '''
        if docs==None:
            docs = self.dpre.docs
        log_per = 0
        for m in range(self.dpre.docs_count):
            for word in self.dpre.docs[m].words:
                log_per -= np.log(np.asarray(self.theta[m]*self.phi.T[word]).sum())
        return log_per/self.dpre.docs_count

    def _showTopicWord(self):
        self.top_words_num = min(self.top_words_num, self.dpre.words_count)
        table = pt.PrettyTable(["第"+str(i)+"类主题" for i in range(self.K)])
        for x in range(self.K):
            topicwords = []
            twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
            twords.sort(key=lambda i: i[1], reverse=True)
            for y in range(self.top_words_num):
                word = self.dpre.id2word[[y][0]]
                topicwords.append(word)
            table.add_column("第"+str(x)+"类主题", topicwords)
        print(table)
        print(self.TopicWord)

    def _showDocsTopic(self, num):
        docxName = [self.dpre.docs[x].docuName for x in range(self.dpre.docs_count)]
        table = pt.PrettyTable(docxName)

        for x in range(self.dpre.docs_count):
            topics = [(n, self.theta[x][n]) for n in range(self.K)]
            topics.sort(key=lambda x:x[1], reverse=True)
            topicwords = []
            for y in range(num):
                topic, probility = topics[y]
                words = self.dpre.id2word[np.argmax(self.phi[topic])]
                topicwords.append(words+"*"+str(probility))
            table.add_column(self.dpre.docs[x].docuName, topicwords)
        print(table)

    def _get_topic_term(self, topicId):
        '''
        Parameters
        ----------
        topicId 主题编号
        Returns 主题词语,概率
        -------

        '''
        topics = [(n, self.phi[topicId][n]) for n in range(self.dpre.words_count)]
        topics.sort(lambda x:x[1], reverse=True)
        return topics[:100][0], topics[:100][1]

def preprocessing():
    #读取配置文件内容
    conf = configparser.ConfigParser()
    conf.read("settings.conf")
    filename = conf.get('Setting', 'filename')
    stanfordpath = conf.get('Setting', 'stanfordpath')
    stopwordpath = conf.get('Setting', 'stopwordpath')
    logpath = conf.get('Setting', 'logpath')
    #获取文件控制器
    log = logger.getlog(logpath)
    #获取去停止词
    stopword = [item.strip('\n').strip() for item in codecs.open(stopwordpath, 'r', 'utf-8').readlines()]
    nlp = StanfordCoreNLP(stanfordpath, lang='zh')
    item_idx = 0
    dpre = DataPreProcessing()
    log.info("读取文件夹内文件并处理中...")
    for path in os.listdir(filename):
        # ff = open(os.path.join("/home/beacon/software/自然语言处理/LDAModel/test"
        #                       ,path), 'w+', encoding="utf-8")
        if item_idx > 10000:
            break
        obspath = os.path.join(filename, path)  #获取文件路径
        # 生成一个文档对象：包含单词序列（w1,w2,w3,,,,,wn）可以重复的
        doc = Document()
        #读文件内容
        with codecs.open(obspath, 'r', 'utf-8') as f:
            docs = f.readlines()
        #一行行读取文件内容
        for item in docs:
            item = item.strip().strip('\n').strip()
            if len(item) == 0:
                continue
            words = nlp.pos_tag(item)#切词
            for (word, tag) in words:
                if tag in ['NN', 'NR']:
                    if word not in stopword and len(word) >= 2:#去停止词
                        # ff.write(word + " ")
                        if word in dpre.word2id.keys():
                            #已经在字典,加入到doc中
                            doc.words.append(dpre.word2id[word])
                        else:
                            #加入到字典和文档中
                            dpre.word2id[word] = item_idx
                            doc.words.append(item_idx)
                            item_idx += 1
            # ff.write("\n")
        doc.docuName = path
        doc.length = len(doc.words)
        dpre.docs.append(doc)
    dpre.docs_count = len(dpre.docs)
    dpre.words_count = len(dpre.word2id)
    dpre.id2word = OrderedDict({value:key for key, value in dpre.word2id.items()})
    log.info("处理文件夹内文件数据成功！")
    log.info("文件夹内文本数量为:{}, 单词数量为:{}".format(dpre.docs_count, dpre.words_count))
    print(dpre.word2id)
        
    return dpre, log

def main():
    dpre, log = preprocessing()
    lda = LDA(
        alpha=10,
        beta=0.01,
        topic_num=5,
        iters_time=50,
        top_words_num=10,
        log=log,
        dpre=dpre
    )
    lda._est()
    lda._showTopicWord()
    lda._showDocsTopic(3)
    print(lda.perplexity())

if __name__ == '__main__':
    main()
