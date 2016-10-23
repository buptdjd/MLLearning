
import math
import pickle


class CCDomainDetector:

    def __init__(self):
        self.domain_char_lists = "abcdefghijklmnopqrstuvwxyz.-0123456789"
        self.pos = dict([(char, index) for index, char in enumerate(self.domain_char_lists)])

    # domain can be consist of lowercase letters, '.', '-' and digits
    # we need to transfer some abnormal domains to normal domains
    # we convert a string line to list which contains normal character
    # @param line represent as one line in file
    def normalize(self, line):
        return [c.lower() for c in line if c.lower() in self.domain_char_lists]

    # convert domain to n-gram list
    # @param n represent as n in n-gram model, for instance, n = 2, 3, 4, 5......
    # @param line represent as one line in file
    def n_gram(self, n, line):
        normalize_line = self.normalize(line)
        for i in range(0, len(normalize_line)-n+1):
            yield ''.join(normalize_line[i: i+n])

    # Markov chain can solve some pronounceable domains problem
    # here we use markov chain model to determine whether a domain
    # can be read out in human language
    # @param line represent as one line in a file
    # @param mm_matrix represent as markov chain matrix
    def avg_transition_prob(self, line, mm_matrix):
        base_prob = 0.0
        transition_ct = 0
        for a, b in self.n_gram(2, line):
            base_prob += mm_matrix[self.pos[a]][self.pos[b]]
            transition_ct += 1
        return math.exp(base_prob / (transition_ct or 1))

    # train markov chain model
    # @param train_set represent as corpus
    # @param good_set represent as alexa top 10000
    # @param bad_set represent as nonexist domains
    # @param train_matrix represent as markov model matrix which will be saved in local disk
    # @param mm_model represent as markov model which will be saved in local disk
    def train(self, train_set, good_set, bad_set, train_matrix, mm_model):
        k = len(self.domain_char_lists)
        # Assume we have seen 10 of each character pair.  This acts as a kind of
        # prior or smoothing factor.  This way, if we see a character transition
        # live that we've never observed in the past, we won't assume the entire
        # string has 0 probability.
        mm_matrix = [[10 for i in xrange(k)] for i in xrange(k)]
        # load corpus to train hmm matrix
        for line in open(train_set):
            for a, b in self.n_gram(2, line):
                mm_matrix[self.pos[a]][self.pos[b]] += 1

        # Normalize the mm_matrix so that they become log probabilities.
        # I use log probabilities rather than straight probabilities to avoid
        # numeric underflow issues with long texts.
        for i, row in enumerate(mm_matrix):
            r_sum = float(sum(row))
            for j in xrange(len(row)):
                row[j] = math.log(row[j]/r_sum)

        # Find the probability of generating a few arbitrarily choosen good and bad phrases.
        good_probs = [self.avg_transition_prob(line, mm_matrix) for line in open(good_set)]
        bad_probs = [self.avg_transition_prob(line, mm_matrix) for line in open(bad_set)]

        # save markov chain matrix to local disk
        writer = open(train_matrix, 'w')
        for i in xrange(k):
            for j in xrange(k):
                writer.write(str(mm_matrix[i][j]) + ' ')
            writer.write('\n')
        writer.close()

        thresh = (min(good_probs) + max(bad_probs)) / 4.0
        print thresh
        pickle.dump({'mat': mm_matrix, 'thresh': thresh},
                    open(mm_model, 'wb'))

if __name__ == '__main__':
    ccDomainDetector = CCDomainDetector()
    train_set = "D:\\Users\\Michael\\PycharmProjects\\MLLearning\\datasets\\big.txt"
    good_set = "D:\\Users\\Michael\\PycharmProjects\\MLLearning\\datasets\\good.txt"
    bad_set = "D:\\Users\\Michael\\PycharmProjects\\MLLearning\\datasets\\bad.txt"
    train_matrix = "D:\\Users\\Michael\\PycharmProjects\\MLLearning\\ouput\\ccdomains\\train_matrix.txt"
    mm_model = "D:\\Users\\Michael\\PycharmProjects\\MLLearning\\ouput\\ccdomains\\cc_domains_model.pki"
    ccDomainDetector.train(train_set, good_set, bad_set, train_matrix, mm_model)
    model_data = pickle.load(open(mm_model, 'rb'))
    print "======================================"

    model_mat = model_data['mat']
    threshold = model_data['thresh']
    # for l in open('E:\\Download\\Gibberish-Detector-master\\whiteTop1w_MainDomain_phishing.phishing'):

    number = 0
    count = 0
    for line in open(bad_set):
        ext = tldextract.extract(line)
        if ccDomainDetector.avg_transition_prob(ext.domain, model_mat) < threshold:
            if len(ext.domain) > 9 and line.find('gov.cn') == -1:
                count += 1
                print line
        number += 1
    print count, number




